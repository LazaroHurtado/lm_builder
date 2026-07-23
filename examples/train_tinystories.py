import math
from functools import partial
from pathlib import Path

import torch
from datasets import load_dataset
from torch import nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm, trange
from transformers import AutoTokenizer

from lm_builder.transformer import Transformer, TransformerConfig

DATASET_NAME = "roneneldan/TinyStories"
TOKENIZER_NAME = "openai-community/gpt2"
CHECKPOINT_PATH = "tinystories_200m_weights.pth"
MODEL_CONFIG_PATH = Path(__file__).with_name("tinystories_200m.yml")

SHUFFLE_BUFFER_SIZE = 10_000
BATCH_SIZE = 16
GRADIENT_ACCUMULATION_STEPS = 4
EPOCHS = 3
LEARNING_RATE = 3e-4
SEED = 42


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_amp_dtype(device):
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def initialize_weights(module):
    if isinstance(module, (nn.Embedding, nn.Linear)):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
    if isinstance(module, nn.Linear) and module.bias is not None:
        nn.init.zeros_(module.bias)


def build_model(tokenizer):
    config = TransformerConfig.from_yml(MODEL_CONFIG_PATH)
    if config.vocab_size != len(tokenizer):
        raise ValueError(
            f"{MODEL_CONFIG_PATH} has vocab_size={config.vocab_size}, "
            f"but {TOKENIZER_NAME} has {len(tokenizer)} tokens."
        )

    model = Transformer(config)
    model.apply(initialize_weights)

    residual_std = 0.02 / math.sqrt(2 * config.num_layers)
    for name, parameter in model.named_parameters():
        if name.endswith(("attn.out_proj.weight", "ffn.down_proj.weight")):
            nn.init.normal_(parameter, mean=0.0, std=residual_std)

    return model


def collate_stories(stories, tokenizer, context_length):
    encoded = tokenizer(
        [story["text"] + tokenizer.eos_token for story in stories],
        max_length=context_length + 1,
        padding=True,
        return_tensors="pt",
        truncation=True,
    )
    targets = encoded.input_ids[:, 1:].clone()
    targets.masked_fill_(encoded.attention_mask[:, 1:] == 0, -1)

    return {
        "input_ids": encoded.input_ids[:, :-1],
        "attention_mask": encoded.attention_mask[:, :-1],
        "targets": targets,
    }


def build_dataloader(stories, tokenizer, context_length, epoch, pin_memory):
    epoch_stories = stories.shuffle(
        seed=SEED + epoch - 1,
        buffer_size=SHUFFLE_BUFFER_SIZE,
    )
    return DataLoader(
        epoch_stories,
        batch_size=BATCH_SIZE,
        collate_fn=partial(
            collate_stories,
            tokenizer=tokenizer,
            context_length=context_length,
        ),
        pin_memory=pin_memory,
    )


def train_epoch(
    model,
    tokenizer,
    stories,
    optimizer,
    scaler,
    epoch,
):  # pylint: disable=too-many-locals
    device = next(model.parameters()).device
    amp_dtype = get_amp_dtype(device)
    model.train()
    running_loss = 0.0
    total_batches = math.ceil(stories.info.splits["train"].num_examples / BATCH_SIZE)
    batch_progress = tqdm(
        build_dataloader(
            stories,
            tokenizer,
            model.context_length,
            epoch,
            pin_memory=device.type == "cuda",
        ),
        total=total_batches,
        desc=f"Epoch {epoch}/{EPOCHS}",
        leave=False,
    )

    average_loss = None
    grad_norm = None
    optimizer_steps = 0
    optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(batch_progress, start=1):
        batch = {
            name: tensor.to(
                device,
                non_blocking=device.type == "cuda",
            )
            for name, tensor in batch.items()
        }

        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=device.type == "cuda",
        ):
            _, loss = model(
                batch["input_ids"],
                targets=batch["targets"],
                attention_mask=batch["attention_mask"],
            )
        if not torch.isfinite(loss).item():
            raise FloatingPointError(
                f"Non-finite loss at epoch {epoch}, training step {step}."
            )

        accumulation_group_start = (
            (step - 1) // GRADIENT_ACCUMULATION_STEPS
        ) * GRADIENT_ACCUMULATION_STEPS
        accumulation_group_size = min(
            GRADIENT_ACCUMULATION_STEPS,
            total_batches - accumulation_group_start,
        )
        scaler.scale(loss / accumulation_group_size).backward()

        should_update = step % GRADIENT_ACCUMULATION_STEPS == 0 or step == total_batches
        if should_update:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
                error_if_nonfinite=True,
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps += 1

        batch_loss = loss.item()
        running_loss += batch_loss
        average_loss = running_loss / step
        batch_progress.set_postfix(
            loss=f"{batch_loss:.3f}",
            avg_loss=f"{average_loss:.3f}",
            perplexity=f"{math.exp(min(average_loss, 20)):.1f}",
            grad_norm=f"{grad_norm.item():.2f}" if grad_norm is not None else "-",
            lr=f"{optimizer.param_groups[0]['lr']:.1e}",
            updates=optimizer_steps,
        )

    if average_loss is None:
        raise RuntimeError(
            "The TinyStories stream did not produce any training batches."
        )
    return average_loss


def train(model, tokenizer, stories, device):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )
    amp_dtype = get_amp_dtype(device)
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=device.type == "cuda" and amp_dtype == torch.float16,
    )

    epoch_progress = trange(1, EPOCHS + 1, desc="Training")
    for epoch in epoch_progress:
        average_loss = train_epoch(
            model,
            tokenizer,
            stories,
            optimizer,
            scaler,
            epoch,
        )

        epoch_progress.set_postfix(
            loss=f"{average_loss:.3f}",
            perplexity=f"{math.exp(min(average_loss, 20)):.1f}",
        )


def main():
    torch.manual_seed(SEED)
    device = get_device()

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    stories = load_dataset(DATASET_NAME, split="train", streaming=True)
    model = build_model(tokenizer).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"Training a {parameter_count / 1e6:.1f}M parameter model on {device}.")
    print(
        f"Physical batch size: {BATCH_SIZE}; "
        f"effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}."
    )

    train(model, tokenizer, stories, device)
    torch.save(model.state_dict(), CHECKPOINT_PATH)
    print(f"Saved weights to {CHECKPOINT_PATH}.")


if __name__ == "__main__":
    main()
