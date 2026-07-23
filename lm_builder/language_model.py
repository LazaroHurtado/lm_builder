import time

import torch
import torch.nn.functional as F

from .inference import KVCache
from .transformer import Transformer, TransformerConfig


class LanguageModel(Transformer):
    def __init__(self, config: TransformerConfig, tokenizer):
        super().__init__(config)
        self.tokenizer = tokenizer

    def _prepare_generation_inputs(
        self,
        full_sequence,
        full_attention_mask,
        kv_caches,
    ):
        model_input = full_sequence
        model_attention_mask = full_attention_mask

        cache_is_populated = kv_caches is not None and kv_caches[0].sequence_length > 0
        # Absolute input embeddings cannot be reindexed without recomputing the window.
        # For example, if our context length is 4 tokens; A, B, C, D
        # then they will be assigned position ids 0, 1, 2, 3 respectively.
        # Then, when we go to compute the new next token E we are forced to re-use
        # B, C, D with positions ids 1, 2, 3. So to be correct E must take position id 4.
        # The problem is that our positional embedding table is only of size context_length,
        # and id 4 does not exist when our context length is 4, only ids 0, 1, 2, 3 exist.
        # For this reason we cannot roll the kv cache once it exceeds the context_length.
        can_roll_cache = (
            kv_caches is not None and self.config.positional_embedding is None
        )

        if cache_is_populated and (
            can_roll_cache or full_sequence.shape[-1] <= self.context_length
        ):
            model_input = full_sequence[:, -1:]
        elif full_sequence.shape[-1] > self.context_length:
            model_input = full_sequence[:, -self.context_length :]
            if full_attention_mask is not None and not can_roll_cache:
                model_attention_mask = full_attention_mask[:, -self.context_length :]

            if kv_caches is not None:
                for kv_cache in kv_caches:
                    kv_cache.reset()

        return model_input, model_attention_mask

    @torch.inference_mode()
    def generate(
        self,
        input_ids,
        top_k=None,
        max_new_tokens=20,
        temperature=1.0,
        attention_mask=None,
        use_cache=True,
        eos_token_id=None,
        **kwargs,
    ):
        assert temperature >= 0, "Temperature must be non-negative"
        full_sequence = input_ids
        full_attention_mask = attention_mask
        if eos_token_id is None and self.tokenizer is not None:
            eos_token_id = self.tokenizer.eos_token_id
        finished = torch.zeros(
            input_ids.size(0),
            dtype=torch.bool,
            device=input_ids.device,
        )

        if full_attention_mask is not None:
            if full_attention_mask.shape != full_sequence.shape:
                raise ValueError("Attention mask must match the input IDs shape.")
            full_attention_mask = full_attention_mask.to(input_ids.device)

        kv_caches = None
        if use_cache:
            cache_capacity = min(
                self.context_length,
                input_ids.size(1) + max_new_tokens,
            )
            kv_caches = [KVCache(capacity=cache_capacity) for _ in self.blocks]

        for _ in range(max_new_tokens):
            model_input, model_attention_mask = self._prepare_generation_inputs(
                full_sequence,
                full_attention_mask,
                kv_caches,
            )

            logits, _, _ = self(
                model_input,
                attention_mask=model_attention_mask,
                _kv_caches=kv_caches,
            )
            logits = logits[:, -1, :]

            if temperature > 0:
                logits = logits / temperature

                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1)
                    logits[logits < v[:, [-1]]] = float("-inf")

                probs = F.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
            else:
                next_id = torch.argmax(logits, dim=-1, keepdim=True)

            if eos_token_id is not None:
                next_id = torch.where(
                    finished[:, None],
                    torch.full_like(next_id, eos_token_id),
                    next_id,
                )
                finished |= next_id.squeeze(-1) == eos_token_id

            full_sequence = torch.cat((full_sequence, next_id), dim=1)
            if full_attention_mask is not None:
                full_attention_mask = torch.cat(
                    (
                        full_attention_mask,
                        full_attention_mask.new_ones((next_id.size(0), 1)),
                    ),
                    dim=1,
                )

            yield next_id
            if eos_token_id is not None and finished.all():
                break

    def prompt(
        self,
        prompts,
        apply_chat_template=False,
        stream=False,
        debug=False,
        device="cpu",
        **kwargs,
    ):
        if isinstance(prompts, str):
            prompts = [prompts]

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        if apply_chat_template:
            prompts = self.tokenizer.apply_chat_template(
                prompts, add_generation_prompt=True, tokenize=False
            )

        tokenized_prompts = self.tokenizer(prompts, return_tensors="pt", padding=True)
        input_ids = tokenized_prompts.input_ids.to(device)
        attention_mask = tokenized_prompts.attention_mask.to(device)

        start = time.monotonic()
        next_token_ids = self.generate(
            input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )
        output_ids = input_ids
        if stream:
            previous_text = self.tokenizer.decode(
                output_ids[0], skip_special_tokens=True
            )
            for next_id in next_token_ids:
                output_ids = torch.cat((output_ids, next_id), dim=1)
                current_text = self.tokenizer.decode(
                    output_ids[0], skip_special_tokens=True
                )
                print(current_text[len(previous_text) :], end="", flush=True)
                previous_text = current_text
            print()
        else:
            output_ids = torch.cat(
                [output_ids, torch.cat(list(next_token_ids), dim=1)], dim=1
            )
        end = time.monotonic()

        if debug:
            print(f"Generation took {end - start:.2f} seconds")

        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)

        return outputs[0] if len(outputs) == 1 else outputs
