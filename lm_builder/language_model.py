import time

import torch
import torch.nn.functional as F

from .transformer import Transformer, TransformerConfig
from tqdm import tqdm


class LanguageModel(Transformer):
    def __init__(self, config: TransformerConfig, tokenizer):
        super().__init__(config)
        self.tokenizer = tokenizer

    @torch.inference_mode()
    def generate(
        self,
        input_ids,
        output_only=False,
        top_k=None,
        max_new_tokens=20,
        temperature=1.0,
        stream=False,
        **kwargs,
    ):
        assert temperature >= 0, "Temperature must be non-negative"
        full_sequence = input_ids

        iterator = range(max_new_tokens)
        if not stream:
            iterator = tqdm(iterator)

        for _ in iterator:
            if full_sequence.shape[-1] > self.context_length:
                model_input = full_sequence[:, -self.context_length :]
            else:
                model_input = full_sequence

            position_ids = (
                torch.arange(model_input.shape[1], device=model_input.device)
                .unsqueeze(0)
                .repeat(model_input.shape[0], 1)
            )

            logits, _ = self(model_input, position_ids=position_ids)
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

            full_sequence = torch.cat((full_sequence, next_id), dim=1)

            if stream:
                yield next_id

        if not stream:
            if output_only:
                return full_sequence[:, -max_new_tokens:]
            return full_sequence

    def prompt(
        self,
        prompts,
        apply_chat_template=False,
        debug=False,
        device="cpu",
        stream=False,
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

        input_ids = self.tokenizer(
            prompts, return_tensors="pt", padding=True
        ).input_ids.to(device)

        start = time.monotonic()
        output_ids = self.generate(input_ids, stream=stream, **kwargs)
        if stream:
            next_token_ids = output_ids
            output_ids = input_ids

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
        end = time.monotonic()

        if debug:
            print(f"Generation took {end - start:.2f} seconds")

        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)

        return outputs[0] if len(outputs) == 1 else outputs
