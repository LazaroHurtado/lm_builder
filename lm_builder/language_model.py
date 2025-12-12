import time

import torch
import torch.nn.functional as F

from .transformer import Transformer, TransformerConfig


class LanguageModel(Transformer):
    def __init__(
        self,
        config: TransformerConfig,
        tokenizer,
        device="cpu",
    ):
        super().__init__(config)
        self.tokenizer = tokenizer
        self.device = device

    @torch.inference_mode()
    def generate(
        self,
        input_ids,
        output_only=False,
        top_k=None,
        max_new_tokens=20,
        temperature=1.0,
        **kwargs,
    ):
        assert temperature >= 0, "Temperature must be non-negative"
        full_sequence = input_ids

        for _ in range(max_new_tokens):
            if full_sequence.shape[-1] > self.context_length:
                model_input = full_sequence[:, -self.context_length :]
            else:
                model_input = full_sequence

            logits, _ = self(model_input)
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

        if output_only:
            return full_sequence[:, -max_new_tokens:]
        return full_sequence

    def prompt(self, prompts, apply_chat_template=False, debug=False, **kwargs):
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
        ).input_ids.to(self.device)

        start = time.monotonic()
        output_ids = self.generate(input_ids, **kwargs)
        end = time.monotonic()

        if debug:
            print(f"Generation took {end - start:.2f} seconds")

        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)

        return outputs[0] if len(outputs) == 1 else outputs
