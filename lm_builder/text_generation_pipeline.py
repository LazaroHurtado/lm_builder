import time

import torch
import torch.nn.functional as F

from .transformer import Transformer
from .utils import is_positive_integer


class TextGenerationPipeline:
    def __init__(self, model: Transformer, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    @staticmethod
    def _sample_next_token(logits, temperature, top_k, top_p):
        if temperature == 0:
            return torch.argmax(logits, dim=-1, keepdim=True)

        logits = logits / temperature
        if top_k is None and top_p is None:
            return torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)

        if top_k is None:
            candidate_logits, candidate_ids = logits.sort(dim=-1, descending=True)
        else:
            candidate_logits, candidate_ids = torch.topk(
                logits, min(top_k, logits.size(-1)), dim=-1
            )

        probabilities = F.softmax(candidate_logits, dim=-1)
        if top_p is not None:
            remove = probabilities.cumsum(dim=-1) - probabilities >= top_p
            probabilities.masked_fill_(remove, 0)

        sampled_index = torch.multinomial(probabilities, num_samples=1)
        return candidate_ids.gather(dim=-1, index=sampled_index)

    @torch.no_grad()
    def generate(
        self,
        input_ids,
        top_k=None,
        top_p=None,
        max_new_tokens=20,
        temperature=1.0,
        attention_mask=None,
        eos_token_id=None,
    ):
        if not temperature >= 0:
            raise ValueError("Temperature must be non-negative.")
        if top_p is not None and not 0 < top_p <= 1:
            raise ValueError("top_p must be between 0 and 1.")
        if top_k is not None and not is_positive_integer(top_k):
            raise ValueError("top_k must be a positive integer.")
        if (
            not isinstance(max_new_tokens, int)
            or isinstance(max_new_tokens, bool)
            or max_new_tokens < 0
        ):
            raise ValueError("max_new_tokens must be a non-negative integer.")
        if input_ids.ndim != 2 or input_ids.size(1) == 0:
            raise ValueError("Input IDs must have a non-empty 2D sequence shape.")

        full_sequence = input_ids
        if attention_mask is not None:
            if attention_mask.shape != input_ids.shape:
                raise ValueError("Attention mask must match the input IDs shape.")
            attention_mask = attention_mask.to(input_ids.device)

        if eos_token_id is None and self.tokenizer is not None:
            eos_token_id = self.tokenizer.eos_token_id
        eos_token_ids = (
            torch.as_tensor(eos_token_id, device=input_ids.device).flatten()
            if eos_token_id is not None
            else None
        )
        if eos_token_ids is not None and eos_token_ids.numel() == 0:
            raise ValueError("eos_token_id must contain at least one token ID.")
        finished = torch.zeros(
            input_ids.size(0), dtype=torch.bool, device=input_ids.device
        )

        for _ in range(max_new_tokens):
            model_input = full_sequence[:, -self.model.context_length :]
            model_attention_mask = (
                attention_mask[:, -model_input.size(1) :]
                if attention_mask is not None
                else None
            )
            logits, _, _ = self.model(model_input, attention_mask=model_attention_mask)
            next_id = self._sample_next_token(
                logits[:, -1, :], temperature, top_k, top_p
            )

            if eos_token_ids is not None:
                next_id = torch.where(finished[:, None], eos_token_ids[0], next_id)
                finished |= (next_id == eos_token_ids).any(dim=-1)

            full_sequence = torch.cat((full_sequence, next_id), dim=1)
            if attention_mask is not None:
                attention_mask = torch.cat(
                    (
                        attention_mask,
                        attention_mask.new_ones((next_id.size(0), 1)),
                    ),
                    dim=1,
                )

            yield next_id
            if eos_token_ids is not None and finished.all():
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
            input_ids, attention_mask=attention_mask, **kwargs
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
            output_ids = torch.cat([output_ids, *next_token_ids], dim=1)

        if debug:
            print(f"Generation took {time.monotonic() - start:.2f} seconds")

        outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        return outputs[0] if len(outputs) == 1 else outputs
