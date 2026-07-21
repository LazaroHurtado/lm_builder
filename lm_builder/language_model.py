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

        if full_sequence.shape[-1] > self.context_length:
            model_input = full_sequence[:, -self.context_length :]
            model_attention_mask = (
                full_attention_mask[:, -self.context_length :]
                if full_attention_mask is not None
                else None
            )
            if kv_caches is not None:
                for kv_cache in kv_caches:
                    kv_cache.reset()
        elif kv_caches is not None and kv_caches[0].sequence_length:
            model_input = full_sequence[:, -1:]

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
        **kwargs,
    ):
        assert temperature >= 0, "Temperature must be non-negative"
        full_sequence = input_ids
        full_attention_mask = attention_mask
        if full_attention_mask is not None:
            if full_attention_mask.shape != full_sequence.shape:
                raise ValueError("Attention mask must match the input IDs shape.")
            full_attention_mask = full_attention_mask.to(input_ids.device)

        cache_enabled = use_cache and len(self.transformer.blocks) > 0
        kv_caches = (
            [KVCache(self.context_length) for _ in self.transformer.blocks]
            if cache_enabled
            else None
        )

        for _ in range(max_new_tokens):
            model_input, model_attention_mask = self._prepare_generation_inputs(
                full_sequence,
                full_attention_mask,
                kv_caches,
            )

            logits, _ = self(
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
