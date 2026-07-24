import time

import torch
import torch.nn.functional as F

from .inference import KVCache
from .transformer import Transformer


class TextGenerationPipeline:
    def __init__(self, model: Transformer, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def _prepare_generation_inputs(
        self,
        full_sequence,
        full_attention_mask,
        kv_caches,
        has_cached_tokens,
    ):
        sequence_length = full_sequence.size(1)
        decode_from_cache = has_cached_tokens and (
            self.model.config.positional_embedding is None
            or sequence_length <= self.model.context_length
        )

        if decode_from_cache:
            model_input = full_sequence[:, -1:]
        else:
            model_input = full_sequence[:, -self.model.context_length :]

        reindex_positions = not decode_from_cache and (
            sequence_length > self.model.context_length
        )
        if reindex_positions and kv_caches is not None:
            # Absolute embeddings require recomputing retained tokens at new positions.
            for kv_cache in kv_caches:
                kv_cache.reset()

        model_attention_mask = full_attention_mask
        if model_attention_mask is not None:
            model_attention_mask = model_attention_mask[:, -model_input.size(1) :]

        position_ids = self._prepare_position_ids(
            full_sequence,
            full_attention_mask,
            model_input.size(1),
            reindex_positions,
        )
        cache_position = None
        if kv_caches is not None:
            end = model_input.size(1) if reindex_positions else sequence_length
            cache_position = torch.arange(
                end - model_input.size(1),
                end,
                dtype=torch.long,
                device=full_sequence.device,
            )
        return model_input, model_attention_mask, position_ids, cache_position

    @staticmethod
    def _sample_next_token(
        logits,
        temperature,
        top_k,
        top_p,
    ):
        if temperature == 0:
            return torch.argmax(logits, dim=-1, keepdim=True)

        logits = logits / temperature
        if top_k is None and top_p is None:
            return torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)

        if top_k is None:
            candidate_logits, candidate_ids = logits.sort(dim=-1, descending=True)
        else:
            candidate_logits, candidate_ids = torch.topk(
                logits,
                min(top_k, logits.size(-1)),
                dim=-1,
            )

        probabilities = F.softmax(candidate_logits, dim=-1)
        if top_p is not None:
            remove = probabilities.cumsum(dim=-1) - probabilities >= top_p
            probabilities.masked_fill_(remove, 0)

        sampled_index = torch.multinomial(probabilities, num_samples=1)
        return candidate_ids.gather(dim=-1, index=sampled_index)

    @staticmethod
    def _prepare_position_ids(
        full_sequence,
        full_attention_mask,
        input_length,
        reindex_positions,
    ):
        if full_attention_mask is not None:
            position_mask = (
                full_attention_mask[:, -input_length:]
                if reindex_positions
                else full_attention_mask
            )
            position_ids = position_mask.long().cumsum(dim=-1) - 1
            position_ids.masked_fill_(~position_mask.bool(), 0)
            return position_ids[:, -input_length:]

        start = 0 if reindex_positions else full_sequence.size(1) - input_length
        return torch.arange(
            start,
            start + input_length,
            dtype=torch.long,
            device=full_sequence.device,
        ).expand(full_sequence.size(0), -1)

    @torch.inference_mode()
    def generate(
        self,
        input_ids,
        top_k=None,
        top_p=None,
        max_new_tokens=20,
        temperature=1.0,
        attention_mask=None,
        use_cache=True,
        eos_token_id=None,
        **kwargs,
    ):
        assert temperature >= 0, "Temperature must be non-negative"
        if top_p is not None and not 0 < top_p <= 1:
            raise ValueError("top_p must be between 0 and 1.")
        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be positive.")
        full_sequence = input_ids
        full_attention_mask = attention_mask
        if eos_token_id is None and self.tokenizer is not None:
            eos_token_id = self.tokenizer.eos_token_id
        eos_token_ids = (
            torch.as_tensor(eos_token_id, device=input_ids.device).flatten()
            if eos_token_id is not None
            else None
        )
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
                self.model.context_length,
                input_ids.size(1) + max_new_tokens,
            )
            kv_caches = [KVCache(capacity=cache_capacity) for _ in self.model.blocks]

        for generation_step in range(max_new_tokens):
            model_input, model_attention_mask, position_ids, cache_position = (
                self._prepare_generation_inputs(
                    full_sequence,
                    full_attention_mask,
                    kv_caches,
                    kv_caches is not None and generation_step > 0,
                )
            )
            logits, _, _ = self.model(
                model_input,
                attention_mask=model_attention_mask,
                position_ids=position_ids,
                cache_position=cache_position,
                _kv_caches=kv_caches,
            )
            logits = logits[:, -1, :]
            next_id = self._sample_next_token(
                logits,
                temperature,
                top_k,
                top_p,
            )

            if eos_token_ids is not None:
                next_id = torch.where(
                    finished[:, None],
                    eos_token_ids[0],
                    next_id,
                )
                finished |= (next_id == eos_token_ids).any(dim=-1)

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
