import time
import torch

from .transformer import Transformer, TransformerConfig

from torch.functional import F


class LanguageModel(Transformer):

    def __init__(self, config: TransformerConfig, tokenizer, device="cpu"):
        super().__init__(config)

        self.tokenizer = tokenizer
        self.device = device

    @torch.inference_mode()
    def generate(  # pylint: disable=too-many-arguments
        self,
        input_ids,
        output_only=False,
        top_k=None,
        max_new_tokens=20,
        temperature=1.0,
        **kwargs,
    ):

        for _ in range(max_new_tokens):
            input_context_length = input_ids.shape[-1]
            if input_context_length > self.context_length:
                input_ids = input_ids[:, -self.context_length :]

            logits, _ = self(input_ids)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, top_k, dim=-1)
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)

            next_id = torch.multinomial(probs, num_samples=1)

            input_ids = torch.cat((input_ids, next_id), dim=1)

        if output_only:
            return input_ids[:, -max_new_tokens:]
        return input_ids

    def prompt(self, prompt, **kwargs):
        input_ids = self.tokenizer.encode(prompt)
        input_ids = torch.tensor(
            input_ids, dtype=torch.long, device=self.device
        ).unsqueeze(0)

        start = time.monotonic()
        output = self.generate(input_ids, **kwargs)
        end = time.monotonic()

        output = self.tokenizer.decode(output.flatten().tolist())
        elapsed_time = end - start

        print(f"Output: {output}")
        print(f"Response time: {elapsed_time:.2f}s")
