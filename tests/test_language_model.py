from types import SimpleNamespace

import torch

from lm_builder import LanguageModel
from lm_builder.attention import (
    AttentionConfig,
    CausalMultiHeadAttention,
)
from lm_builder.ffn import FeedForward, FeedForwardConfig
from lm_builder.transformer import TransformerConfig


class FakeTokenizer:
    def __init__(self):
        self.pad_token = None
        self.eos_token = "<eos>"
        self.padding_side = None

    def __call__(self, prompts, return_tensors, padding):
        assert return_tensors == "pt"
        assert padding
        return SimpleNamespace(
            input_ids=torch.tensor([[0, 0, 2], [3, 4, 5]]),
            attention_mask=torch.tensor([[0, 0, 1], [1, 1, 1]]),
        )

    def batch_decode(self, output_ids, skip_special_tokens):
        assert skip_special_tokens
        return ["decoded"] * output_ids.size(0)


class RecordingLanguageModel(LanguageModel):
    def __init__(self, config, tokenizer):
        super().__init__(config, tokenizer)
        self.attention_masks = []

    def forward(
        self,
        x,
        targets=None,
        attention_mask=None,
        position_ids=None,
        *,
        _kv_caches=None,
    ):
        self.attention_masks.append(attention_mask.clone())
        logits = torch.zeros(
            (*x.shape, self.config.vocab_size),
            device=x.device,
        )
        logits[..., 1] = 1
        return logits, None


def test_prompt_passes_and_extends_tokenizer_attention_mask():
    tokenizer = FakeTokenizer()
    model = RecordingLanguageModel(
        TransformerConfig(
            embedding_dimension=8,
            context_length=3,
            attention_config=[
                AttentionConfig(
                    context_length=3,
                    embedding_dimension=8,
                    num_heads=2,
                    attention_type=CausalMultiHeadAttention,
                )
            ],
            ffn_config=FeedForwardConfig(
                embedding_dimension=8,
                intermediate_dimension=16,
                ffn_type=FeedForward,
            ),
            vocab_size=10,
            num_layers=1,
        ),
        tokenizer,
    )

    outputs = model.prompt(
        ["short", "longer"],
        max_new_tokens=2,
        temperature=0,
        use_cache=False,
    )

    assert outputs == ["decoded", "decoded"]
    assert tokenizer.pad_token == tokenizer.eos_token
    assert tokenizer.padding_side == "left"
    assert torch.equal(
        model.attention_masks[0],
        torch.tensor([[0, 0, 1], [1, 1, 1]]),
    )
    assert torch.equal(
        model.attention_masks[1],
        torch.tensor([[0, 1, 1], [1, 1, 1]]),
    )
