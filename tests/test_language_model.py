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
        self.eos_token_id = None
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
        return logits, None, None


class ScriptedLanguageModel(LanguageModel):
    def __init__(self, config, tokenizer, generated_token_ids):
        super().__init__(config, tokenizer)
        self.generated_token_ids = generated_token_ids
        self.forward_calls = 0

    def forward(
        self,
        x,
        targets=None,
        attention_mask=None,
        position_ids=None,
        *,
        _kv_caches=None,
    ):
        next_token_ids = torch.tensor(
            self.generated_token_ids[self.forward_calls],
            device=x.device,
        )
        logits = torch.full(
            (*x.shape, self.config.vocab_size),
            float("-inf"),
            device=x.device,
        )
        logits[:, -1].scatter_(1, next_token_ids[:, None], 0)
        self.forward_calls += 1
        return logits, None, None


def build_scripted_language_model(generated_token_ids, tokenizer=None):
    return ScriptedLanguageModel(
        TransformerConfig(
            embedding_dimension=8,
            context_length=8,
            attention_config=[
                AttentionConfig(
                    context_length=8,
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
        generated_token_ids,
    ).eval()


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


def test_generate_stops_after_tokenizer_eos():
    tokenizer = SimpleNamespace(eos_token_id=2)
    model = build_scripted_language_model([[2], [4]], tokenizer)

    generated = list(
        model.generate(
            torch.tensor([[1]]),
            max_new_tokens=5,
            temperature=0,
            use_cache=False,
        )
    )

    assert torch.equal(torch.cat(generated, dim=1), torch.tensor([[2]]))
    assert model.forward_calls == 1


def test_generate_without_eos_uses_max_new_tokens():
    model = build_scripted_language_model([[3], [4], [5]])

    generated = list(
        model.generate(
            torch.tensor([[1]]),
            max_new_tokens=3,
            temperature=0,
            use_cache=False,
        )
    )

    assert torch.equal(torch.cat(generated, dim=1), torch.tensor([[3, 4, 5]]))
    assert model.forward_calls == 3


def test_generate_stops_after_secondary_eos_token_id():
    model = build_scripted_language_model([[3], [4]])

    generated = list(
        model.generate(
            torch.tensor([[1]]),
            max_new_tokens=5,
            temperature=0,
            use_cache=False,
            eos_token_id=[2, 3],
        )
    )

    assert torch.equal(torch.cat(generated, dim=1), torch.tensor([[3]]))
    assert model.forward_calls == 1


def test_generate_repeats_eos_for_finished_batch_rows():
    model = build_scripted_language_model(
        [
            [2, 4],
            [7, 5],
            [8, 2],
            [9, 9],
        ],
    )

    generated = torch.cat(
        list(
            model.generate(
                torch.tensor([[1, 1], [1, 1]]),
                max_new_tokens=5,
                temperature=0,
                use_cache=False,
                eos_token_id=2,
            )
        ),
        dim=1,
    )

    assert torch.equal(
        generated,
        torch.tensor(
            [
                [2, 2, 2],
                [4, 5, 2],
            ]
        ),
    )
    assert model.forward_calls == 3
