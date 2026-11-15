from pathlib import Path
import runpy
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers import AutoTokenizer, GenerationConfig

from lm_builder import TextGenerationPipeline
from lm_builder.attention import (
    AttentionConfig,
    AttentionLayerConfig,
    CausalMultiHeadAttention,
)
from lm_builder.ffn import FeedForward, FeedForwardConfig
from lm_builder.positional_embeddings import (
    AbsolutePE,
    LearnablePE,
    PositionalEmbeddingConfig,
    RotaryPE,
)
from lm_builder.transformer import Transformer, TransformerConfig


def build_config(context_length=4, position_type=None):
    return TransformerConfig(
        embedding_dimension=8,
        context_length=context_length,
        attention_config=AttentionConfig(
            qk_positional_embedding=(
                PositionalEmbeddingConfig(positional_embedding_type=RotaryPE)
                if position_type is RotaryPE
                else None
            ),
            layers=[
                AttentionLayerConfig(
                    context_length=context_length,
                    embedding_dimension=8,
                    num_heads=2,
                    attention_type=CausalMultiHeadAttention,
                )
            ],
        ),
        ffn_config=FeedForwardConfig(
            embedding_dimension=8,
            intermediate_dimension=16,
            ffn_type=FeedForward,
        ),
        vocab_size=10,
        num_layers=1,
        positional_embedding=None if position_type is RotaryPE else position_type,
    )


class FakeTokenizer:
    pad_token = None
    eos_token = "<eos>"
    eos_token_id = None
    padding_side = None

    def __call__(self, prompts, return_tensors, padding):
        assert return_tensors == "pt"
        assert padding
        return SimpleNamespace(
            input_ids=torch.tensor([[0, 0, 2], [3, 4, 5]]),
            attention_mask=torch.tensor([[0, 0, 1], [1, 1, 1]]),
        )

    def decode(self, output_ids, skip_special_tokens):
        assert skip_special_tokens
        return " ".join(str(token) for token in output_ids.tolist())

    def batch_decode(self, output_ids, skip_special_tokens):
        return [self.decode(row, skip_special_tokens) for row in output_ids]


class ScriptedModel(nn.Module):
    def __init__(self, generated_token_ids, context_length=4):
        super().__init__()
        self.context_length = context_length
        self.generated_token_ids = generated_token_ids
        self.inputs = []
        self.attention_masks = []

    def forward(self, x, attention_mask=None):
        next_token_ids = torch.tensor(
            self.generated_token_ids[len(self.inputs)], device=x.device
        )
        self.inputs.append(x.clone())
        self.attention_masks.append(
            None if attention_mask is None else attention_mask.clone()
        )
        logits = torch.full((*x.shape, 10), float("-inf"), device=x.device)
        logits[:, -1].scatter_(1, next_token_ids[:, None], 0)
        return logits, None, None


@pytest.mark.parametrize("stream", [False, True])
def test_prompt_extends_and_crops_input_and_padding_mask(stream, capsys):
    tokenizer = FakeTokenizer()
    model = ScriptedModel([[1, 1], [1, 1]], context_length=3)
    pipeline = TextGenerationPipeline(model, tokenizer)

    outputs = pipeline.prompt(
        ["short", "longer"], max_new_tokens=2, temperature=0, stream=stream
    )

    assert outputs == ["0 0 2 1 1", "3 4 5 1 1"]
    assert tokenizer.pad_token == tokenizer.eos_token
    assert tokenizer.padding_side == "left"
    assert torch.equal(model.inputs[0], torch.tensor([[0, 0, 2], [3, 4, 5]]))
    assert torch.equal(model.inputs[1], torch.tensor([[0, 2, 1], [4, 5, 1]]))
    assert torch.equal(model.attention_masks[0], torch.tensor([[0, 0, 1], [1, 1, 1]]))
    assert torch.equal(model.attention_masks[1], torch.tensor([[0, 1, 1], [1, 1, 1]]))
    assert capsys.readouterr().out == (" 1 1\n" if stream else "")


@pytest.mark.parametrize(
    ("position_type", "compile_model"),
    [(None, False), (AbsolutePE, False), (LearnablePE, False), (RotaryPE, True)],
)
def test_generation_matches_full_context_recomputation(position_type, compile_model):
    torch.manual_seed(17)
    model = Transformer(build_config(position_type=position_type)).eval()
    input_ids = torch.tensor([[0, 2, 3], [4, 5, 6]])
    attention_mask = torch.tensor([[0, 1, 1], [1, 1, 1]])
    expected_tokens = []
    sequence = input_ids
    mask = attention_mask
    with torch.no_grad():
        for _ in range(5):
            logits, _, _ = model(sequence[:, -4:], attention_mask=mask[:, -4:])
            next_token = logits[:, -1].argmax(dim=-1, keepdim=True)
            expected_tokens.append(next_token)
            sequence = torch.cat((sequence, next_token), dim=1)
            mask = torch.cat((mask, torch.ones(2, 1, dtype=mask.dtype)), dim=1)

    input_lengths = []

    def record_input(_model, inputs):
        assert not torch.is_grad_enabled()
        input_lengths.append(inputs[0].size(1))

    if compile_model:
        model = torch.compile(model, backend="eager", fullgraph=True)
    handle = model.register_forward_pre_hook(record_input)
    pipeline = TextGenerationPipeline(model, tokenizer=None)
    generated = list(
        pipeline.generate(
            input_ids, attention_mask=attention_mask, max_new_tokens=5, temperature=0
        )
    )
    handle.remove()

    assert input_lengths == [3, 4, 4, 4, 4]
    torch.testing.assert_close(
        torch.cat(generated, dim=1), torch.cat(expected_tokens, dim=1)
    )


def test_top_p_sampling_keeps_the_token_that_crosses_the_threshold(monkeypatch):
    sampled_probabilities = None

    def sample(probabilities, num_samples):
        nonlocal sampled_probabilities
        assert num_samples == 1
        sampled_probabilities = probabilities
        return torch.tensor([[1]])

    monkeypatch.setattr(torch, "multinomial", sample)
    token = TextGenerationPipeline._sample_next_token(
        logits=torch.tensor([[4.0, 3.0, 2.0]]),
        temperature=1.0,
        top_k=3,
        top_p=0.7,
    )

    assert token.item() == 1
    assert sampled_probabilities[0, 0] > 0
    assert sampled_probabilities[0, 1] > 0
    assert sampled_probabilities[0, 2] == 0


@pytest.mark.parametrize(
    ("tokenizer", "eos_token_id", "expected"),
    [
        (SimpleNamespace(eos_token_id=2), None, [3, 2]),
        (None, [2, 3], [3]),
        (None, None, [3, 2, 5]),
    ],
)
def test_generation_stops_at_eos_or_token_limit(tokenizer, eos_token_id, expected):
    model = ScriptedModel([[3], [2], [5]])
    pipeline = TextGenerationPipeline(model, tokenizer)
    generated = list(
        pipeline.generate(
            torch.tensor([[1]]),
            max_new_tokens=3,
            temperature=0,
            eos_token_id=eos_token_id,
        )
    )

    assert torch.equal(torch.cat(generated, dim=1), torch.tensor([expected]))
    assert len(model.inputs) == len(expected)


def test_generate_repeats_eos_for_finished_batch_rows():
    model = ScriptedModel([[2, 4], [7, 5], [8, 2]])
    pipeline = TextGenerationPipeline(model, tokenizer=None)
    generated = list(
        pipeline.generate(
            torch.tensor([[1, 1], [1, 1]]),
            max_new_tokens=5,
            temperature=0,
            eos_token_id=2,
        )
    )

    assert torch.equal(
        torch.cat(generated, dim=1), torch.tensor([[2, 2, 2], [4, 5, 2]])
    )
    assert len(model.inputs) == 3


def test_prompt_with_zero_new_tokens_returns_input():
    pipeline = TextGenerationPipeline(ScriptedModel([]), FakeTokenizer())

    assert pipeline.prompt(["short", "longer"], max_new_tokens=0) == ["0 0 2", "3 4 5"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"temperature": -1},
        {"top_k": 0},
        {"top_p": 0},
        {"max_new_tokens": -1},
        {"attention_mask": torch.ones(1, 2)},
    ],
)
def test_generation_rejects_invalid_inputs(kwargs):
    pipeline = TextGenerationPipeline(ScriptedModel([[1]]), tokenizer=None)

    with pytest.raises(ValueError):
        list(pipeline.generate(torch.tensor([[1]]), **kwargs))


@pytest.mark.parametrize(
    ("example", "weights_file", "max_new_tokens", "chat"),
    [
        ("gpt2.py", "gpt2_weights.pth", 200, False),
        ("llama2.py", "llama2_weights.pth", 100, True),
        ("qwen3.py", "qwen3_0_6b_weights.pth", 4096, True),
    ],
)
def test_example_entry_point_runs_generation(
    example, weights_file, max_new_tokens, chat, tmp_path, monkeypatch
):
    example_path = Path(__file__).resolve().parents[1] / "examples" / example
    torch.save(Transformer(build_config()).state_dict(), tmp_path / weights_file)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(TransformerConfig, "from_yml", lambda _path: build_config())
    monkeypatch.setattr(torch, "compile", lambda model, **_kwargs: model)
    monkeypatch.setattr(
        AutoTokenizer, "from_pretrained", lambda *_args, **_kwargs: FakeTokenizer()
    )
    monkeypatch.setattr(
        GenerationConfig,
        "from_pretrained",
        lambda *_args, **_kwargs: SimpleNamespace(
            temperature=0.7, top_k=3, top_p=0.9, eos_token_id=[2, 3]
        ),
    )
    calls = []

    def prompt(pipeline, prompts, **kwargs):
        assert not pipeline.model.training
        calls.append((prompts, kwargs))
        return "generated"

    monkeypatch.setattr(TextGenerationPipeline, "prompt", prompt)
    runpy.run_path(str(example_path), run_name="__main__")

    assert len(calls) == 1
    prompts, options = calls[0]
    assert prompts == (
        [{"role": "user", "content": "Who is Claude Shannon?"}]
        if chat
        else "Claude Shannon, the"
    )
    assert options["max_new_tokens"] == max_new_tokens
    assert options.get("apply_chat_template", False) == chat
    assert options["stream"] and options["debug"]
    if example == "qwen3.py":
        assert options["top_k"] == 3
        assert options["top_p"] == 0.9
        assert options["eos_token_id"] == [2, 3]
