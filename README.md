## Description
This project provides reusable transformer building blocks for training language
models, including grouped-query attention, mixture-of-experts layers, and
positional embeddings.

## Getting started

Install the dependencies and run the TinyStories training example:

```zsh
$ uv sync
$ source ./.venv/bin/activate
$ python3 ./examples/train_tinystories.py
```

The GPT-2, Llama 2, and Qwen 3 example scripts convert pretrained checkpoints
to this framework's parameter names and generate text.

### Example text generation

`TextGenerationPipeline` supports prompts, chat templates, streaming, temperature,
top-k and top-p sampling, and EOS stopping. Generation recomputes the full context
for each new token, cropping inputs and padding masks to `model.context_length`.
The model's training forward path remains stateless.

```python
from lm_builder import TextGenerationPipeline

model.eval()
pipeline = TextGenerationPipeline(model, tokenizer)
output = pipeline.prompt("Hello", max_new_tokens=20, temperature=0)
```

### Attention layer patterns

Place attention implementations and their overrides in `attention_config.layers`.
Every non-container option on `attention_config` is inherited as a shared
default, and every layer must declare its own `type`. A `ratio` is required when
there is more than one entry; it contains one positive integer per entry:

```yaml
context_length: 16384
embedding_dimension: 4096
attention_config:
  qk_positional_embedding:
    type: RotaryPE
    base: 10000.0
  num_heads: 32
  norm:
    type: RMSNorm
    eps: 1.0e-5
  qk_norm:
    type: RMSNorm
    eps: 1.0e-6
  ratio: [5, 2]
  layers:
    - type: GroupedQueryAttention
      kv_heads: 8
      window_size: 4096
    - type: CausalMultiHeadAttention
      window_size: null
```

This repeats five windowed grouped-query layers followed by two full causal
layers. `num_layers` must be divisible by the sum of `ratio`. A single entry
does not require `ratio`. `TransformerConfig` injects the top-level
`context_length` and `embedding_dimension` into every resolved attention config,
and injects `embedding_dimension` into the feed-forward config. Loading this
schema through `TransformerConfig.from_yml()` returns a `TransformerConfig` whose
`attention_config` holds one `PositionalEmbeddingConfig` and an independent
`AttentionLayerConfig` for every transformer layer. Positional embedding fields
other than `type` are stored
as constructor keyword arguments. `Transformer` uses that config to build one
shared positional embedding instance for the model.

`norm` configures residual-stream normalization before attention. Optional
`qk_norm` builds independent query and key normalizers over each head's
`head_dim`. They run after head shaping and before positional embeddings and
GQA/MQA key-value head sharing. Values are not normalized. Omit
`qk_norm` to preserve the original attention behavior and parameter layout.

Q/K positional embeddings use a two-phase interface. `Transformer.forward()`
calls `prepare()` once to create position data shared by the full layer stack.
The configured embedding is bound to each attention module during construction,
so layer `forward()` calls receive only that prepared position data. Each layer
then calls `apply_qk()` to transform its own query and key tensors. RoPE therefore
computes its cosines and sines once per model forward without storing mutable
position tables.

### Tied input and output embeddings

Set `tie_word_embeddings: true` at the model level to share the token embedding
weight with the language-model output projection:

```yaml
embedding_dimension: 4096
vocab_size: 32000
tie_word_embeddings: true
```

Both modules reference the same `Parameter`, so gradients and optimizer updates
remain shared. The LM-head bias, when enabled, remains independent. The option
defaults to `false`.

### Mixture-of-experts feed-forward layers

Select `MixtureOfExperts` as the feed-forward type and configure one or more
experts per token:

```yaml
ffn_config:
  type: MixtureOfExperts
  intermediate_dimension: 14336
  activation_fn: SiLU
  num_experts: 8
  top_k: 2
  num_shared_experts: 1
```

`num_experts` must be positive and `top_k` must be between `1` and
`num_experts`. `num_shared_experts` defaults to `0` and must be a non-negative
integer. When enabled, the MoE adds one always-active feed-forward network whose
intermediate width is `intermediate_dimension * num_shared_experts`. Shared
expert output is added to routed expert output and does not participate in
routing or routing loss.

Transformer forward passes return cross-entropy and the mean routing loss
across MoE layers separately. Cross-entropy is available when targets are
provided. Routing loss is computed in training mode and is `None` in evaluation
mode. Combine the losses with a coefficient chosen for the training run:

```python
logits, cross_entropy_loss, routing_loss = model(input_ids, targets)
loss = cross_entropy_loss
if routing_loss is not None:
    loss = loss + 0.01 * routing_loss
```

Top-1 routing depends on this auxiliary loss because its selected expert weight
is always one and therefore receives no routing gradient from cross-entropy.

`GroupedExperts`, defined alongside `MixtureOfExperts`, owns the stacked
routed-expert parameters. Its forward method selects one expert's assigned
tokens, then evaluates them with ordinary `torch.nn.functional.linear` calls.
`MixtureOfExperts` loops over the configured experts and combines those outputs.
This remains computationally sparse: it executes exactly `tokens * top_k`
assignments rather than evaluating every expert for every token. It supports the
dtypes, dimensions, and devices supported by the underlying linear operations.

Per-expert token selection uses `torch.nonzero`, whose output size depends on
routing decisions. Enable dynamic-output capture when requesting a single full
graph:

```python
import torch

torch._dynamo.config.capture_dynamic_output_shape_ops = True
compiled_model = torch.compile(model, fullgraph=True)
```

This portable path avoids a specialized grouped-matrix-multiplication kernel,
but dynamic `nonzero` operations may introduce synchronization or reduce
compiler performance. Routed activation modules must remain stateless.

### Todo

- Add tests
- Attention
    - ~~multi-headed attention~~
    - ~~grouped-query attention~~
    - ~~sliding window attention~~
- ~~Mixture of experts~~
- Positional embedding
    - ~~absolute p.e.~~
    - ~~rotary p.e.~~
- Examples
    - ~~gpt2~~
    - ~~llama 1/2~~
    - mistral
    - mixtral