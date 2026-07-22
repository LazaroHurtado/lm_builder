## Description
This project is dedicated to decreasing the development time for language models (LMs). Most LMs share the same building blocks since they are based on the transformer architecture; several layers/blocks of an attention mechanism and feed forward layer. These building blocks usually differ in minor ways; grouped-query attention, mixture of experts, positional embedding. Thus, a framework dedicated to increase the speed of development for LMs by having prebuilt classes for these varients and making it easier for developers to develop their own is possible.

## Getting started

Begin by creating a virtual environment (venv) and running our gpt2 example file:

```zsh
$ uv sync
$ source ./.venv/bin/activate
$ python3 ./examples/gpt2.py
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
schema through `AttentionConfig.from_yml()` returns one independent, fully
resolved `AttentionConfig` for each transformer layer.

`norm` configures residual-stream normalization before attention. Optional
`qk_norm` builds independent query and key normalizers over each head's
`head_dim`. They run after head shaping and before positional embeddings, KV
caching, and GQA/MQA key-value head sharing. Values are not normalized. Omit
`qk_norm` to preserve the original attention behavior and parameter layout.

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

### KV-cached generation

`LanguageModel.generate()` and `LanguageModel.prompt()` use an inference-only KV
cache by default for causal attention models. The prompt is evaluated once, then
only the newest token is evaluated. The model must be in evaluation mode. Set
`use_cache=False` to use full-sequence recomputation instead:

```python
output = model.prompt("Hello", use_cache=False)
```

For rotary or no positional embeddings, the cache evicts its oldest key/value
entry after `context_length` and continues single-token decoding. Retained cached
states preserve information from tokens that have since been evicted, so output
after overflow can differ from uncached sliding-window recomputation.

Models with absolute input positional embeddings reset the cache and recompute
the current context window after overflow because cached states cannot be safely
reindexed.

### Todo

- Add tests
- Attention
    - ~~multi-headed attention~~
    - ~~grouped-query attention~~
    - ~~sliding window attention~~
- Caching
    - ~~kv cache~~
    - chunked kv cache
    - ~~rolling buffer caching after context overflow~~
- ~~Mixture of experts~~
- Positional embedding
    - ~~absolute p.e.~~
    - ~~rotary p.e.~~
- Examples
    - ~~gpt2~~
    - ~~llama 1/2~~
    - mistral
    - mixtral