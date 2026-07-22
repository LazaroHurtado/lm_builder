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
- ~~Mixture of experts~~ (to be verified)
- Positional embedding
    - ~~absolute p.e.~~
    - ~~rotary p.e.~~
- Examples
    - ~~gpt2~~
    - ~~llama 1/2~~
    - mistral
    - mixtral