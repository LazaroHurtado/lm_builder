## Description
This project is dedicated to decreasing the development time for language models (LMs). Most LMs share the same building blocks since they are based on the transformer architecture; several layers/blocks of an attention mechanism and feed forward layer. These building blocks usually differ in minor ways; grouped-query attention, mixture of experts, positional embedding. Thus, a framework dedicated to increase the speed of development for LMs by having prebuilt classes for these varients and making it easier for developers to develop their own is possible.

## Getting started

Begin by creating a virtual environment (venv) and running our gpt2 example file:

```zsh
$ uv sync
$ source ./.venv/bin/activate
$ python3 ./examples/gpt2.py
```

### Sliding-window and global attention

Set `attention` to an ordered list and `attention_ratio` to a quoted,
colon-separated string of at least two positive integers. Each ratio component
controls the corresponding attention type. For example, `"5:2:3"` repeats five
sliding-window layers, two grouped-query layers, and three full
causal-attention layers:

```yaml
attention:
  - SlidingWindowAttention
  - GroupedQueryAttention
  - CausalMultiHeadAttention
attention_config:
  context_length: 16384
  embedding_dimension: 4096
  num_heads: 32
  kv_heads: 8
  window_size: 4096
  attention_ratio: "5:2:3"
```

Without `attention_ratio`, `attention` must be a single attention type.

### Todo

- Add tests
- Attention
    - ~~multi-headed attention~~
    - ~~grouped-query attention~~
    - ~~sliding window attention~~
- Caching
    - ~~kv cache~~
    - chunked kv cache
    - rolling buffer caching
- ~~Mixture of experts~~ (to be verified)
- Positional embedding
    - ~~absolute p.e.~~
    - ~~rotary p.e.~~
- Examples
    - ~~gpt2~~
    - ~~llama 1/2~~
    - mistral
    - mixtral