## Description
This project is dedicated to increasing the development time for language models (LMs). Most LMs share the same building blocks since they are based on the transformer architecture; several layers of an attention mechanism followed by a feed forward layer. These building blocks usually differ in minor ways; grouped-query attention, mixture of experts, positional embedding. Thus, a framework dedicated to increase the speed of development for LMs by having prebuilt classes for these varients and making it easier for developers to develop their own is possible.

## Getting started
Begin by creating a virtual environment (venv) and running our gpt2 example file:
```zsh
$ make setup
$ source ./lm_builder_venv/bin/activate
$ python3 ./examples/gpt2.py
```
You can then exit out of the venv with `deactivate`. For more make commands please view the `Makefile` file in the root directory.

### Todo
- Add tests 😬
- Attention
    - ~~multi-headed attention~~
    - ~~grouped-query attention~~
    - sliding window attention
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
    - llama 1/2
    - mistral
    - mixtral