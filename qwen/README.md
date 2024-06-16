## **qwen1.5-7b-chat**


Parameters description:
```
    inputs: str
    max_new_tokens: int = 512
    do_sample: bool = True
    temperature: float = 0.7
    top_k: int = 20
    top_p: float = 0.8
    repetition_penalty: float = 1.05
```

- **inputs** (`torch.Tensor` of varying shape depending on the modality, *optional*) — The sequence used as a prompt for the generation or as model inputs to the encoder. If `None` the method initializes it with `bos_token_id` and a batch size of 1. For decoder-only models `inputs` should of in the format of `input_ids`. For encoder-decoder models *inputs* can represent any of `input_ids`, `input_values`, `input_features`, or `pixel_values`.
- **max_new_tokens** (`int`, *optional*, defaults to None) — The maximum numbers of tokens to generate, ignore the current number of tokens. Use either `max_new_tokens` or `max_length` but not both, they serve the same purpose.
- **do_sample** (`bool`, *optional*, defaults to `False`) — Whether or not to use sampling ; use greedy decoding otherwise.
- **temperature** (`float`, *optional*, defaults to 1.0) — The value used to module the next token probabilities.
- **top_k** (`int`, *optional*, defaults to 50) — The number of highest probability vocabulary tokens to keep for top-k-filtering.
- **top_p** (`float`, *optional*, defaults to 1.0) — If set to float < 1, only the most probable tokens with probabilities that add up to `top_p` or higher are kept for generation.
- **repetition_penalty** (`float`, *optional*, defaults to 1.0) — The parameter for repetition penalty. 1.0 means no penalty. See **[this paper](https://arxiv.org/pdf/1909.05858.pdf)** for more details.