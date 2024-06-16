## **Llama2-chat**


Parameters description:
```
    inputs: str
```
- **inputs** (`torch.Tensor` of varying shape depending on the modality, *optional*) — The sequence used as a prompt for the generation or as model inputs to the encoder. If `None` the method initializes it with `bos_token_id` and a batch size of 1. For decoder-only models `inputs` should of in the format of `input_ids`. For encoder-decoder models *inputs* can represent any of `input_ids`, `input_values`, `input_features`, or `pixel_values`.

