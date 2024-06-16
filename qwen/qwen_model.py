from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


class QwenModel:
    def __init__(self, model_path):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map="auto").eval()

    def get_tokens(self, text):
        return [self.tokenizer.decode(tok) for tok in self.tokenizer.encode(text)]
    
    def infer(self, inputs, do_sample=True, repetition_penalty=1.05, temperature=0.7, top_p=0.8, top_k=20, max_new_tokens=512, **kwargs):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": inputs}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs_data = self.tokenizer([text], return_tensors="pt")
        if self.device.type == "cuda":
            inputs_data = inputs_data.to("cuda")

        generated_ids = self.model.generate(
            inputs_data.input_ids,
            do_sample = do_sample,
            repetition_penalty = repetition_penalty,
            temperature = temperature,
            top_p = top_p,
            top_k = top_k,
            max_new_tokens = max_new_tokens,
            **kwargs
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs_data.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return response