from transformers import AutoTokenizer, LlamaForCausalLM
import torch


class Llama2Model:
    def __init__(self, model_path):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = LlamaForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map="auto").eval()
    
    def get_tokens(self, text):
        return [self.tokenizer.decode(tok) for tok in self.tokenizer.encode(text)]
    
    def infer(self, inputs, do_sample=True, temperature=0.6, top_p=0.9, max_new_tokens=4096, **kwargs):
        inputs_data = self.tokenizer(inputs, return_tensors="pt")
        if self.device.type == "cuda":
            inputs_data = inputs_data.to("cuda")
        generate_ids = self.model.generate(inputs_data.input_ids, do_sample = do_sample, temperature = temperature, top_p = top_p, max_new_tokens = max_new_tokens, **kwargs)
        response = self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return response.split(inputs)[1].strip()
    
    
    
