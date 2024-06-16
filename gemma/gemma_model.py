from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


class GemmaModel:
    def __init__(self, model_path):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, device_map="auto").eval()

    def get_tokens(self, text):
        return [self.tokenizer.decode(tok) for tok in self.tokenizer.encode(text)]

    def infer(self, inputs, **kwargs):
        inputs_data = self.tokenizer(inputs, return_tensors="pt")
        if self.device.type == "cuda":
            inputs_data = inputs_data.to("cuda")
        outputs = self.model.generate(**inputs_data, **kwargs)
        out = self.tokenizer.decode(outputs[0])
        return out.lstrip('<bos>').rstrip('<eos>').split(inputs,1)[-1].lstrip('\n\n')