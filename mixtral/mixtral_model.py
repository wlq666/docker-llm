from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers
import torch


class MixtralModel:
    def __init__(self, model_path, load_in_4bit=False, load_in_8bit=False):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        if load_in_4bit: # int4
            self.pipeline = transformers.pipeline(
                "text-generation",
                model=model_path,
                model_kwargs={"torch_dtype": torch.float16, "load_in_4bit": True, "device_map":"auto"},
            )
        elif load_in_8bit: # int8
            self.pipeline = transformers.pipeline(
                "text-generation",
                model=model_path,
                model_kwargs={"torch_dtype": torch.float16, "load_in_8bit": True, "device_map":"auto"},
            )
        else:   # float16
            self.pipeline = transformers.pipeline(
                "text-generation",
                model=model_path,
                model_kwargs={"torch_dtype": torch.float16,  "device_map":"auto"},
            )
    
    def get_tokens(self, text):
        return [self.tokenizer.decode(tok) for tok in self.tokenizer.encode(text)]
        
    def infer(self, inputs, do_sample=True, temperature=0.7, top_p=0.95, top_k=50, max_new_tokens=512, **kwargs):
        messages = [{"role": "user", "content": inputs}]
        inputs_data = self.pipeline.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        outputs = self.pipeline(inputs_data, max_new_tokens=max_new_tokens, do_sample=do_sample, temperature=temperature, top_k=top_k, top_p=top_p, **kwargs)
        out = outputs[0]["generated_text"].split('[/INST]System Information]',1)
        
        if len(out)>1:
            return out[1]
        else:
            return outputs[0]["generated_text"]

