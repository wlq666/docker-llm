import argparse
from typing import Optional, Union, List
from starlette.middleware.cors import CORSMiddleware
from pydantic import BaseModel,Field
from fastapi import File, UploadFile,Response,status
from fastapi import FastAPI, File, UploadFile, APIRouter, Depends, Request, status
from typing_extensions import Annotated
import torch,json
import urllib.request
import time
from upload import upload_file
from fastapi.openapi.utils import get_openapi
from mixtral_model import MixtralModel
from upload import upload_file

app = FastAPI(title="Mixtral-8x7B-Instruct-v0.1",version='0.0.1',docs_url=None)


origins = ["*"]
app.add_middleware(
    CORSMiddleware, 
    allow_origins=origins,  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"])  

class Parameters(BaseModel):
    inputs: str = Field(title="Input", description="The input prompt for the model. (Default: '')", default="")
    max_length: Optional[int] = Field(title="max length", default=20, description="The maximum length the generated tokens can have. Corresponds to the length of the input prompt + `max_new_tokens`. Its effect is overridden by `max_new_tokens`, if also set.")
    max_new_tokens: Optional[int] = Field(title="max new tokens", default=512, description="The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt.")
    min_length: Optional[int] = Field(title="min length", default=0, description="The minimum length of the sequence to be generated. Corresponds to the length of the input prompt + `min_new_tokens`. Its effect is overridden by `min_new_tokens`, if also set.")
    min_new_tokens: Optional[int] = Field(title="min new tokens", default=None, description="The minimum numbers of tokens to generate, ignoring the number of tokens in the prompt.")
    early_stopping: Optional[Union[bool, str]] = Field(title="early stopping", default=False, description="Controls the stopping condition for beam-based methods, like beam-search. It accepts the following values: `True`, where the generation stops as soon as there are `num_beams` complete candidates; `False`, where an heuristic is applied and the generation stops when is it very unlikely to find better candidates; `never`, where the beam search procedure only stops when there cannot be better candidates (canonical beam search algorithm).")
    max_time: Optional[float] = Field(title="max time", default=None, description="The maximum amount of time you allow the computation to run for in seconds. generation will still finish he current pass after allocated time has been passed.")
    stop_strings: Optional[Union[str, List[str]]] = Field(title="stop strings", default=None, description="A string or a list of strings that should terminate generation if the model outputs them.")
    do_sample: Optional[bool] = Field(title="do sample", default=True, description="Whether or not to use sampling ; use greedy decoding otherwise.")
    num_beams: Optional[int] = Field(title="num beams", default=1, description="Number of beams for beam search. 1 means no beam search.")
    num_beam_groups: Optional[int] = Field(title="num beam groups", default=1, description="Number of groups to divide `num_beams` into for diversity.")
    penalty_alpha: Optional[float] = Field(title="penalty alpha", default=None, description="The values balance the model confidence and the degeneration penalty in contrastive search decoding.")
    use_cache: Optional[bool] = Field(title="use cache", default=True, description="Whether the model should use the past last key/values attentions (if applicable to the model) to speed up decoding.")
    temperature: Optional[float] = Field(title="temperature", default=0.7, description="The value used to modulate the next token probabilities. Increasing the temperature will make the model answer more creatively.")
    top_k: Optional[int] = Field(title="top k", default=50, description="The number of highest probability vocabulary tokens to keep for top-k-filtering.")
    top_p: Optional[float] = Field(title="top p", default=0.95, description="If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to `top_p` or higher are kept for generation.")
    repetition_penalty: Optional[float] = Field(title="repetition penalty", default=1.0, description="The parameter for repetition penalty. 1.0 means no penalty. See the paper for more details.")
 
class Result(BaseModel):
    output: List[str]=Field(title="output",  description="output")
    
class InputData(BaseModel):
    input: Parameters=Field(title="Input")

parser = argparse.ArgumentParser(description='')
parser.add_argument('--load_in_4bit', action='store_true', help='Load in 4 bit mode')
parser.add_argument('--load_in_8bit', action='store_true', help='Load in 8 bit mode')
args = parser.parse_args()

if args.load_in_4bit & args.load_in_8bit == 1:
    raise ValueError("load_in_4bit and load_in_8bit cannot both be True.")

model_path = "/data/mixtral/models"

model = MixtralModel(model_path, args.load_in_4bit,args.load_in_8bit)


@app.get("/",status_code=status.HTTP_200_OK)
def Root():
    return Response(content='ok')

@app.get("/health-check")
def Healthcheck():
    return Response(content='ok')

@app.post("/predictions",response_model=Result)
def Predict(inputdata: InputData):
    print('+++++++++++++++++++++++++++++++++++++++++++++++++')
    print('1. read inputs ...')
    # print('1. Downloading inputs ...')
    
    # filename, _ = urllib.request.urlretrieve(inputdata.url,'input.json')
    # with open(filename) as file:
    #     input = json.load(file)
        
    print('2. Inferring ... ')
    t1 = time.time()
    output = model.infer(**inputdata.input.dict())
    print('3. Inference time cost:', time.time() - t1)
    
    ret = {
    "output": model.get_tokens(output)
    }
    return ret
    # outfile = "output.json"
    # with open(outfile, 'w') as fw:
    #     fw.write(json.dumps(ret))
    
    # print('4. Uploading outputs...')
    # t2 = time.time()
    # url = upload_file(outfile)
    # print('5. Uploading time cost:', time.time() - t2)
    
    # return {
    # "code": 200,
    # "msg": "success",
    # "content": url 
    # }
@app.get("/docs", include_in_schema=False)
def get_docs():
    return get_openapi(title="Mixtral-8x7B-Instruct-v0.1",version='0.0.1', routes=app.routes)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app=app, host="0.0.0.0", port=5001)
