from typing import Optional, Union
from starlette.middleware.cors import CORSMiddleware
from pydantic import BaseModel,Field
from fastapi import File, UploadFile,Response,status
from fastapi import FastAPI, File, UploadFile, APIRouter, Depends, Request, status
from typing_extensions import Annotated
import torch
import sys
import oss2
import random,json
import urllib.request
import time

from whisperX_model import WhisperXModel
from upload import upload_file
from fastapi.openapi.utils import get_openapi

app = FastAPI(title="whisperx",version='0.0.1',docs_url=None)


origins = ["*"]
app.add_middleware(
    CORSMiddleware, 
    allow_origins=origins,  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"])  

class Parameters(BaseModel):
    url: str=Field(title="audio url", type="string", description="audio for asr")
    task: str=Field(title="task type", type="string", default="translate", description="transcribe or translate")
    chunk_length_s: Optional[float]=Field(title="chunk length", type="float", default=0, description="The input length for in each chunk. If `chunk_length_s = 0` then chunking is disabled (default)")
    max_new_tokens: Optional[int]=Field(title="max new tokens", type="int", default=128, description="The maximum numbers of tokens to generate, ignoring the number of tokens in the prompt.")
    return_timestamps: Union[bool, str]=Field(title="return timestamps", default=False, description="Only available for pure CTC models (Wav2Vec2, HuBERT, etc) and the Whisper model. Not available for other sequence-to-sequence models Timestamps can take one of two formats: `word`: the pipeline will return timestamps along the text for every word in the text. Word-level timestamps are predicted through the *dynamic-time warping (DTW)* algorithm, an approximation to word-level timestamps by inspecting the cross-attention weights. `True`: the pipeline will return timestamps along the text for *segments* of words in the text. Note that a segment of text refers to a sequence of one or more words, rather than individual sswords as with word-level timestamps.")

class Result(BaseModel):
    output: dict=Field(title="output",  description="output of asr")
    
class InputData(BaseModel):
    input: Parameters=Field(title="Input")


model_path = "/data/whisperX/models"

model = WhisperXModel(model_path)

@app.get("/",status_code=status.HTTP_200_OK)
def Root():
    return Response()

@app.get("/health-check")
def Healthcheck():
    return Response(content='ok')

@app.post("/predictions",response_model=Result)
def Predict(inputdata: InputData):
    print('+++++++++++++++++++++++++++++++++++++++++++++++++')
    print('1. Downloading inputs ...')

    audio_path,_ = urllib.request.urlretrieve(inputdata.input.url, 'input_sample.'+inputdata.input.url.split(".")[-1])
    del inputdata.input.url
    print('2. Inferring ... ')
    t1 = time.time()
    output = model.infer(audio_path, **inputdata.input.dict())
    print('3. Inference time cost:', time.time() - t1)
    ret = {
    "output": output
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
    return get_openapi(title="whisperx",version='0.0.1', routes=app.routes)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app=app, host="0.0.0.0", port=5001)