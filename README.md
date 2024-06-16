
docker for llm




# Qwen
```
docker build -t docker.aigic.ai/dev/llm:qwen1.5_7b_chat_v0614  -f api/docker_0607/qwen/Dockerfile . 
docker push docker.aigic.ai/dev/llm:qwen1.5_7b_chat_v0614 
```
```
docker pull docker.aigic.ai/dev/llm:qwen1.5_7b_chat_v0614 
docker run -itd --gpus all --name foo_qwen -p 5001:5001 docker.aigic.ai/dev/llm:qwen1.5_7b_chat_v0614 
```

# Gemma
```
docker build -t docker.aigic.ai/dev/llm:gemma_7b_v0615 -f api/docker_0607/gemma/Dockerfile . 
docker push docker.aigic.ai/dev/llm:gemma_7b_v0615 
```
```
docker pull docker.aigic.ai/dev/llm:gemma_7b_v0615 
docker run -itd --gpus all --name foo_gemma -p 5001:5001 docker.aigic.ai/dev/llm:gemma_7b_v0615 
```

# LLama2
```
docker build -t docker.aigic.ai/dev/llm:llama2_7b_v0615 -f api/docker_0607/llama2/Dockerfile . 
docker push docker.aigic.ai/dev/llm:llama2_7b_v0615 
```
```
docker pull docker.aigic.ai/dev/llm:llama2_7b_v0615 
docker run -itd --gpus all --name foo_llama2 -p 5001:5001 docker.aigic.ai/dev/llm:llama2_7b_v0615 
```

# Mixtral
```
docker build -t docker.aigic.ai/dev/llm:mixtral_8x7b_v0615 -f api/docker_0607/mixtral/Dockerfile . 
docker push docker.aigic.ai/dev/llm:mixtral_8x7b_v0615 
```
```
docker pull docker.aigic.ai/dev/llm:mixtral_8x7b_v0615 
```
```
docker run -itd --gpus all --name foo_mixtral -p 5001:5001 docker.aigic.ai/dev/llm:mixtral_8x7b_v0615
docker run -itd --gpus all --name foo_mixtral -p 5001:5001 docker.aigic.ai/dev/llm:mixtral_8x7b_v0615 --load_in_4bit
docker run -itd --gpus all --name foo_mixtral -p 5001:5001 docker.aigic.ai/dev/llm:mixtral_8x7b_v0615 --load_in_8bit
```


# EmotiVoice
```
docker pull docker.aigic.ai/dev/llm:EmotiVoice_0613
docker run -itd --gpus all --name foo_llama2 -p 5001:5001 docker.aigic.ai/dev/llm:EmotiVoice_0613 
```
