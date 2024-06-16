FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel

COPY api/docker_0607/qwen /data/qwen
COPY llm_model/hugging_hub_download/Qwen/Qwen1.5-7B-Chat /data/qwen/models

WORKDIR /data/qwen/

RUN apt-get update -y && apt upgrade -y && apt install -y --no-install-recommends vim wget git && \
    apt autoremove && apt autoclean && \
    pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple 
EXPOSE 8007
ENTRYPOINT ["python", "main_qwen_api.py"]