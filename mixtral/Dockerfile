FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel

COPY api/docker_0607/mixtral /data/mixtral

COPY llm_model/hugging_hub_download/mistralai/Mixtral-8x7B-Instruct-v0.1 /data/mixtral/models

WORKDIR /data/mixtral/



RUN apt-get update -y && apt upgrade -y && apt install -y --no-install-recommends vim wget git && \
    apt autoremove && apt autoclean && \
    pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple 
EXPOSE 8007
ENTRYPOINT ["python", "main_mixtral_api.py"]