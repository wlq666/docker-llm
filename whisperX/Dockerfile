FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-devel

COPY api/docker/whisperX /data/whisperX

COPY llm_model/hugging_hub_download/openai/whisper-large-v3 /data/whisperX/models

WORKDIR /data/whisperX
RUN apt-get update -y && apt upgrade -y && apt install -y --no-install-recommends vim wget git && \
    apt autoremove && apt autoclean && \
    pip install --no-cache-dir --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

EXPOSE 8007
CMD ["python", "main_whisperX_api.py"]