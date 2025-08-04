FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    git \
    ffmpeg \
    libcudnn8=8.9.5.* \
    libcudnn8-dev=8.9.5.* \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN python3 -m pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "-u", "runpod_handler.py"]