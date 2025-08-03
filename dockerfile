# Usas la imagen runtime como base
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# Establecer variables para que la instalación no sea interactiva
ENV DEBIAN_FRONTEND=noninteractive

# Instalar dependencias básicas del sistema, INCLUYENDO Python y pip
RUN apt-get update && apt-get install -y --no-install-recommends \
    gnupg \
    wget \
    ffmpeg \
    git \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# --- INICIO: INSTALACIÓN MANUAL DE cuDNN ---
# Añadir el repositorio de NVIDIA para poder instalar cuDNN
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
RUN dpkg -i cuda-keyring_1.1-1_all.deb
RUN apt-get update
# Instalar las librerías de cuDNN para CUDA 12.1
RUN apt-get install -y libcudnn8=8.9.7.29-1+cuda12.1
# --- FIN: INSTALACIÓN MANUAL DE cuDNN ---

# Establecer el directorio de trabajo
WORKDIR /app

# Copiar el archivo de requerimientos
COPY requirements.txt .

# Instalar las dependencias de Python.
# Nota: Ahora también necesitamos instalar PyTorch, ya que no viene en la imagen base.
RUN python3 -m pip install --no-cache-dir -r requirements.txt

# Copiar el resto del código
COPY . .

# Comando de inicio
CMD ["python", "-u", "runpod_handler.py"]