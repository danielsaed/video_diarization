# Usas la imagen runtime como base
FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

# Establecer variables para que la instalación no sea interactiva
ENV DEBIAN_FRONTEND=noninteractive

# Instalar dependencias básicas del sistema, INCLUYENDO Python y pip
RUN apt-get update && apt-get install -y --no-install-recommends \
    gnupg \
    wget \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*


# Instalar las librerías de cuDNN para CUDA 12.1
#RUN apt-get install -y libcudnn8 libcudnn8-dev
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