FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# Instala Python y herramientas necesarias
RUN apt-get update && apt-get install -y \
    python3 python3-pip ffmpeg libsndfile1 git && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    rm -rf /var/lib/apt/lists/*

# Establece la carpeta de trabajo
WORKDIR /app

# Copia requirements e instala
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia el resto del código
COPY . .

# Crea carpetas por si no existen
RUN mkdir -p input output voice_library

# Ejecutar por defecto con un video en input
CMD ["python", "video_to_text.py"]