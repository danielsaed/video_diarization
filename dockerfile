FROM runpod/pytorch:2.2.1-py3.11-cuda12.1.1-devel-ubuntu22.04

# Establecer el directorio de trabajo dentro del contenedor
WORKDIR /app

# Instalar dependencias del sistema operativo.
# Se añade 'git' para poder instalar paquetes desde repositorios de GitHub.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copiar primero el archivo de requerimientos para optimizar la caché de Docker.
COPY requirements.txt .

# Instalar las dependencias de Python.
# Se cambia 'pip' por 'python3 -m pip' para ser explícito y evitar errores de "not found".
RUN python3 -m pip install --no-cache-dir -r requirements.txt

# Copiar el resto del código de la aplicación al directorio de trabajo.
COPY . .

# Comando que se ejecutará cuando el contenedor se inicie.
CMD ["python", "-u", "runpod_handler.py"]