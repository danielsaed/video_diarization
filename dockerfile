FROM runpod/pytorch:2.2.1-py3.11-cuda12.1.1-devel-ubuntu22.04

# Establecer el directorio de trabajo dentro del contenedor
WORKDIR /app

# Instalar dependencias del sistema operativo.
# whisperx (y muchas otras librerías de audio/video) requiere ffmpeg.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copiar primero el archivo de requerimientos.
# Esto es una optimización clave de Docker. Si este archivo no cambia,
# Docker usará la caché para la instalación de pip, haciendo que las
# futuras construcciones sean mucho más rápidas.
COPY requirements.txt .

# Instalar las dependencias de Python.
# --no-cache-dir ayuda a mantener el tamaño de la imagen final más pequeño.
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el resto del código de la aplicación al directorio de trabajo.
# Esto incluye tu runpod_handler.py y la carpeta voice_library/.
COPY . .

# Comando que se ejecutará cuando el contenedor se inicie.
# Le dice a Runpod que inicie tu script handler.
# El flag -u es para salida sin buffer, lo que mejora la visualización de logs en tiempo real.
CMD ["python", "-u", "runpod_handler.py"]