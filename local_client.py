# local_client.py
import os
import requests
import boto3
from botocore.exceptions import NoCredentialsError
from moviepy.editor import VideoFileClip
import json
from dotenv import load_dotenv

# --- CONFIGURACIÓN PRINCIPAL ---
# La URL que obtendrás de Runpod después de crear tu endpoint.
# El formato es: https://api.runpod.ai/v2/TU_ENDPOINT_ID/run
# Usamos /run para peticiones asíncronas, ideal para trabajos largos.
RUNPOD_ENDPOINT_URL = "https://api.runpod.ai/v2/TU_ENDPOINT_ID/run" 

# La ruta al video que quieres procesar
VIDEO_PATH = os.path.join("input", next((f for f in os.listdir("input") if f.endswith(".mp4")), ""))

# --- CONFIGURACIÓN DE AWS S3 ---
# El nombre de tu bucket en S3. Debe ser globalmente único.
S3_BUCKET_NAME = "jpc-f1-transcripciones-2025"
# La región donde creaste tu bucket. Ejemplo: "us-east-1"
S3_REGION = "US East (Ohio) us-east-2" 


def extract_audio(video_path):
    """Extrae el audio de un video y lo guarda como un archivo temporal .wav."""
    audio_path = "temp_local_audio.wav"
    try:
        print(f"\nExtrayendo audio del video: {video_path}...")
        with VideoFileClip(video_path) as video_clip:
            # codec='pcm_s16le' asegura que sea un WAV sin comprimir
            video_clip.audio.write_audiofile(audio_path, codec='pcm_s16le')
        print(f"Audio extraído con éxito en: {audio_path}")
        return audio_path
    except Exception as e:
        print(f"Error al extraer el audio: {e}")
        return None

def upload_to_s3(file_path, bucket_name, object_name=None):
    """
    Sube un archivo a un bucket de S3 y genera una URL pre-firmada para su descarga.
    """
    if object_name is None:
        object_name = os.path.basename(file_path)

    # El cliente de S3 usará las credenciales configuradas en tu entorno
    s3_client = boto3.client('s3', region_name=S3_REGION)
    try:
        print(f"Subiendo '{file_path}' al bucket S3 '{bucket_name}'...")
        s3_client.upload_file(file_path, bucket_name, object_name)
        print("Subida completada.")
        
        # Generar una URL pre-firmada que permite a Runpod descargar el archivo.
        # Expira en 3600 segundos (1 hora) para dar tiempo suficiente.
        print("Generando URL segura de descarga...")
        url = s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': bucket_name, 'Key': object_name},
            ExpiresIn=3600
        )
        print("URL generada con éxito.")
        return url
    except NoCredentialsError:
        print("\nERROR CRÍTICO: Credenciales de AWS no encontradas.")
        print("Asegúrate de tener tus claves en el archivo .env o configuradas en tu sistema.")
        return None
    except Exception as e:
        print(f"Error subiendo a S3: {e}")
        return None

def main():
    """Función principal que orquesta el proceso del cliente."""
    load_dotenv() # Carga las variables desde el archivo .env

    if not os.path.exists(VIDEO_PATH):
        print(f"Error: No se encuentra el archivo de video en la carpeta 'input'.")
        return

    # 1. Extraer el audio localmente
    audio_path = extract_audio(VIDEO_PATH)
    if not audio_path:
        return

    # 2. Subir el audio a S3 y obtener la URL de descarga segura
    audio_url = upload_to_s3(audio_path, S3_BUCKET_NAME)
    
    # Limpiar el archivo de audio local tan pronto como se sube
    if os.path.exists(audio_path):
        os.remove(audio_path)
        print(f"Archivo de audio local '{audio_path}' eliminado.")

    if not audio_url:
        print("Proceso detenido debido a un error en la subida a S3.")
        return

    # 3. Preparar el payload para la API de Runpod
    # Ahora solo enviamos la URL, el payload es muy pequeño.
    payload = {
        "input": {
            "audio_url": audio_url
        }
    }

    print("\nEnviando petición de trabajo a Runpod Serverless...")
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.getenv('RUNPOD_API_KEY')}"
    }

    try:
        # Hacemos la llamada a /run para iniciar el trabajo
        response = requests.post(RUNPOD_ENDPOINT_URL, headers=headers, json=payload)
        response.raise_for_status()
        
        job_data = response.json()
        job_id = job_data.get("id")
        print(f"Trabajo enviado con éxito. ID del trabajo: {job_id}")
        print("Puedes monitorear el estado en el dashboard de Runpod.")
        print("Este script terminará ahora. El resultado se procesará en el servidor.")
        print("\nPara obtener el resultado, necesitarías implementar una llamada a la URL de estado o un webhook.")

    except requests.exceptions.RequestException as e:
        print(f"Error al contactar con el endpoint de Runpod: {e}")
        # Intenta imprimir el cuerpo de la respuesta si hay un error
        if e.response:
            print(f"Respuesta del servidor: {e.response.text}")
    except json.JSONDecodeError:
        print("Error: La respuesta del servidor no es un JSON válido.")

if __name__ == "__main__":
    main()