# runpod_handler.py
import os
import gc
import re
from collections import Counter

import torch
import whisperx
import pandas as pd
import numpy as np
import openai
import speechbrain.pretrained as sb # pyright: ignore[reportMissingImports]
from scipy.spatial.distance import cdist

import runpod
import requests
import traceback

# --- CARGA DE VARIABLES DE ENTORNO (Se configuran en Runpod UI) ---
HUGGING_FACE_TOKEN = os.getenv("HF_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- INICIALIZACIÓN DE MODELOS GLOBALES (Se ejecutan UNA VEZ por worker) ---
print("--- INICIANDO WORKER MULTILINGÜE ---")
DEVICE = "cuda"
COMPUTE_TYPE = "float16"

print(f"Usando dispositivo: {DEVICE} con compute_type: {COMPUTE_TYPE}")

# Variables para los modelos que se cargarán
embedding_model = None
model = None
diarize_model = None
# Precargamos el modelo de alineación para el idioma más común (español)
# para acelerar las peticiones en ese idioma.
model_a_es, metadata_es = None, None

try:
    embedding_model = sb.EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": DEVICE},
        savedir=os.path.join("pretrained_models", "spkrec-ecapa-voxceleb")
    )
    print("Modelo de huellas vocales (SpeechBrain) cargado.")
    
    # Cargar el modelo de transcripción SIN especificar idioma para activar la detección automática.
    model = whisperx.load_model("large-v3", DEVICE, compute_type=COMPUTE_TYPE)
    print("Modelo de transcripción (WhisperX large-v3) cargado en modo multilingüe.")

    # Precargar el modelo de alineación para español
    model_a_es, metadata_es = whisperx.load_align_model(language_code="es", device=DEVICE)
    print("Modelo de alineación para 'es' precargado.")

    diarize_model = whisperx.diarize.DiarizationPipeline(use_auth_token=HUGGING_FACE_TOKEN, device=DEVICE)
    print("Modelo de diarización cargado.")

except Exception as e:
    print(f"ERROR CRÍTICO DURANTE LA CARGA DE MODELOS: {e}")
    traceback.print_exc()

# --- CONFIGURACIONES Y LISTAS GLOBALES ---
SIMILARITY_THRESHOLD = 0.5
VOICE_LIBRARY_PATH = "voice_library"
voice_library_embeddings = {}

F1_PILOTS_2025 = [
    "Max Verstappen", "Sergio Pérez", "Lewis Hamilton", "George Russell",
    "Charles Leclerc", "Carlos Sainz", "Lando Norris", "Oscar Piastri",
    "Fernando Alonso", "Lance Stroll", "Esteban Ocon", "Pierre Gasly",
    "Alexander Albon", "Logan Sargeant", "Yuki Tsunoda", "Daniel Ricciardo",
    "Valtteri Bottas", "Zhou Guanyu", "Kevin Magnussen", "Nico Hülkenberg",
    "Franco Colapinto", "Ollie Bearman", "Isak Hadjar", "Kimi Antonelli", "Liam Lawson", "Gabriel Bortoleto"
]

# --- FUNCIONES DE PROCESAMIENTO ---
def format_timestamp(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds_rem = int(seconds % 60)
    return f"{hours:02}:{minutes:02}:{seconds_rem:02}"

def initialize_voice_library():
    global voice_library_embeddings
    if not embedding_model: return
    if not os.path.exists(VOICE_LIBRARY_PATH) or not os.path.isdir(VOICE_LIBRARY_PATH): return

    print(f"Inicializando biblioteca de voces desde '{VOICE_LIBRARY_PATH}'...")
    for file_name in os.listdir(VOICE_LIBRARY_PATH):
        if file_name.endswith(".wav"):
            speaker_name = os.path.splitext(file_name)[0]
            file_path = os.path.join(VOICE_LIBRARY_PATH, file_name)
            try:
                signal_np = whisperx.load_audio(file_path)
                signal = torch.from_numpy(signal_np).unsqueeze(0).to(DEVICE)
                embedding = embedding_model.encode_batch(signal).squeeze()
                voice_library_embeddings[speaker_name] = embedding
            except Exception as e:
                print(f"Error procesando {file_name}: {e}")
    print(f"Biblioteca de voces cargada con {len(voice_library_embeddings)} hablantes.")

def identify_speakers_automatically(diarization_df, audio_waveform):
    if not voice_library_embeddings or diarization_df.empty: return {}
    speaker_mapping = {}
    for speaker_label, segments in diarization_df.groupby('speaker'):
        try:
            longest_segment = segments.loc[(segments['end'] - segments['start']).idxmax()]
            start, end = longest_segment['start'], longest_segment['end']
            clip = audio_waveform[int(start * 16000):int(end * 16000)]
            if len(clip) == 0: continue
            
            clip_tensor = torch.from_numpy(clip).unsqueeze(0).to(DEVICE)
            embedding_unknown = embedding_model.encode_batch(clip_tensor).squeeze()
            
            highest_similarity, best_match_name = 0.0, "DESCONOCIDO"
            for name, emb_known in voice_library_embeddings.items():
                emb_unknown_np = embedding_unknown.cpu().numpy()
                emb_known_np = emb_known.cpu().numpy()
                if emb_unknown_np.ndim == 1: emb_unknown_np = np.expand_dims(emb_unknown_np, axis=0)
                if emb_known_np.ndim == 1: emb_known_np = np.expand_dims(emb_known_np, axis=0)
                similarity = 1 - cdist(emb_unknown_np, emb_known_np, "cosine")[0][0]
                if similarity > highest_similarity:
                    highest_similarity, best_match_name = similarity, name
            
            if highest_similarity > SIMILARITY_THRESHOLD:
                speaker_mapping[speaker_label] = best_match_name
            else:
                speaker_mapping[speaker_label] = f"{speaker_label} (Desconocido)"
        except Exception as e:
            print(f"Error identificando {speaker_label}: {e}")
            speaker_mapping[speaker_label] = speaker_label
    return speaker_mapping

# --- INICIALIZACIÓN DEL WORKER ---
initialize_voice_library()
print("--- WORKER LISTO Y ESPERANDO TRABAJOS ---")


# --- EL HANDLER PRINCIPAL DE RUNPOD ---
def handler(job):
    if not all([embedding_model, model, diarize_model]):
        return {"error": "El worker no se inicializó correctamente, uno o más modelos fallaron al cargar."}
        
    job_input = job['input']
    audio_url = job_input.get("audio_url")
    if not audio_url:
        return {"error": "No se recibió 'audio_url' en la petición."}
    
    language_code_input = job_input.get("language_code")

    temp_audio_path = "/tmp/job_audio.wav"
    try:
        print(f"Descargando audio desde la URL...")
        with requests.get(audio_url, stream=True) as r:
            r.raise_for_status()
            with open(temp_audio_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print("Descarga completada.")
    except Exception as e:
        return {"error": f"No se pudo descargar el archivo de audio: {str(e)}"}

    try:
        print("Cargando forma de onda de audio...")
        audio_waveform = whisperx.load_audio(temp_audio_path)
        
        # --- LÓGICA DE DETECCIÓN DE IDIOMA ---
        detected_language = language_code_input
        if not detected_language:
            print("No se especificó idioma. Detectando automáticamente...")
            segments = model.transcribe(audio_waveform[:30*16000])["segments"]
            detected_language = segments[0]["language"] if segments else "en"
            print(f"Idioma detectado: {detected_language}")
        else:
            print(f"Idioma forzado por el usuario: {detected_language}")

        # --- CARGA DINÁMICA DEL MODELO DE ALINEACIÓN ---
        local_model_a, local_metadata = None, None
        if detected_language == "es" and model_a_es is not None:
            print("Usando modelo de alineación 'es' precargado.")
            local_model_a, local_metadata = model_a_es, metadata_es
        else:
            print(f"Cargando modelo de alineación para el idioma: '{detected_language}'...")
            try:
                local_model_a, local_metadata = whisperx.load_align_model(
                    language_code=detected_language, device=DEVICE
                )
                print("Nuevo modelo de alineación cargado.")
            except Exception as e:
                return {"error": f"No se pudo cargar el modelo de alineación para '{detected_language}': {e}"}

        # --- PROCESAMIENTO PRINCIPAL ---
        print("1. Transcribiendo audio completo...")
        result = model.transcribe(audio_waveform, batch_size=8, language=detected_language)
        if not result.get("segments"):
            return {"transcription": "No se detectó texto en el audio.", "detected_language": detected_language}
        
        print(f"2. Alineando transcripción para '{detected_language}'...")
        result = whisperx.align(result["segments"], local_model_a, local_metadata, audio_waveform, DEVICE, return_char_alignments=False)
        
        print("3. Realizando diarización...")
        diarization_df = diarize_model(audio_waveform, max_speakers=4)
        result = whisperx.assign_word_speakers(diarization_df, result)
        
        print("4. Identificando hablantes...")
        final_speaker_mapping = identify_speakers_automatically(diarization_df, audio_waveform)
        
        print("Generando transcripción final...")
        full_transcript_content = []
        for segment in result.get("segments", []):
            start_time = segment.get("start", 0)
            speaker_label = segment.get("speaker", "NARRADOR")
            text = segment.get("text", "").strip()
            if text:
                display_name = final_speaker_mapping.get(speaker_label, speaker_label)
                line = f"[{format_timestamp(start_time)}] {display_name}: {text}"
                full_transcript_content.append(line)
        
        final_text = "\n".join(full_transcript_content)
        
        # El análisis de GPT está desactivado por defecto para evitar costos y porque el prompt es específico de español.
        # Puedes reactivarlo si lo adaptas para ser multilingüe.
        gpt_analysis_text = "El análisis con GPT está desactivado en el modo multilingüe."
        
        print("Proceso completado con éxito.")
        
        return {
            "transcription": final_text,
            "detected_language": detected_language,
            "gpt_analysis": gpt_analysis_text
        }

    except Exception as e:
        print(f"ERROR DURANTE EL PROCESAMIENTO DEL TRABAJO: {e}")
        traceback.print_exc()
        return {"error": f"Ocurrió un error en el procesamiento: {str(e)}"}
    finally:
        # Limpieza
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
            print("Archivo de audio temporal eliminado.")
        
        # Liberar memoria de la GPU
        gc.collect()
        torch.cuda.empty_cache()

# Iniciar el servidor de Runpod
runpod.serverless.start({"handler": handler})