# runpod_handler.py
import os
import gc
import re
import base64
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

# --- CARGA DE VARIABLES DE ENTORNO (Se configuran en Runpod UI) ---
HUGGING_FACE_TOKEN = os.getenv("HF_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- INICIALIZACIÓN DE MODELOS GLOBALES (Se ejecutan UNA VEZ por worker) ---
print("--- INICIANDO WORKER: CARGA DE MODELOS GLOBALES ---")
DEVICE = "cuda"
COMPUTE_TYPE = "float16" # "int8" si tienes problemas de memoria

# Carga de modelos de IA
print(f"Usando dispositivo: {DEVICE} con compute_type: {COMPUTE_TYPE}")

try:
    embedding_model = sb.EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": DEVICE},
        savedir=os.path.join("pretrained_models", "spkrec-ecapa-voxceleb")
    )
    print("Modelo de huellas vocales (SpeechBrain) cargado.")
    
    model = whisperx.load_model("large-v3", DEVICE, compute_type=COMPUTE_TYPE, language="es")
    print("Modelo de transcripción (WhisperX large-v3) cargado.")

    model_a, metadata = whisperx.load_align_model(language_code="es", device=DEVICE)
    print("Modelo de alineación cargado.")

    diarize_model = whisperx.diarize.DiarizationPipeline(use_auth_token=HUGGING_FACE_TOKEN, device=DEVICE)
    print("Modelo de diarización cargado.")

except Exception as e:
    print(f"ERROR CRÍTICO DURANTE LA CARGA DE MODELOS: {e}")
    # Si un modelo no carga, el worker no puede funcionar.
    embedding_model = model = model_a = diarize_model = None

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

# --- FUNCIONES DE PROCESAMIENTO (Copiadas de tu script original) ---
# (Aquí van todas tus funciones: format_timestamp, initialize_voice_library, 
# identify_speakers_automatically, count_pilot_mentions, analyze_with_gpt)
# Las he copiado y pegado abajo para mantener la estructura limpia.

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
            
            highest_similarity = 0.0
            best_match_name = "DESCONOCIDO"
            for name, emb_known in voice_library_embeddings.items():
                emb_unknown_np = embedding_unknown.cpu().numpy()
                emb_known_np = emb_known.cpu().numpy()
                if emb_unknown_np.ndim == 1: emb_unknown_np = np.expand_dims(emb_unknown_np, axis=0)
                if emb_known_np.ndim == 1: emb_known_np = np.expand_dims(emb_known_np, axis=0)

                similarity = 1 - cdist(emb_unknown_np, emb_known_np, "cosine")[0][0]
                if similarity > highest_similarity:
                    highest_similarity = similarity
                    best_match_name = name
            
            if highest_similarity > SIMILARITY_THRESHOLD:
                speaker_mapping[speaker_label] = best_match_name
            else:
                speaker_mapping[speaker_label] = f"{speaker_label} (Desconocido)"
        except Exception as e:
            print(f"Error identificando {speaker_label}: {e}")
            speaker_mapping[speaker_label] = speaker_label
    return speaker_mapping

def count_pilot_mentions(text, pilots):
    counts = Counter()
    text_lower = text.lower()
    pilots_sorted = sorted(pilots, key=len, reverse=True)
    for pilot in pilots_sorted:
        pilot_lower = pilot.lower()
        pattern = r'\b' + re.escape(pilot_lower) + r'\b'
        matches = re.findall(pattern, text_lower)
        if matches:
            counts[pilot] += len(matches)
            text_lower = re.sub(pattern, '', text_lower)
    return counts

def analyze_with_gpt(full_text, pilot_mentions):
    if not OPENAI_API_KEY or not full_text.strip(): return "Análisis GPT no disponible."
    pilotos_mencionados = [pilot for pilot, count in pilot_mentions.most_common()]
    pilotos_str = ", ".join(pilotos_mencionados) if pilotos_mencionados else "ninguno en particular"
    prompt = (
        f"Eres un analista experto en relaciones publicas, percepcion publica y de imagen personal especializado en motorsport especificamente en Fórmula 1. \n"
        f"Tu tarea es atravez de una transcripcion de una sesion de F1, identificar patrones, ya sea de comportamiento, sesgos, intereses personales, conocimiento, ideologia, xenophobia, racismo y cualquier cosa relacionada al campo en el cual eres experto. El objetivo es conocer la calidad de la transmicion de F1 en espanol asi como la percepcion hacia los pilotos, equipos y personajes de F1 de parte de los relatores.\n" 
        f"Los comentaristas identificados son Fernando Tornello ( Relator, Argentino, mas de 30 anos narrando F1, edad: 72), Cochito Lopez ( Analista, Argentino, Expiloto profesional, edad: 45), Albert Fabrega (Espanol, Periodista, Exmecanico de f1 experto en cosas tecnicas referentes a motorsport, edad: 53), Juan Fosarolli ( periodista, edad: 50 - 60 anos )\n"
        f"Si se habla bien de un piloto que hace un buen performance es normal no existe sesgo, al igual que si tiene un mal performance sea lo contrario\n"
        f"1. **Resumen General:** Crea un resumen de los eventos clave de la sesión.\n"
        f"2. **Análisis por Piloto:** Para los pilotos mencionados ({pilotos_str}), en un pequeno resumen analiza en que contexto y tono fue mencionado, como es la percepcion de los comentariastas hacia el y en comparacion a los demas pilotos que tanto fue mencionado, si se emiten opiniones o cosas veridicas\n"
        f"3. **Evaluación de Comentaristas:** Evalúa la imparcialidad y profesionalismo de los comentaristas basándote en su diálogo y el contexto proporcionado, ademas identifica la relevancia del aporte que hace cada uno respectivamente a la transmicion. con que frecuencia emiten opiniones y con que frecuencia emiten hechos.\n\n "
        "Estructura tu respuesta claramente usando Markdown.\n\n"
        "--- INICIO DE LA TRANSCRIPCIÓN ---\n"
        f"{full_text}"
        "\n--- FIN DE LA TRANSCRIPCIÓN ---"
    )
    try:
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": prompt}], max_tokens=4096, temperature=0.5)
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error al contactar con la API de OpenAI: {e}"

# --- INICIALIZACIÓN DEL WORKER ---
# Se inicializa la biblioteca de voces una vez, al arrancar el worker.
initialize_voice_library()
print("--- WORKER LISTO Y ESPERANDO TRABAJOS ---")


# --- EL HANDLER PRINCIPAL DE RUNPOD ---
def handler(job):
    """
    Esta es la función que Runpod ejecutará con cada petición.
    """
    if not all([embedding_model, model, model_a, diarize_model]):
        return {"error": "El worker no se inicializó correctamente, uno o más modelos fallaron al cargar."}
        
    job_input = job['input']
    
    # 1. Obtener la URL del audio y descargarlo
    audio_url = job_input.get("audio_url")
    if not audio_url:
        return {"error": "No se recibió 'audio_url' en el input."}

    # Ruta temporal dentro del contenedor para guardar el audio
    temp_audio_path = "/tmp/job_audio.wav"
    
    try:
        print(f"Descargando audio desde la URL pre-firmada...")
        with requests.get(audio_url, stream=True) as r:
            r.raise_for_status()
            with open(temp_audio_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print("Descarga de audio completada.")
    except Exception as e:
        return {"error": f"No se pudo descargar el archivo de audio: {str(e)}"}

    # 2. Ejecutar el pipeline de procesamiento de IA
    try:
        print("Cargando forma de onda de audio...")
        audio_waveform = whisperx.load_audio(temp_audio_path)
        
        print("1. Transcribiendo audio...")
        result = model.transcribe(audio_waveform, batch_size=8)
        if not result.get("segments"):
            return {"transcription": "No se detectó texto en el audio.", "gpt_analysis": ""}
        
        print("2. Alineando transcripción...")
        result = whisperx.align(result["segments"], model_a, metadata, audio_waveform, DEVICE, return_char_alignments=False)
        
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
        
        print("5. Realizando análisis con GPT-4o...")
        mentions = count_pilot_mentions(final_text, F1_PILOTS_2025)
        gpt_analysis_text = analyze_with_gpt(final_text, mentions)
        
        print("Proceso completado con éxito.")
        
        # 3. Devolver el resultado final
        return {
            "transcription": final_text,
            "gpt_analysis": gpt_analysis_text
        }

    except Exception as e:
        # Imprime el error en los logs de Runpod para depuración
        print(f"ERROR DURANTE EL PROCESAMIENTO DEL TRABAJO: {e}")
        return {"error": f"Ocurrió un error durante el procesamiento: {str(e)}"}
    finally:
        # 4. Limpieza: Asegurarse de borrar el archivo de audio descargado
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
            print("Archivo de audio temporal eliminado.")
        
        # Liberar memoria de la GPU
        gc.collect()
        torch.cuda.empty_cache()

# Iniciar el servidor de Runpod para que escuche peticiones
runpod.serverless.start({"handler": handler})