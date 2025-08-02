import os
import whisperx
import torch
from moviepy.editor import VideoFileClip
from collections import Counter
import re
from dotenv import load_dotenv
import gc
import openai
import numpy as np
import pandas as pd

# Se requiere speechbrain para el modelo de reconocimiento
# Asegúrese de haberlo instalado: pip install speechbrain
import speechbrain.pretrained as sb # pyright: ignore[reportMissingImports]
from scipy.spatial.distance import cdist

# --- Carga de Variables de Entorno ---
load_dotenv()
HUGGING_FACE_TOKEN = os.getenv("HF_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- Inicialización de Modelos Globales ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Usando dispositivo: {DEVICE}")

# --- CONFIGURACIÓN DE RECONOCIMIENTO DE HABLANTE ---
VOICE_LIBRARY_PATH = "voice_library" 
SIMILARITY_THRESHOLD = 0.5

# --- CONFIGURACIÓN PRINCIPAL ---

VIDEO_PATH = os.path.join("input", next((f for f in os.listdir("input") if f.endswith(".mp4")), ""))
MODEL_SIZE = "large-v3"  
LANGUAGE_CODE = "es"


# --- INTERRUPTOR DE DIARIZACIÓN ---
PERFORM_DIARIZATION = True

# --- Lista para análisis posterior (ej. Pilotos de F1) ---
F1_PILOTS_2025 = [
    "Max Verstappen", "Sergio Pérez", "Lewis Hamilton", "George Russell",
    "Charles Leclerc", "Carlos Sainz", "Lando Norris", "Oscar Piastri",
    "Fernando Alonso", "Lance Stroll", "Esteban Ocon", "Pierre Gasly",
    "Alexander Albon", "Logan Sargeant", "Yuki Tsunoda", "Daniel Ricciardo",
    "Valtteri Bottas", "Zhou Guanyu", "Kevin Magnussen", "Nico Hülkenberg",
    "Franco Colapinto", "Ollie Bearman", "Isak Hadjar", "Kimi Antonelli", "Liam Lawson", "Gabriel Bortoleto"
]



embedding_model = None
try:
    print("Cargando modelo de huellas vocales (SpeechBrain)...")
    embedding_model = sb.EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": DEVICE},
        savedir=os.path.join("pretrained_models", "spkrec-ecapa-voxceleb")
    )
    print("Modelo de huellas vocales cargado.")
except Exception as e:
    print(f"Error al cargar el modelo de SpeechBrain: {e}")

voice_library_embeddings = {}


def extract_audio(video_path):
    """Extrae el audio de un video y lo guarda como un archivo temporal .wav."""
    audio_path = "temp_full_audio.wav"
    try:
        print(f"\nExtrayendo audio del video: {video_path}...")
        with VideoFileClip(video_path) as video_clip:
            video_clip.audio.write_audiofile(audio_path, codec='pcm_s16le')
        print("Audio extraído con éxito.")
        return audio_path
    except Exception as e:
        print(f"Error al extraer el audio: {e}")
        return None

def format_timestamp(seconds):
    """Convierte segundos a un formato de tiempo HH:MM:SS."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds_rem = int(seconds % 60)
    return f"{hours:02}:{minutes:02}:{seconds_rem:02}"

def initialize_voice_library():
    """
    Analiza los archivos .wav en la biblioteca de voces, carga su audio,
    calcula sus huellas vocales (embeddings) usando SpeechBrain y las guarda.
    """
    global voice_library_embeddings
    if not embedding_model:
        print("El modelo de embedding no está cargado. Saltando inicialización de biblioteca de voces.")
        return

    print("\n--- Inicializando Biblioteca de Voces ---")
    if not os.path.exists(VOICE_LIBRARY_PATH) or not os.path.isdir(VOICE_LIBRARY_PATH):
        print(f"ADVERTENCIA: La carpeta de la biblioteca de voces no existe en '{VOICE_LIBRARY_PATH}'.")
        return

    for file_name in os.listdir(VOICE_LIBRARY_PATH):
        if file_name.endswith(".wav"):
            speaker_name = os.path.splitext(file_name)[0]
            file_path = os.path.join(VOICE_LIBRARY_PATH, file_name)
            try:
                print(f" - Procesando referencia para: {speaker_name}")
                
                signal_np = whisperx.load_audio(file_path)
                signal = torch.from_numpy(signal_np).unsqueeze(0).to(DEVICE)

                embedding = embedding_model.encode_batch(signal)
                embedding = embedding.squeeze()

                voice_library_embeddings[speaker_name] = embedding
            except Exception as e:
                print(f"Error procesando el archivo de referencia {file_name}: {e}")
    if not voice_library_embeddings:
        print("ADVERTENCIA: No se cargó ninguna voz de referencia.")
    else:
        print(f"--- Biblioteca de Voces Cargada con {len(voice_library_embeddings)} hablantes ---")

def identify_speakers_automatically(diarization_df, audio_waveform):
    """
    Identifica hablantes usando el DataFrame de la diarización.
    """
    if not voice_library_embeddings or not isinstance(diarization_df, pd.DataFrame) or diarization_df.empty:
        print("Biblioteca de voces vacía o diarización inválida. Saltando identificación automática.")
        return {}

    print("\n--- Realizando Identificación Automática de Hablantes ---")
    speaker_mapping = {}
    
    # Agrupar la tabla de datos por la columna 'speaker'
    for speaker_label, segments_df in diarization_df.groupby('speaker'):
        try:
            # Encontrar el segmento más largo para este hablante
            durations = segments_df['end'] - segments_df['start']
            if durations.empty:
                continue
            longest_segment_idx = durations.idxmax()
            longest_segment = segments_df.loc[longest_segment_idx]
            
            start_time = longest_segment['start']
            end_time = longest_segment['end']
            
            sample_rate = 16000 # Whisper trabaja a 16kHz
            clip = audio_waveform[int(start_time * sample_rate):int(end_time * sample_rate)]
            
            if len(clip) == 0: continue

            clip_tensor = torch.from_numpy(clip).unsqueeze(0).to(DEVICE)
            embedding_unknown = embedding_model.encode_batch(clip_tensor)
            embedding_unknown = embedding_unknown.squeeze()

            best_match_name = "DESCONOCIDO"
            highest_similarity = 0.0

            for name, embedding_known in voice_library_embeddings.items():
                emb_unknown_np = embedding_unknown.cpu().numpy()
                emb_known_np = embedding_known.cpu().numpy()
                
                if emb_unknown_np.ndim == 1: emb_unknown_np = np.expand_dims(emb_unknown_np, axis=0)
                if emb_known_np.ndim == 1: emb_known_np = np.expand_dims(emb_known_np, axis=0)

                similarity = 1 - cdist(emb_unknown_np, emb_known_np, "cosine")[0][0]
                if similarity > highest_similarity:
                    highest_similarity = similarity
                    best_match_name = name

            if highest_similarity > SIMILARITY_THRESHOLD:
                speaker_mapping[speaker_label] = best_match_name
                print(f"  - Coincidencia encontrada: {speaker_label} -> {best_match_name} (Similitud: {highest_similarity:.2f})")
            else:
                speaker_mapping[speaker_label] = speaker_label
                print(f"  - Sin coincidencia clara para {speaker_label}. Mejor intento: {best_match_name} (Similitud: {highest_similarity:.2f} < {SIMILARITY_THRESHOLD})")
        except Exception as e:
            print(f"Error procesando el hablante {speaker_label}: {e}")

    return speaker_mapping

def run_transcription_and_recognition():
    """
    Función principal que orquesta todo el proceso.
    """
    if not os.path.exists(VIDEO_PATH):
        print(f"Error: El archivo de video no se encuentra en '{VIDEO_PATH}'")
        return None, None

    initialize_voice_library()

    full_audio_path = extract_audio(VIDEO_PATH)
    if not full_audio_path:
        return None, None

    print("\n--- Cargando Modelos de Transcripción y Diarización ---")
    compute_type = "float16" if DEVICE == "cuda" else "int8"
    
    print("Cargando modelo de transcripción (Whisper)...")
    model = whisperx.load_model(MODEL_SIZE, DEVICE, compute_type=compute_type, language=LANGUAGE_CODE)
    
    print("Cargando modelo de alineación...")
    model_a, metadata = whisperx.load_align_model(language_code=LANGUAGE_CODE, device=DEVICE)
    
    diarize_model = None
    if PERFORM_DIARIZATION:
        print("Cargando modelo de diarización...")
        if not HUGGING_FACE_TOKEN:
            print("ADVERTENCIA: No se encontró HF_TOKEN. La descarga del modelo de diarización puede fallar.")
        diarize_model = whisperx.diarize.DiarizationPipeline(use_auth_token=HUGGING_FACE_TOKEN, device=DEVICE)
        print("¡Modelo de diarización cargado!")
    
    print("\n--- Iniciando Procesamiento del Audio ---")
    
    audio_waveform = whisperx.load_audio(full_audio_path)

    print("1. Transcribiendo el audio completo...")
    result = model.transcribe(audio_waveform, batch_size=8)
    
    if not result.get("segments"):
        print("No se detectó texto en el audio.")
        os.remove(full_audio_path)
        return None, None

    print("2. Alineando la transcripción...")
    result = whisperx.align(result["segments"], model_a, metadata, audio_waveform, DEVICE, return_char_alignments=False)

    final_speaker_mapping = {}
    if PERFORM_DIARIZATION and diarize_model:
        print("3. Realizando diarización para separar hablantes...")
        # La salida ahora es un DataFrame
        diarization_df = diarize_model(audio_waveform,max_speakers=4)
        
        # assign_word_speakers funciona correctamente con el DataFrame
        result = whisperx.assign_word_speakers(diarization_df, result)
        
        print("4. Identificando quién es cada hablante automáticamente...")
        final_speaker_mapping = identify_speakers_automatically(diarization_df, audio_waveform)
        
    else:
        print("Diarización desactivada. No se identificarán hablantes.")

    print("\n--- Generando Transcripción Final con Nombres Identificados ---")
    full_transcript_content = []
    for segment in result.get("segments", []):
        start_time = segment.get("start", 0)
        # La clave "speaker" ahora existe gracias a assign_word_speakers
        speaker_label = segment.get("speaker", "NARRADOR")
        text = segment.get("text", "").strip()

        if text:
            display_name = final_speaker_mapping.get(speaker_label, speaker_label)
            line = f"[{format_timestamp(start_time)}] {display_name}: {text}"
            print(line)
            full_transcript_content.append(line)
    
    print("\nLiberando memoria de modelos y eliminando archivos temporales...")
    del model, model_a, diarize_model
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    if os.path.exists(full_audio_path):
        os.remove(full_audio_path)
    
    final_text = "\n".join(full_transcript_content)
    
    transcript_filename = "transcripcion_completa_identificada.txt"
    with open(transcript_filename, "w", encoding="utf-8") as f:
        f.write(final_text)
    print(f"\nTranscripción completa guardada en '{transcript_filename}'")
    
    return final_text, {}

def count_pilot_mentions(text, pilots):
    """Cuenta las menciones de cada piloto en el texto."""
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
    """Envía la transcripción a OpenAI para un análisis detallado."""
    if not OPENAI_API_KEY:
        print("\nERROR: OPENAI_API_KEY no encontrada. No se puede realizar el resumen con GPT.")
        return
    if not full_text.strip():
        print("\nLa transcripción está vacía. Saltando el análisis con GPT.")
        return
        
    print("\n--- Realizando Análisis Final con GPT-4o ---")
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
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4096,
            temperature=0.5,
        )
        resumen_general = response.choices[0].message.content.strip()

        summary_filename = "resumen_analisis_narradores.txt"
        with open(summary_filename, "w", encoding="utf-8") as f:
            f.write(resumen_general)
        print(f"\nAnálisis de GPT-4o guardado con éxito en '{summary_filename}'")
        print("\n--- Contenido del Análisis ---\n" + resumen_general)
    except Exception as e:
        print(f"\nOcurrió un error al contactar con la API de OpenAI: {e}")

if __name__ == "__main__":
    transcribed_text, _ = run_transcription_and_recognition()
    
    if transcribed_text:
        mentions = count_pilot_mentions(transcribed_text, F1_PILOTS_2025)
        print("\n--- Conteo Total de Menciones de Pilotos ---")
        if not mentions:
            print("No se encontraron menciones de pilotos.")
        else:
            for pilot, count in mentions.most_common():
                print(f"{pilot}: {count}")
        
        analyze_with_gpt(transcribed_text, mentions)
    
    print("\nProceso finalizado.")