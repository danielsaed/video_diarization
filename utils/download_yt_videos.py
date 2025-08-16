
import os
import yt_dlp
import datetime

# --- Funciones de Ayuda (sin cambios) ---
def formatear_duracion(segundos):
    if segundos is None: return "N/A"
    return str(datetime.timedelta(seconds=int(segundos)))

def formatear_fecha(fecha_str):
    if fecha_str is None: return "N/A"
    try:
        return datetime.datetime.strptime(fecha_str, '%Y%m%d').strftime('%d de %B de %Y')
    except (ValueError, TypeError):
        return fecha_str

def guardar_metadata_txt(info_dict, ruta_base):
    ruta_txt = os.path.splitext(ruta_base)[0] + '.txt'
    print(f"-> Guardando metadatos en: {os.path.basename(ruta_txt)}")
    try:
        with open(ruta_txt, 'w', encoding='utf-8') as f:
            f.write(f"Título: {info_dict.get('title', 'N/A')}\n")
            f.write(f"URL: {info_dict.get('webpage_url', 'N/A')}\n")
            f.write(f"Canal: {info_dict.get('uploader', 'N/A')}\n")
            f.write(f"Fecha de Subida: {formatear_fecha(info_dict.get('upload_date'))}\n")
            f.write(f"Duración: {formatear_duracion(info_dict.get('duration'))}\n")
            f.write(f"Vistas: {info_dict.get('view_count', 0):,}\n\n")
            f.write("--- DESCRIPCIÓN ---\n")
            f.write(info_dict.get('description', 'No disponible'))
    except Exception as e:
        print(f"Error al guardar el archivo de metadatos: {e}")

# --- Funciones Principales (CON LA CORRECCIÓN) ---

def descargar_con_ytdlp(url, opciones, guardar_meta):
    """
    Función de descarga con un hook de progreso a prueba de errores.
    """
    # Usamos una clausura (closure) para que el hook pueda acceder a 'guardar_meta'
    def mi_hook_de_progreso_robusto(d):
        if d['status'] == 'downloading':
            # --- INICIO DE LA CORRECCIÓN ---
            # Obtenemos los valores de forma segura, asignando 0 si no existen o son None
            total_bytes = d.get('total_bytes') or d.get('total_bytes_estimate') or 0
            downloaded_bytes = d.get('downloaded_bytes', 0)
            speed = d.get('speed', 0)

            # Nos aseguramos de que las variables no sean None antes de usarlas
            if downloaded_bytes is None:
                downloaded_bytes = 0
            if speed is None:
                speed = 0

            percent_str = "---%"
            if total_bytes > 0:
                percent_str = f"{(downloaded_bytes / total_bytes * 100):.1f}%"
            
            speed_str = "Unknown B/s"
            if speed > 0:
                speed_str = f"{speed / (1024*1024):.1f} MB/s" if speed > 1024*1024 else f"{speed / 1024:.1f} KB/s"
            
            eta = d.get('eta')
            eta_str = str(eta) if eta else "Unknown"
            
            print(f"\rDescargando: {percent_str} | Velocidad: {speed_str} | ETA: {eta_str}s  ", end="")
            # --- FIN DE LA CORRECCIÓN ---
        
        elif d['status'] == 'finished':
            print(f"\nDescarga de '{os.path.basename(d['filename'])}' completada.")
            if guardar_meta:
                guardar_metadata_txt(d['info_dict'], d['filename'])

    opciones_completas = opciones.copy()
    opciones_completas['progress_hooks'] = [mi_hook_de_progreso_robusto]
    opciones_completas['ignoreerrors'] = True

    try:
        with yt_dlp.YoutubeDL(opciones_completas) as ydl:
            ydl.download([url])
        print("\n¡Proceso finalizado!")
    except Exception as e:
        print(f"\nOcurrió un error general: {e}")


def obtener_datos_canal_o_playlist(url):
    print("\nObteniendo información del canal/playlist...")
    ydl_opts = {'quiet': True, 'no_warnings': True, 'extract_flat': True, 'playlistend': 15}
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(url, download=False)
            print("\n" + "="*50)
            print(f"Título: {info_dict.get('title', 'N/A')}")
            print(f"Uploader: {info_dict.get('uploader', 'N/A')}")
            print(f"Total de Videos (aprox): {info_dict.get('playlist_count', len(info_dict.get('entries', [])))}")
            print("\n--- ÚLTIMOS 15 VIDEOS EN LA LISTA ---")
            for i, entry in enumerate(info_dict.get('entries', [])):
                print(f"{i+1}. {entry.get('title')}")
            print("-" * 50)
    except Exception as e:
        print(f"\nNo se pudo obtener la información: {e}")


# --- Bloque Principal de Ejecución (sin cambios) ---
if __name__ == "__main__":
    while True:
        print("\n=========== Herramienta Definitiva para YouTube con yt-dlp ===========")
        print("1. Descargar desde un Canal o Playlist")
        print("2. Descargar un solo Video")
        print("3. Obtener Información de un Canal/Playlist (sin descargar)")
        print("4. Salir")
        opcion_principal = input("Elige una opción: ")

        if opcion_principal == '4': break
        if opcion_principal not in ['1', '2', '3']:
            print("Opción no válida."); continue

        if opcion_principal == '3':
            url = input("\nIngresa la URL del Canal o Playlist: ")
            obtener_datos_canal_o_playlist(url)
            continue
        
        if opcion_principal == '1':
            print("\n** INSTRUCCIÓN IMPORTANTE **")
            print("Para mejores resultados, ve al canal y haz clic en la pestaña 'VIDEOS'.")
            print("Luego, copia la URL de esa página (ej: youtube.com/@canal/videos).")
            url = input("Ingresa la URL del Canal (pestaña 'videos') o Playlist: ")
        else:
            url = input("\nIngresa la URL del video: ")

        ruta_salida = input("Ruta de descarga (deja en blanco para la carpeta actual): ") or '.'
        guardar_meta = input("¿Guardar archivo .txt con metadatos? (s/n): ").lower() == 's'

        num_videos = ""
        if opcion_principal == '1':
            num_videos = input("¿Cuántos videos recientes descargar? (ej: 5, 10. Deja en blanco para TODOS): ")

        print("\n--- Opciones de Calidad ---")
        print("1. Máxima calidad (requiere FFmpeg)")
        print("2. Solo audio (MP3, requiere FFmpeg)")
        calidad_opcion = input("Elige una opción de calidad: ")

        ydl_opts = {'outtmpl': os.path.join(ruta_salida, '%(playlist_title)s/%(title).150s [%(id)s].%(ext)s' if 'playlist' in url or '/videos' in url else '%(title).150s [%(id)s].%(ext)s')}
        
        if num_videos.isdigit() and int(num_videos) > 0:
            ydl_opts['playlist_items'] = f"1-{num_videos}"

        if calidad_opcion == '1':
            ydl_opts['format'] = 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best'
        elif calidad_opcion == '2':
            ydl_opts['format'] = 'bestaudio/best'
            ydl_opts['postprocessors'] = [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'mp3', 'preferredquality': '192'}]
        else:
            print("Opción no válida. Abortando."); continue
        
        descargar_con_ytdlp(url, ydl_opts, guardar_meta)