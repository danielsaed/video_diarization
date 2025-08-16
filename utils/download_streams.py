import os
import yt_dlp

def mi_hook_de_progreso(d):
    """Muestra el progreso de la descarga y fusión."""
    if d['status'] == 'downloading':
        total_bytes = d.get('total_bytes') or d.get('total_bytes_estimate', 0)
        downloaded_bytes = d.get('downloaded_bytes', 0)
        speed = d.get('speed', 0) or 0
        eta = d.get('eta', 0) or 0
        
        speed_str = f"{speed / (1024*1024):.1f} MB/s" if speed > 1024*1024 else f"{speed / 1024:.1f} KB/s"
        percent_str = f"{(downloaded_bytes / total_bytes * 100):.1f}%" if total_bytes > 0 else "---%"
        
        print(f"\rDescargando: {percent_str} | Velocidad: {speed_str} | ETA: {eta}s  ", end="")

    elif d['status'] == 'finished':
        # yt-dlp llama a este hook al terminar la descarga de una parte (video o audio)
        # El mensaje de fusión se maneja mejor fuera, pero podemos indicar que una parte terminó.
        print(f"\nDescarga de un componente finalizada. Esperando para fusionar si es necesario...")
    
    elif d['status'] == 'error':
        print("\n¡Ocurrió un error durante la descarga!")

def descargar_con_ytdlp(url, opciones):
    """Función central para descargar usando yt-dlp."""
    try:
        with yt_dlp.YoutubeDL(opciones) as ydl:
            print("Iniciando proceso... yt-dlp analizará la URL.")
            print("Si es un canal, puede tardar un momento en recopilar la lista de videos.")
            ydl.download([url])
        print("\n¡Proceso de descarga finalizado con éxito!")
    except yt_dlp.utils.DownloadError as e:
        # Errores específicos de la descarga (ej. video no encontrado)
        print(f"\nError de yt-dlp: {e}")
    except Exception as e:
        # Otros errores inesperados
        print(f"\nOcurrió un error inesperado: {e}")

if __name__ == "__main__":
    while True:
        print("\n=========== Descargador de Twitch y Kick con yt-dlp ===========")
        print("1. Descargar un solo Video (VOD) o Clip")
        print("2. Descargar TODOS los VODs de un Canal")
        print("3. Salir")
        opcion_principal = input("Elige una opción: ")

        if opcion_principal == '3':
            break
        
        if opcion_principal not in ['1', '2']:
            print("Opción no válida.")
            continue

        url = input("Ingresa la URL (VOD, Clip o Canal): ")
        ruta_salida = input("Ingresa la ruta de descarga (deja en blanco para la carpeta actual): ") or '.'
        
        print("\n--- Formato de Descarga ---")
        print("1. Video (máxima calidad, requiere FFmpeg)")
        print("2. Solo audio (convertido a .mp3, requiere FFmpeg)")
        formato_opcion = input("Elige un formato: ")

        # Opciones base de yt-dlp
        ydl_opts = {
            'progress_hooks': [mi_hook_de_progreso],
            'ignoreerrors': True, # Esencial para canales, para saltar VODs borrados o con errores.
        }

        # Configurar la plantilla de nombres de archivo para organizar todo
        # Los títulos de los streams pueden ser muy largos, los limitamos a 150 caracteres
        if opcion_principal == '2': # Descarga de canal completo
             ydl_opts['outtmpl'] = os.path.join(ruta_salida, '%(uploader)s', '%(upload_date)s - [%(id)s] - %(title).150s.%(ext)s')
        else: # Descarga de VOD o Clip individual
             ydl_opts['outtmpl'] = os.path.join(ruta_salida, '[%(id)s] - %(title).150s.%(ext)s')

        if formato_opcion == '1':
            print("\nConfigurando para descarga de video en máxima calidad...")
            # 'bestvideo+bestaudio' fusiona los mejores flujos. '/best' es un fallback.
            ydl_opts['format'] = 'bestvideo+bestaudio/best'
            
        elif formato_opcion == '2':
            print("\nConfigurando para solo audio (MP3)...")
            ydl_opts['format'] = 'bestaudio/best'
            ydl_opts['postprocessors'] = [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }]
            # Asegurarse de que el nombre del archivo termine en .mp3
            ydl_opts['outtmpl'] = ydl_opts['outtmpl'].replace('.%(ext)s', '.mp3')
        else:
            print("Opción de formato no válida. Abortando.")
            continue
        
        descargar_con_ytdlp(url, ydl_opts)