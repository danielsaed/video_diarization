import os
import yt_dlp

def mi_hook_de_progreso(d):
    """Muestra el progreso de la descarga de una forma limpia."""
    if d['status'] == 'downloading':
        # Extrae la información y la formatea en una sola línea
        total_bytes = d.get('total_bytes') or d.get('total_bytes_estimate', 0)
        downloaded_bytes = d.get('downloaded_bytes', 0)
        speed = d.get('speed', 0) or 0
        eta = d.get('eta', 0) or 0
        
        # Convierte la velocidad a un formato legible
        speed_str = f"{speed / 1024:.1f} KB/s"
        if speed > 1024 * 1024:
            speed_str = f"{speed / (1024 * 1024):.1f} MB/s"

        percent_str = "---%"
        if total_bytes > 0:
            percentage = downloaded_bytes / total_bytes * 100
            percent_str = f"{percentage:.1f}%"
        
        print(f"\rDescargando: {percent_str} | Velocidad: {speed_str} | ETA: {eta}s  ", end="")

    elif d['status'] == 'finished':
        # Limpiamos la línea de progreso antes de imprimir el mensaje final
        print(f"\nDescarga de '{d['filename']}' completada.")
    elif d['status'] == 'error':
        print("\n¡Ocurrió un error durante la descarga!")

def descargar_con_ytdlp(url, opciones):
    """Función central para descargar usando yt-dlp con las opciones dadas."""
    try:
        with yt_dlp.YoutubeDL(opciones) as ydl:
            # Informamos al usuario que el proceso puede tardar, especialmente con perfiles grandes
            print("Iniciando proceso... Esto puede tardar si es un perfil con muchos videos.")
            ydl.download([url])
        print("\n¡Proceso de descarga finalizado con éxito!")
    except yt_dlp.utils.DownloadError as e:
        print(f"\nError de yt-dlp: {e}")
    except Exception as e:
        print(f"\nOcurrió un error inesperado: {e}")

if __name__ == "__main__":
    while True:
        print("\n=========== Descargador de TikTok con yt-dlp ===========")
        print("1. Descargar un solo video de TikTok")
        print("2. Descargar TODOS los videos de un usuario de TikTok")
        print("3. Salir")
        opcion_principal = input("Elige una opción: ")

        if opcion_principal == '3':
            break
        
        if opcion_principal not in ['1', '2']:
            print("Opción no válida.")
            continue

        # Lógica para las opciones 1 y 2
        if opcion_principal == '1':
            url = input("Ingresa la URL del video de TikTok: ")
        elif opcion_principal == '2':
            url = input("Ingresa la URL del perfil de usuario de TikTok (ej: https://www.tiktok.com/@nombredeusuario): ")

        ruta_salida = input("Ingresa la ruta de descarga (deja en blanco para la carpeta actual): ") or '.'
        
        print("\n--- Formato de Descarga ---")
        print("1. Video (máxima calidad)")
        print("2. Solo audio (convertido a .mp3, requiere FFmpeg)")
        formato_opcion = input("Elige un formato: ")

        # Opciones base de yt-dlp
        ydl_opts = {
            'progress_hooks': [mi_hook_de_progreso],
            'ignoreerrors': True, # Muy útil para perfiles, para que no se detenga si un video falla
        }

        # Configurar la plantilla de nombres de archivo para organizar todo
        # Si es un perfil, crea una carpeta con el nombre del usuario.
        # %(title).100s limita el título a 100 caracteres para evitar errores de "nombre de archivo muy largo".
        if opcion_principal == '2':
             ydl_opts['outtmpl'] = os.path.join(ruta_salida, '%(uploader)s', '%(upload_date)s - %(id)s - %(title).100s.%(ext)s')
        else:
             ydl_opts['outtmpl'] = os.path.join(ruta_salida, '%(id)s - %(title).100s.%(ext)s')


        if formato_opcion == '1':
            print("\nConfigurando para descarga de video...")
            # Para TikTok, 'best' es suficiente para obtener el archivo original.
            ydl_opts['format'] = 'best'
            
        elif formato_opcion == '2':
            print("\nConfigurando para solo audio (MP3)...")
            # Extrae el mejor audio y lo convierte a mp3
            ydl_opts['format'] = 'bestaudio/best'
            ydl_opts['postprocessors'] = [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192', # Calidad en kbps
            }]
            # Asegurarse de que el nombre del archivo termine en .mp3
            ydl_opts['outtmpl'] = ydl_opts['outtmpl'].replace('.%(ext)s', '.mp3')

        else:
            print("Opción de formato no válida. Abortando.")
            continue
        
        descargar_con_ytdlp(url, ydl_opts)