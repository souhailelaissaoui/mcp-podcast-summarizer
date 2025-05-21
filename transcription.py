import os
import whisper
import yt_dlp
import torch
import time

# Initialisation du modèle Whisper - using tiny for faster processing
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Using device: {device} for Whisper model")
whisper_model = whisper.load_model("tiny", device)  # Using tiny model for faster processing

def download_audio(url, output_dir="temp"):
    """Télécharge l'audio d'une vidéo YouTube."""
    print(f"[INFO] Starting download of audio from {url}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Clear any existing files in temp directory
    for file in os.listdir(output_dir):
        try:
            os.remove(os.path.join(output_dir, file))
        except Exception as e:
            print(f"[WARNING] Could not remove file {file}: {e}")
    
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': f'{output_dir}/%(id)s.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'quiet': False,  # Show output for debugging
        'no_warnings': False,  # Show warnings for debugging
        'progress': True,  # Show progress
        'max_filesize': 100 * 1024 * 1024,  # 100 MB max file size
        'socket_timeout': 30,  # 30 second socket timeout
    }
    
    try:
        start_time = time.time()
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            print(f"[INFO] Extracting video info...")
            info_dict = ydl.extract_info(url, download=False)
            video_id = info_dict.get('id')
            duration = info_dict.get('duration')
            
            if duration and duration > 1800:  # 30 minutes
                return None, f"Video is too long ({duration} seconds). Maximum allowed is 30 minutes."
                
            print(f"[INFO] Downloading video {video_id} (duration: {duration}s)...")
            ydl.download([url])
            
            audio_file = f"{output_dir}/{video_id}.mp3"
            if not os.path.exists(audio_file):
                # Try to find any mp3 file that was created
                mp3_files = [f for f in os.listdir(output_dir) if f.endswith('.mp3')]
                if mp3_files:
                    audio_file = f"{output_dir}/{mp3_files[0]}"
                else:
                    return None, "Download completed but no MP3 file was found"
            
            file_size = os.path.getsize(audio_file) / (1024 * 1024)  # Size in MB
            elapsed = time.time() - start_time
            print(f"[INFO] Download completed in {elapsed:.2f}s. File size: {file_size:.2f} MB")
            
            return audio_file, None
    except Exception as e:
        print(f"[ERROR] Download failed: {str(e)}")
        return None, f"Erreur lors du téléchargement: {str(e)}"

def transcribe_audio_file(audio_file):
    """Transcrit un fichier audio avec Whisper et reporting de progression."""
    if not os.path.exists(audio_file):
        return None, f"File not found: {audio_file}"
        
    file_size = os.path.getsize(audio_file) / (1024 * 1024)  # Size in MB
    print(f"[INFO] Starting transcription of {audio_file} ({file_size:.2f} MB)")
    
    try:
        start_time = time.time()
        print(f"[INFO] Transcribing with Whisper model on {device}...")
        
        # Use a smaller chunk size if the file is large
        result = whisper_model.transcribe(audio_file)
        
        elapsed = time.time() - start_time
        text_length = len(result["text"])
        print(f"[INFO] Transcription completed in {elapsed:.2f}s. Generated {text_length} characters.")
        
        return result["text"], None
    except Exception as e:
        print(f"[ERROR] Transcription failed: {str(e)}")
        return None, f"Erreur lors de la transcription: {str(e)}"