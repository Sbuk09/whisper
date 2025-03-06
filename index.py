import whisper
import os
import datetime
import tqdm
import ffmpeg

# Load the model
model = whisper.load_model("medium")

# Input audio file
audio_file = "test3.m4a"

# Get audio duration using ffmpeg
def get_audio_duration(audio_path):
    try:
        probe = ffmpeg.probe(audio_path)
        duration = float(probe["format"]["duration"])
        return duration
    except Exception as e:
        print(f"Error getting duration: {e}")
        return None

# Get audio duration
audio_duration = get_audio_duration(audio_file)
if audio_duration:
    print(f"🔄 Transcribing: {audio_file} | Duration: {audio_duration:.2f} seconds")

# Track progress manually
with tqdm.tqdm(total=audio_duration, unit="sec", desc="Processing", dynamic_ncols=True) as pbar:
    result = model.transcribe(audio_file, language="isiZulu", verbose=True)
    pbar.update(audio_duration)  # Simulate progress

# Get current timestamp for file naming
current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
transcription_filename = f"transcription_{current_time}.txt"

# Save transcript with metadata
with open(transcription_filename, "w", encoding="utf-8") as f:
    f.write(f"🕒 Transcription Date: {current_time}\n")
    f.write(f"🎵 Audio File: {audio_file}\n")
    f.write(f"⏳ Duration: {audio_duration:.2f} seconds\n\n")
    f.write("📜 Transcription:\n")
    f.write(result["text"])

print(f"\n✅ Transcription saved as: {transcription_filename}")
