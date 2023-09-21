import whisper_timestamped as whisper
import tgt
import json
import tkinter as tk
from tkinter import filedialog
import os
import threading
import tkinter.messagebox as messagebox


def select_audio_files():
    file_paths = filedialog.askopenfilenames(filetypes=[("Audio files", "*.wav;*.mp3;*.mp4;*.mpeg;*.mpga;*.m4a;*.webm;*.flac;*.ogg")])
    for file_path in file_paths:
        audio_listbox.insert(tk.END, file_path)


def transcribe_audios():
    selected_files = audio_listbox.get(0, tk.END)
    base_output_name = output_entry.get()

    # Disable the "Transcribe" button while processing
    transcribe_button.config(state=tk.DISABLED)

    def on_transcription_completed():
        # Notify the user that batch transcription is complete
        audio_listbox.insert(tk.END, "Batch transcription complete")
        audio_listbox.see(tk.END)
        # Clear the listbox
        audio_listbox.delete(0, tk.END)
        # Re-enable the "Transcribe" button
        transcribe_button.config(state=tk.NORMAL)

    # Create a thread for each audio file to transcribe in the background
    transcription_threads = []
    for audio_path in selected_files:
        transcription_thread = threading.Thread(target=lambda path=audio_path, output_name=base_output_name, threads=transcription_threads: transcribe_audio_thread(path, output_name, threads))
        transcription_threads.append(transcription_thread)
        transcription_thread.start()

def transcribe_audio_thread(audio_path, output_name, transcription_threads):
    # Load audio
    audio = whisper.load_audio(audio_path)

    # Load a whisper model
    model = whisper.load_model("bofenghuang/whisper-medium-french ", device="cpu")

    # Transcribe and fine-tune
    result = whisper.transcribe(
        model,
        audio,
        language="fr",
        beam_size=5,
        best_of=5,
        temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        trust_whisper_timestamps=False
    )

    # Create TextGrid
    tg = tgt.TextGrid()

    # Create Sentences and Word tiers
    sentences_tier = tgt.IntervalTier(start_time=0, end_time=result["segments"][-1]["end"], name="phrase")
    word_tier = tgt.IntervalTier(start_time=0, end_time=result["segments"][-1]["end"], name="mot")

    # Create intervals and add them to the tier
    for segment in result["segments"]:
        interval = tgt.Interval(start_time=float(segment["start"]), end_time=float(segment["end"]), text=segment["text"])
        sentences_tier.add_interval(interval)

    for segment in result["segments"]:
        for word in segment["words"]:
            interval = tgt.Interval(start_time=float(word["start"]), end_time=float(word["end"]), text=word["text"])
            word_tier.add_interval(interval)

    # Add Tier to TextGrid
    tg.add_tier(sentences_tier)
    tg.add_tier(word_tier)

    # Determine the output file name based on the input audio file name
    input_file_name = os.path.basename(audio_path)
    output_file_name = os.path.splitext(input_file_name)[0] + ".TextGrid"
    output_path = os.path.join(os.path.dirname(audio_path), output_file_name)

    # Write TextGrid
    tgt.write_to_file(tg, output_path, format='short')

     # Notify that transcription is complete
    notify_transcription_complete(audio_path)

      # Check if all threads have completed
    if all(not thread.is_alive() for thread in transcription_threads):
        # All transcriptions are complete, call the completion callback
        on_transcription_completed()

def notify_transcription_complete(audio_path):
     # Show a popup message
    messagebox.showinfo("Transcription Complete", f"Transcription complete for {audio_path}")
    
    # Clear the listbox
    audio_listbox.delete(0, tk.END)
    
    # Re-enable the "Transcribe" button
    transcribe_button.config(state=tk.NORMAL)

# Create the main application window
app = tk.Tk()
app.title("WhispGrid")

# Create labels and entry fields
audio_label = tk.Label(app, text="Select Audio Files:")
audio_label.pack()

audio_listbox = tk.Listbox(app, selectmode=tk.MULTIPLE, width=50)
audio_listbox.pack()

output_entry = tk.Entry(app, width=50)
output_entry.pack()

# Create buttons
select_button = tk.Button(app, text="Select Audio Files", command=select_audio_files)
select_button.pack()

transcribe_button = tk.Button(app, text="Transcribe", command=transcribe_audios)
transcribe_button.pack()

app.mainloop()