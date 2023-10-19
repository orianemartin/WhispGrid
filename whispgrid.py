import whisper_timestamped as whisper
import tgt
import json
import tkinter as tk
from tkinter import filedialog, StringVar, OptionMenu, messagebox, ttk, Entry, simpledialog
from tkinter.simpledialog import askstring
from tktooltip import ToolTip
import sv_ttk
import os
import time

import threading
import datetime
import subprocess

import torch
import pyannote.audio

from pyannote.audio import Audio
from pyannote.core import Segment

from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
embedding_model = PretrainedSpeakerEmbedding(
    "speechbrain/spkrec-ecapa-voxceleb",
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

import wave
import contextlib

from sklearn.cluster import AgglomerativeClustering
import numpy as np

predefined_models = ["small", "base", "medium", "large", "bofenghuang/whisper-medium-french"]
predefined_languages = ["en", "fr", "es", "de"]
num_speakers = 0 
initials = []

st = time.process_time()


def select_audio_files():
    file_paths = filedialog.askopenfilenames(filetypes=[("Audio files", "*.wav;*.mp3;*.mp4;*.mpeg;*.mpga;*.m4a;*.webm;*.flac;*.ogg")])
    for file_path in file_paths:
        audio_listbox.insert(tk.END, file_path)

def transcribe_audios():
    start_time = time.time() 
    selected_files = audio_listbox.get(0, tk.END)
    selected_model = model_var.get()
    selected_language = language_var.get()
    num_speakers = speaker_entry.get()

    # Check if num_speakers is empty or 0
    if not num_speakers or int(num_speakers) == 0:
        messagebox.showerror("Invalid Input", "Please enter a valid number of speakers.")
        return

    if selected_model == "other":
        custom_model = askstring("Custom Model", "Enter a custom model name:")
        if custom_model is None:
            return
        selected_model = custom_model

    num_speakers = int(num_speakers)

    if selected_language == "other":
        custom_language = askstring("Custom Language", "Enter the language code:")
        if selected_language is None:
            return
        selected_language = custom_language

    for i in range(0, num_speakers):
        initial = simpledialog.askstring("Speaker Name or Initials", f"Enter Name or Initials for Speaker {i + 1}:")
        if initial is None:
            return
        initials.append(initial)


    transcribe_button.config(state=tk.DISABLED)

    def format_time(seconds):
        # Convert seconds to hours, minutes, and seconds
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        return f"{int(h):02d}:{int(m):02d}:{int(s):02d}"

    def on_transcription_completed():
        end_time = time.time()
        elapsed_time = end_time - start_time  # Calculate the elapsed time
        formatted_time = format_time(elapsed_time)
        et = time.process_time()
        res = et - st
        messagebox.showinfo("Transcription Complete", f"Batch transcription complete\nElapsed Time: {formatted_time} \nCPU execution time: {res} seconds")
        audio_listbox.see(tk.END)
        audio_listbox.delete(0, tk.END)
        transcribe_button.config(state=tk.NORMAL)
        


    transcription_threads = []
    for audio_path in selected_files:
        transcription_thread = threading.Thread(target=transcribe_audio_thread, args=(audio_path, selected_model, on_transcription_completed, transcription_threads, selected_language, int(num_speakers)))
        transcription_threads.append(transcription_thread)
        transcription_thread.start()

    def check_transcription_completion():
        if all(not thread.is_alive() for thread in transcription_threads):
            on_transcription_completed()
        else:
            app.after(1000, check_transcription_completion)

    check_transcription_completion()

def transcribe_audio_thread(audio_path, selected_model, on_transcription_completed, transcription_threads, selected_language, num_speakers):
    
    original_file_name, original_file_ext = os.path.splitext(os.path.basename(audio_path))

    audio = whisper.load_audio(audio_path)

    model = whisper.load_model(selected_model, device="cpu")

    result = whisper.transcribe(
        model,
        audio,
        language=selected_language,
        beam_size=5,
        best_of=5,
        temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        trust_whisper_timestamps=False
    )

    segments = result["segments"]
    previous_end_time = 0.0


#Create TextGrid and edit it

    tg = tgt.TextGrid()

    sentences_tier = tgt.IntervalTier(start_time=0, end_time=result["segments"][-1]["end"], name="phrase")
    word_tier = tgt.IntervalTier(start_time=0, end_time=result["segments"][-1]["end"], name="mot")

    #work on words before it's too late
    for segment in result["segments"]:
        if "words" in segment:
            for word in segment["words"]:
                interval = tgt.Interval(start_time=float(word["start"]), end_time=float(word["end"]), text=word["text"])
                word_tier.add_interval(interval)

    if int(num_speakers) > 1:

        if audio_path[-3:] != 'wav':
            subprocess.call(['ffmpeg', '-i', audio_path, f'{original_file_name}.wav', '-y'])
            audio_path = f'{original_file_name}.wav'

        with contextlib.closing(wave.open(audio_path,'r')) as f:
            frames = f.getnframes()
            rate = f.getframerate()
            duration = frames / float(rate)

        audio_dia = Audio()

        def segment_embedding(segment):
            start = segment["start"]
            # Whisper overshoots the end timestamp in the last segment
            end = min(duration, segment["end"])
            clip = Segment(start, end)
            waveform, sample_rate = audio_dia.crop(audio_path, clip)
            return embedding_model(waveform[None])

        embeddings = np.zeros(shape=(len(segments), 192))
        for i, segment in enumerate(segments):
            embeddings[i] = segment_embedding(segment)

        embeddings = np.nan_to_num(embeddings)

        clustering = AgglomerativeClustering(num_speakers).fit(embeddings)
        labels = clustering.labels_
        for i in range(len(segments)):
            segments[i]["speaker"] = str(labels[i] + 1)
            try:
                speaker_initial = initials[int(segments[i]["speaker"]) - 2]
            except IndexError:
                # Handle the case where the label is out of range
                speaker_initial = "Unknown"
            #speaker_initial = initials[labels[i] + 1]
            #segments[i]["speaker"] = speaker_initial
            #segments[i]["speaker"] = str(labels[i] + 1)

            concatenated_text = f"{speaker_initial} {segments[i]['text']}"
            interval = tgt.Interval(start_time=float(segments[i]["start"]), end_time=float(segments[i]["end"]), text=concatenated_text)
            sentences_tier.add_interval(interval)

    else:
        for segment in result["segments"]:
            interval = tgt.Interval(start_time=float(segment["start"]), end_time=float(segment["end"]), text=segment["text"])
            sentences_tier.add_interval(interval)


    tg.add_tier(sentences_tier)
    tg.add_tier(word_tier)

    input_file_name = os.path.basename(audio_path)
    output_file_name = os.path.splitext(input_file_name)[0] + ".TextGrid"
    output_path = os.path.join(os.path.dirname(audio_path), output_file_name)

    tgt.write_to_file(tg, output_path, format='short')

    if all(not thread.is_alive() for thread in transcription_threads):
        on_transcription_completed()

# App making

app = tk.Tk()
app.iconbitmap("audio-wave-32.ico")

sv_ttk.set_theme("light")

app.title("WhispGrid")

audio_label = ttk.Label(app, text="Selected Audio Files")
audio_label.pack()

audio_listbox = tk.Listbox(app, selectmode=tk.MULTIPLE, width=100)
audio_listbox.pack()

select_button = ttk.Button(app, text="Select Audio Files", command=select_audio_files)
select_button.pack()

model_label = ttk.Label(app, text="Select or Enter a Model:")
model_label.pack()
ToolTip(model_label, msg="Open Ai Whisper Model or custom Model. Please note that diarization seems to work only with Whisper's models.")


model_var = StringVar(app)
model_var.set(predefined_models[0])
model_option_menu = OptionMenu(app, model_var, *predefined_models, "other")
model_option_menu.pack()

#language to use

language_label = ttk.Label(app, text="Select or Enter a Language:")
language_label.pack()
ToolTip(language_label, msg="Please use the initials of the language as defined in Whisper's API.")


language_var = StringVar(app)
language_var.set(predefined_languages[0])
language_option_menu = OptionMenu(app, language_var, *predefined_languages, "other")
language_option_menu.pack()

#number of speakers

speaker_label = ttk.Label(app, text="Enter the Number of Speakers:")
speaker_label.pack()
ToolTip(speaker_label, msg="You will then be asked for speakers' names. If there is only one speaker, no diarization will be performed.")


speaker_entry = Entry(app)
speaker_entry.pack()

transcribe_button = ttk.Button(app, text="Transcribe", command=transcribe_audios)
transcribe_button.pack()

app.mainloop()