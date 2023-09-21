import whisper_timestamped as whisper
import tgt
import json
import tkinter as tk
from tkinter import filedialog, StringVar, OptionMenu, messagebox, ttk
from tkinter.simpledialog import askstring
import sv_ttk
import os
import threading

predefined_models = ["tiny", "small", "base", "medium", "large", "bofenghuang/whisper-medium-french"]

def select_audio_files():
    file_paths = filedialog.askopenfilenames(filetypes=[("Audio files", "*.wav;*.mp3;*.mp4;*.mpeg;*.mpga;*.m4a;*.webm;*.flac;*.ogg")])
    for file_path in file_paths:
        audio_listbox.insert(tk.END, file_path)

def transcribe_audios():
    selected_files = audio_listbox.get(0, tk.END)
    selected_model = model_var.get()

    if selected_model == "other":
        custom_model = askstring("Custom Model", "Enter a custom model name:")
        if custom_model is None:
            return
        selected_model = custom_model

    transcribe_button.config(state=tk.DISABLED)

    def on_transcription_completed():
        messagebox.showinfo("Transcription Complete", "Batch transcription complete")
        audio_listbox.see(tk.END)
        audio_listbox.delete(0, tk.END)
        transcribe_button.config(state=tk.NORMAL)

    transcription_threads = []
    for audio_path in selected_files:
        transcription_thread = threading.Thread(target=transcribe_audio_thread, args=(audio_path, selected_model, on_transcription_completed, transcription_threads))
        transcription_threads.append(transcription_thread)
        transcription_thread.start()

    def check_transcription_completion():
        if all(not thread.is_alive() for thread in transcription_threads):
            on_transcription_completed()
        else:
            app.after(1000, check_transcription_completion)

    check_transcription_completion()

def transcribe_audio_thread(audio_path, selected_model, on_transcription_completed, transcription_threads):
    audio = whisper.load_audio(audio_path)

    model = whisper.load_model(selected_model, device="cpu")

    result = whisper.transcribe(
        model,
        audio,
        language="fr",
        beam_size=5,
        best_of=5,
        temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        trust_whisper_timestamps=False
    )

    tg = tgt.TextGrid()

    sentences_tier = tgt.IntervalTier(start_time=0, end_time=result["segments"][-1]["end"], name="phrase")
    word_tier = tgt.IntervalTier(start_time=0, end_time=result["segments"][-1]["end"], name="mot")

    for segment in result["segments"]:
        interval = tgt.Interval(start_time=float(segment["start"]), end_time=float(segment["end"]), text=segment["text"])
        sentences_tier.add_interval(interval)

    for segment in result["segments"]:
        for word in segment["words"]:
            interval = tgt.Interval(start_time=float(word["start"]), end_time=float(word["end"]), text=word["text"])
            word_tier.add_interval(interval)

    tg.add_tier(sentences_tier)
    tg.add_tier(word_tier)

    input_file_name = os.path.basename(audio_path)
    output_file_name = os.path.splitext(input_file_name)[0] + ".TextGrid"
    output_path = os.path.join(os.path.dirname(audio_path), output_file_name)

    tgt.write_to_file(tg, output_path, format='short')

    if all(not thread.is_alive() for thread in transcription_threads):
        on_transcription_completed()

app = tk.Tk()

sv_ttk.set_theme("light")

app.title("WhispGrid")

audio_label = ttk.Label(app, text="Select Audio Files:")
audio_label.pack()

audio_listbox = tk.Listbox(app, selectmode=tk.MULTIPLE, width=100)
audio_listbox.pack()

model_label = ttk.Label(app, text="Select or Enter a Model:")
model_label.pack()

model_var = StringVar(app)
model_var.set(predefined_models[0])
model_option_menu = OptionMenu(app, model_var, *predefined_models, "other")
model_option_menu.pack()

select_button = ttk.Button(app, text="Select Audio Files", command=select_audio_files)
select_button.pack()

transcribe_button = ttk.Button(app, text="Transcribe", command=transcribe_audios)
transcribe_button.pack()

app.mainloop()
