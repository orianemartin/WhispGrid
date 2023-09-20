import whisper_timestamped as whisper
import tgt
import json

# File paths
audio_path = r""
output_path = "output.TextGrid"

# Load audio
audio = whisper.load_audio(audio_path)

# Load a whisper model

    #Fine tuned for French 
#model = whisper.load_model("bofenghuang/whisper-medium-french", device="cpu") 

    # Save CPU 
model = whisper.load_model("small", device="cpu") 

# Transcribe and fine tune 
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

    interval = tgt.Interval(start_time=float(segment["start"]), end_time= float(segment["end"]), text=segment["text"])
    sentences_tier.add_interval(interval)
    
for segment in result["segments"]:
   for word in segment["words"]:
   
    interval = tgt.Interval(start_time=float(word["start"]), end_time= float(word["end"]), text=word["text"])
    word_tier.add_interval(interval)
   

# Add Tier to TextGrid
tg.add_tier(sentences_tier)
tg.add_tier(word_tier)

# Write TextGrid
tgt.write_to_file(tg, output_path, format='short')
