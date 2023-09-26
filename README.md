# WhispGrid
A little script to use OpenAI's [Whisper](https://github.com/openai/whisper) and then automatically create a TextGrid to use on [Praat](https://www.fon.hum.uva.nl/praat/). While a manual check is necessary, it saves some time. 
The app lets you choose a model from the basic OpenAI Model, or lets you enter the name of a more specific model. The language can also be chosen.

This script uses [Whisper Timestamped](https://github.com/linto-ai/whisper-timestamped) to align words and sentences, and uses two Tiers, one for segments and one for words. 
WhispGrid supports diarization based on [Riteshhere](https://github.com/riteshhere/Speaker_diarization)'s code. 

## Functionalities and limitations

WhispGrid supports multiple files but has a limit of 25MB per file (defined by Whisper's API).
- Language Choice
- Model Choice
- Diarization (choice of number of speakers)
