# WhispGrid
A little script to use OpenAI's [Whisper](https://github.com/openai/whisper) and then automatically create a TextGrid to use on [Praat](https://www.fon.hum.uva.nl/praat/). While a manual check is necessary, it saves some time. 
The app lets you choose a model from the basic OpenAI Model, or let you enter the name of a more specific model. 

This script uses [Whisper Timestamped](https://github.com/linto-ai/whisper-timestamped) to align words and sentences, and uses two Tiers, one for segments and one for words.

## Functionalities and limitations

WhispGrid supports multiple files, but has a limit of 25MB per file (defined by Whisper's API).

