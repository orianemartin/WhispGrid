# WhispGrid
A little script to use OpenAI's [Whisper](https://github.com/openai/whisper) and then automatically create a TextGrid to use on [Praat](https://www.fon.hum.uva.nl/praat/), saving hours of work.
Any model can be used, but I use [bofenghuang](https://huggingface.co/bofenghuang/whisper-medium-french]) fine-tuned model for French.

This script uses [Whisper Timestamped](https://github.com/linto-ai/whisper-timestamped) to align words and sentences, and uses two Tiers. 

WhispGrid supports multiple files. 
