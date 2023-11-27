import torch
import pandas as pd
import TTS
from TTS.TTS.api import TTS
from pydub import AudioSegment
import glob
import os
import time
import re
import spacy

import Text_Conversion
import Dialogue_Label_Without_AI

'''

This script combines functionality from the Text_Conversion and Dialogue_Label_Without_AI py files. The overall
process first takes a text file and determines which characters are speaking which pieces of dialogue. Then, it returns
this labelled data to the audio_generation functionality of Coqui-tts to automatically generate different voices
depending on the characters speaking.

What is demonstrated here is one of the simplest implementations of this process. A number of NLP preprocessing
techniques and other audio-generation models may be used for better audio q   uality and accurate labelling.

'Shire_meeting_with_music' is an example of the final product of all this code!

'''



# Specify device to use for TTS computation
device = "cuda" if torch.cuda.is_available() else "cpu"

# load in text data
# format should be a list of tuples containing (sentence, voice) like so: ('Bilbo ran away', Tolkien)
text_sample = "Hobbit_sample.txt"
characters = ['Bilbo', 'Gandalf']
narrator = 'Tolkien'

# call our text-labelling functionality to prepare the data
data = Dialogue_Label_Without_AI.entire_process(text_sample, characters, narrator)

# define the model for all generation tasks (xtts_v1 has the best results so far)
tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v1", progress_bar=True).to(device)

# call the audio conversion functions
generated_file_name = 'Shire_meeting'
Text_Conversion.make_audiobook(data, 'voices', 'input_dir', 'output_dir', generated_file_name, tts)
Text_Conversion.overlay_music('output_dir/Shire_meeting.wav', 'music/shire_music.wav', 'Shire_meeting_with_music.wav')

# ---------------------------------------------------------------------------------------------------------------------
