import torch
import pandas as pd
import TTS
from TTS.TTS.api import TTS
from pydub import AudioSegment
import glob
import os
import time
import re


# checks to make sure all voices are represented for each character (otherwise program will fail half-way through)
def check_voices(our_data, voices_dir):
    need_to_fix = False
    unique_voices = []
    [unique_voices.append(sample[1]) for sample in our_data if sample[1] not in unique_voices]
    to_check = []
    if os.path.exists(voices_dir) and os.path.isdir(voices_dir):
        file_names = os.listdir(voices_dir)
        for name in file_names:
            to_check.append(name.rstrip(".wav"))
    for voice in unique_voices:
        if voice not in to_check:
            print("WARNING:", voice, "is not in the voices directory!!!")
            print("This voice must be added manually to the directory!\n")
            need_to_fix = True
    if need_to_fix:
        return True
    else:
        return False


# generates audio in different voices from list of tuples of format: (sentence, voice)
def generate_audio(our_data, voices_dir, save_to_dir, tts_model):
    total_number_of_samples = len(our_data)
    iteration = 0
    for sample in our_data:
        iteration = iteration + 1
        sentence, voice = sample[0], sample[1]
        print("GENERATING SAMPLE", iteration, "/", total_number_of_samples)
        tts_model.tts_to_file(text=sentence,
                        speaker_wav=f"{voices_dir}/{voice}.wav",
                        language='en',
                        file_path=f"{save_to_dir}/{iteration}.wav")


# concatenates all audio into a single wav file
def join_audio(dir_of_files_to_join, output_directory, name_of_output):
    audio_clips = []
    for file in os.listdir('input_dir'):
        audio_clips.append(f"{'input_dir'}\\{file}")

    def sort_by_numbers(s):
        return [int(x) if x.isdigit() else x for x in re.split('([0-99999999]+)', s)]

    sorted_list = sorted(audio_clips, key=sort_by_numbers)
    file_paths = sorted_list
    output_audio = AudioSegment.silent(duration=0)
    print(file_paths)
    for file_path in file_paths:
        current_audio = AudioSegment.from_file(file_path, format="wav")
        output_audio += current_audio + AudioSegment.silent(duration=400)
        del current_audio
        os.remove(file_path)
    output_audio = output_audio[:-400]
    output_audio.export(os.path.join(output_directory, f"{name_of_output}.wav"), format="wav")


# final function containing entire audio generation process
def make_audiobook(text, voices_directory, generation_space, output_directory, name_of_output, tts_model):
    if not check_voices(text, voices_directory):
        to_check = []
        if os.path.exists(output_directory) and os.path.isdir(output_directory):
            file_names = os.listdir(output_directory)
            for name in file_names:
                to_check.append(name)
        if f"{name_of_output}.wav" in to_check:
            print("File name already used. Please change the name of the output file.")
            return

        start_time = time.time()
        generate_audio(text, voices_directory, generation_space, tts_model)
        join_audio(generation_space, output_directory, name_of_output)
        end_time = time.time()
        total_time = end_time - start_time
        print("Audiobook named", name_of_output, "successfully generated!")
        print("Generation time:", total_time / 60, "minutes")
    else:
        print("Voices directory must have all necessary voices!")


# puts music on the audiotrack - just for fun
def overlay_music(audio_file_path, music_file_path, file_name):
    main_audio = AudioSegment.from_file(audio_file_path, format="wav")
    music = AudioSegment.from_file(music_file_path, format="wav")

    if main_audio.frame_rate != music.frame_rate:
        music = music.set_frame_rate(main_audio.frame_rate)

    duration = len(main_audio)
    music = music[:duration]
    music = music - 10
    result = main_audio.overlay(music)

    result.export(file_name, format="wav")

# ---------------------------------------------------------------------------------------------------------------------

'''
make_audiobook:
                text                =   the text to generate from in format (sentence, voice)
                voices_directory    =   the directory containing all available voices to generate from
                generation_space    =   a directory where clips will be temporarily saved while generating
                output_directory    =   the directory where the final audiobook will be generated
                name_of_output      =   the name of the audiobook file to be generated
'''
