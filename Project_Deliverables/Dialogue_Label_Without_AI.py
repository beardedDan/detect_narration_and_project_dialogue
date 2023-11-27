import re
import spacy

nlp = spacy.load('en_core_web_sm')

# This file contains helper functions used to classify which dialogue is spoken by which character


# define a function for splitting at quotes
def split_sentence_with_quotation(sentence):
    doc = nlp(sentence)

    parts = []
    current_part = ""

    for token in doc:
        if token.text == '”':
            current_part += token.text
            parts.append(current_part)
            current_part = ""
        else:
            current_part += token.text + token.whitespace_

    if current_part:
        parts.append(current_part.strip())

    return parts


# define a function for breaking text according to changes in dialogue and narration
def break_text(text):
    splits = nlp(text)
    sentences = [sentence.text for sentence in splits.sents]

    all_parts = []
    for sentence in sentences:
        sentence = split_sentence_with_quotation(sentence)
        for part in sentence:
            all_parts.append(part)

    return all_parts


# define a function that splits the text into groups according to who is speaking or narrator sections
# this takes the output of the break_text function
def break_by_character(broken_up_parts):
    dialogue_starters = []
    dialogue_enders = []
    for part in broken_up_parts:
        if ('“' in part):
            dialogue_starters.append(part)
        if ('”' in part):
            dialogue_enders.append(part)

    all_globs = []
    micro = []
    for part in broken_up_parts:
        if part in dialogue_starters and part in dialogue_enders:
            if len(micro) > 0:
                all_globs.append(micro)
                micro = []
            all_globs.append([part])

        elif part in dialogue_starters:
            if len(micro) > 0:
                all_globs.append(micro)
                micro = []
            micro.append(part)

        elif part in dialogue_enders:
            if len(micro) > 0:
                micro.append(part)
                all_globs.append(micro)
                micro = []

        else:
            micro.append(part)

    return [item for item in all_globs if item != ['']]


# define a function to turn the character sections into a dictionary for easier processing
def make_dictionary(voice_sections):
    my_dict = {}
    for index, section in enumerate(voice_sections):
        section = " ".join(section)
        my_dict[index] = section

    return my_dict


# define a function that labels the dialogue based on neighboring Names and return as a dictionary
def label_dialogue(sections, characters_of_story, do_print):
    last_speaker = ''
    dialogue_partner = ''
    dialogue_dictionary = {}
    for key, value in sections.items():
        if ('“' in value and '”' in value):

            if (0 < key + 1 < len(sections)) and '“' not in sections[key + 1] and '”' not in sections[key + 1]:
                in_front = sections[key + 1]
            else:
                in_front = "None"
            if (0 < key - 1 < len(sections)) and '“' not in sections[key - 1] and '”' not in sections[key - 1]:
                behind = sections[key - 1]
            else:
                behind = "None"

            current_dialogue = value

            current_voice = ''

            possible_voices = []
            for name in characters_of_story:
                if name in in_front:
                    possible_voices.append(name)
            if len(possible_voices) > 0:
                current_voice = possible_voices[0]

            if len(possible_voices) < 1:
                for name in characters_of_story:
                    if name in behind:
                        possible_voices.append(name)
                if len(possible_voices) > 0:
                    current_voice = possible_voices[-1]

            if len(possible_voices) == 0:
                current_voice = 'None'

            if current_voice == 'None' and behind != 'None':
                current_voice = last_speaker

            if current_voice == 'None':
                current_voice = dialogue_partner

            if do_print:
                print(behind)
                print(key, current_dialogue)
                print(in_front)
                print("")
                print('current voice:', current_voice)
                print('last speaker:', last_speaker)
                print('dialogue partner:', dialogue_partner)
                print('-----------')

            if current_voice != last_speaker:
                dialogue_partner = last_speaker

            last_speaker = current_voice

            dialogue_dictionary[current_dialogue] = current_voice

    return dialogue_dictionary


# define a function to combine narrated parts with labelled dialogue
def label_all_text(sections, dialogue_dict, narrator_of_all):
    final_tuple = []
    for key, value in sections.items():
        if ('“' in value and '”' in value):
            final_tuple.append((value, dialogue_dict[value]))
        else:
            final_tuple.append((value, narrator_of_all))
    return final_tuple


# function containing the entire process
def entire_process(txt_file_name, characters_in_story, narrator_of_story):
    my_text = ""
    with open(txt_file_name, encoding="utf8") as f:
        lines = f.readlines()
    for line in lines:
        my_text = my_text + (line.strip("\n") + " ")

    my_voice_sections = break_by_character(break_text(my_text))
    my_voice_dic = make_dictionary(my_voice_sections)
    my_dialogue_dictionary = label_dialogue(my_voice_dic, characters_in_story, False)

    my_labelled_text = label_all_text(my_voice_dic, my_dialogue_dictionary, narrator_of_story)

    return my_labelled_text
