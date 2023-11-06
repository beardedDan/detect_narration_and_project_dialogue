import spacy
nlp = spacy.load("en_core_web_sm")
import re

def remove_special_characters(text, remove_digits=False):
    pattern = r'[^a-zA-z0-9\s"]' if not remove_digits else r'[^A-Za-z\s"]'
    text = re.sub(pattern, ' ', text) # Remove special characters
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces    
    return text


def named_persons_w_spacy(text):
    text_nlp = nlp(text)
    ner_tagged = [(word.text,word.ent_type_) for word in text_nlp]
    named_entities = []
    temp_entity_name = ''
    temp_named_entity = None
    for term, tag in ner_tagged:
        if tag:
            temp_entity_name = ''.join([temp_entity_name, term]).strip()
            temp_named_entity = (temp_entity_name, tag)
        else:
            if temp_named_entity:
                named_entities.append(temp_named_entity)
                temp_entity_name = ''
                temp_named_entity = None
    filtered_entities = [entity for entity in named_entities if entity[1] == "PERSON"]
    characters = []
    for item in filtered_entities:
        characters.append(item[0])
    characters = list(set(characters))
    num_characters = len(characters)
    return characters
