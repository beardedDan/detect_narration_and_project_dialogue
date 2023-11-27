import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1" 
import tensorflow as tf
tf.load_library("/etc/alternatives/libcudnn_so")

current_directory = os.getcwd()
import spacy
nlp = spacy.load("en_core_web_sm")
import re
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from itertools import groupby
from operator import itemgetter
from datasets import load_dataset
from datetime import datetime
today = datetime.now()

import requests
import zipfile
import random
random.seed(123)
import nltk

from collections import Counter
import matplotlib.pyplot as plt
import csv
from itertools import groupby
from sklearn.preprocessing import LabelBinarizer
label_binarizer = LabelBinarizer()


strategy = tf.distribute.MirroredStrategy()
from tensorflow import keras   
from keras.optimizers import Adam
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint




from transformers import AutoTokenizer
bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
bert_vocab_size = len(bert_tokenizer)

from operator import itemgetter
import multiprocessing
from concurrent.futures import ThreadPoolExecutor


import AI574Project




###################################################################################################
# /////////////////////////////////////////////////////////////////////////////////////////////////
# Begin Global Methods and Variables
# /////////////////////////////////////////////////////////////////////////////////////////////////
###################################################################################################



def remove_special_characters(text, remove_digits=False):
    pattern = r'[^a-zA-z0-9\s\n"]' if not remove_digits else r'[^A-Za-z\s\n"]'
    text = text.replace('\r','\n') # Replace carriage returns with new line
    text = re.sub(pattern, ' ', text) # Remove special characters
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces    
    return text

def consolidate_double_quotes(text):
    pattern = r'["“”〝〞‟＂]'    
    text = re.sub(pattern,' " ', text)
    text = re.sub(r'\s+', ' ', text)   
    return text

def consolidate_double_quotes_w_direction(text):
    pattern = r'["“”〝〞‟＂]'
    text = re.sub(r'(?<=\S)' + pattern, ' <-- " ', text)
    text = re.sub(pattern + r'(?=\S)', ' " --> ', text)
    text = re.sub(pattern, ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text

def insert_newline_every_n_words(text, n=20):
    words = text.split()
    for i in range(n, len(words), n):
        words.insert(i, '\n')
        i += 1 
    return ' '.join(words)

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


def conll_tag_lookup_table():
    iob_labels = ["B", "I"]
    ner_labels = ["PER", "ORG", "LOC", "MISC"]
    all_labels = [(label1, label2) for label2 in ner_labels for label1 in iob_labels]
    all_labels = ["-".join([a, b]) for a, b in all_labels]
    all_labels = ["[PAD]", "O"] + all_labels
    return dict(zip(range(0, len(all_labels) + 1), all_labels))

MAPPING = conll_tag_lookup_table()
NUM_TAGS = len(MAPPING)
VOCAB_PATH = os.path.join(current_directory,'vocabulary.txt')
FDATE = today.strftime("%Y%m%d")


def plot_training_charts(history):
    
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')

    plt.tight_layout()
    plt.show()


###################################################################################################
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
# End Global Methods and Variables
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
###################################################################################################      










###################################################################################################
# /////////////////////////////////////////////////////////////////////////////////////////////////
# Begin Statistical NER Model
# /////////////////////////////////////////////////////////////////////////////////////////////////
###################################################################################################

UNIQUE_NAMES_LIST = []


class QuotationIndicator():
    
    def __init__(self, num_threads=1, **kwargs):
        self.pattern = re.compile(r'(".*?")')
        self.num_threads = num_threads
        for key, value in kwargs.items():
            setattr(self, key, value) 

    def process_chunk(self, chunk):
        matches = [(m.start(1), m.end(1)) for m in self.pattern.finditer(chunk)]
        words = chunk.split()
        indicators = [0] * len(words)
        word_start = 0
        for i, word in enumerate(words):
            word_end = word_start + len(word)
            if any(start <= word_start < end for start, end in matches):
                indicators[i] = 1
            word_start = word_end + 1
        return indicators

    def combine_text_and_indicators(self, text, indicators):
        text = text.split()
        word_indicators = list(zip(text, indicators))
        return word_indicators
    
    def group_consecutive_words(self, text_w_word_indicators):
        grouped_words = []
        for key, group in groupby(text_w_word_indicators, key=itemgetter(1)):
            words = [word for word, _ in group]
            grouped_words.append((words, key))
        return grouped_words   
    
    def indicators_for_sentence(self, text):
        if not isinstance(text, str):
            raise ValueError("indicators for sentence function input must be a string")
        text = consolidate_double_quotes_w_direction(text)
        words = text.split()
        num_threads = self.num_threads
        chunk_size = len(words) // num_threads
        chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
        indicators = [0] * len(words)
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            results = list(executor.map(self.process_chunk, chunks))
        flat_indicators = [indicator for sublist in results for indicator in sublist]
        quotes_at_word_level = self.combine_text_and_indicators(text,flat_indicators[:len(words)])
        quotes_at_sentence_level = self.group_consecutive_words(quotes_at_word_level)
        return quotes_at_sentence_level
        
    def quote_tree(self, text):
        tree_str = ""
        for group in self.indicators_for_sentence(text):
            words, indicator = group
            if indicator == 1:
                quoted_group = ' '.join([f"{word}" for word in words])
                tree_str += f"(QUOTATION {quoted_group})"
            else:
                tree_str += ' '.join(words)
        tree_str = '(' + tree_str.strip() + ')'
        return tree_str
    
    def per_tree(self, text):
        tree_str = ""
        for group in text:
            words, indicator = group
            if indicator == '1': 
                quoted_group = ' '.join([f"{word}" for word in words])
                tree_str += f"(PER {quoted_group})"
            else:
                tree_str += ' '.join(words) + " "
        tree_str = tree_str.strip()
        return tree_str


class StatisticalNameData():
    def __init__(self, **kwargs):
        super().__init__()
        self.name_pattern = re.compile(r'^[a-zA-Z]+$')
        self.unique_names_in_file = set()
        self.ssa_url = 'https://www.ssa.gov/oact/babynames/names.zip'
        self.pronouns = ['he','she','they','I']
        self.person_titles = [
            'Mr','Mrs','Miss','Ms','Miss','Sir',
            'Madam','Dr','Doctor','Prof',
            'Professor','Rev','Reverend','Capt',
            'Captain','Lt','Lieutenant','Sgt',
            'Sergeant','Col','Colonel'
        ]        
        
        for key, value in kwargs.items():
            setattr(self, key, value)        

    def is_valid_name(self, name):
        return isinstance(name, str) and name.strip() != '' and self.name_pattern.match(name)
    
    def is_name(self, word):
        global UNIQUE_NAMES_LIST
        return word in UNIQUE_NAMES_LIST

    def download_and_unzip_data(self, zip_url, target_directory):
        os.makedirs(target_directory, exist_ok=True)
        zip_file_path = os.path.join(target_directory, 'names.zip')
        response = requests.get(zip_url)
        with open(zip_file_path, 'wb') as zip_file:
            zip_file.write(response.content)
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(target_directory)
        os.remove(zip_file_path)

    def create_common_name_dataset(self, data_directory='./ssa_names', num_top_names=1000, min_year=1880, max_year=2023):
        global UNIQUE_NAMES_LIST
        self.num_top_names = num_top_names
        self.min_year = min_year
        self.max_year = max_year
        
        unique_names = set()
        self.download_and_unzip_data(self.ssa_url, data_directory)
        for year in range(self.min_year, self.max_year):
            filename = os.path.join(data_directory, f'yob{year}.txt')
            try:
                with open(filename, 'r') as file:
                    for line in file:
                        name, gender, count = line.strip().split(',')
                        name = name
                        if int(count) > self.num_top_names and self.is_valid_name(name):
                            unique_names.add(name)
                        else:
                            pass
            except FileNotFoundError:
                print(f"File {filename} not found.")
        UNIQUE_NAMES_LIST = sorted(list(unique_names))
        return UNIQUE_NAMES_LIST, self.pronouns, self.person_titles

    def append_known_names(self, cust_names):
        global UNIQUE_NAMES_LIST
        UNIQUE_NAMES_LIST = UNIQUE_NAMES_LIST + cust_names
        return UNIQUE_NAMES_LIST

    def ner_process_line(self, text):
        clean_pattern = re.compile(r'[^a-zA-Z\s]')
        special_char_pattern = re.compile(r'[^\w\s]')
        clean_words = []
        line_scores = []
        previous_word = None
        previous_dirty_word = ''
        
        words = text.split()        
        for dirty_word in words:
            cleaned_word = clean_pattern.sub('', dirty_word)
            line_scores.append(str(int(self.is_name(cleaned_word))))
            clean_words.append(cleaned_word)
            if cleaned_word not in self.pronouns:
                if cleaned_word.istitle() and cleaned_word not in self.person_titles and previous_word in self.person_titles:
                    self.unique_names_in_file.add(f"{previous_word} {cleaned_word}")
                    line_scores[-2] = '1'
                    line_scores[-1] = '1'
                elif cleaned_word.istitle() and special_char_pattern.search(previous_dirty_word) is None and self.is_name(previous_word):
                    self.unique_names_in_file.add(f"{previous_word} {cleaned_word}")
                    line_scores[-2] = '1'
                    line_scores[-1] = '1'
                elif self.is_name(cleaned_word):
                    self.unique_names_in_file.add(f"{cleaned_word}")
            previous_word = cleaned_word
            previous_dirty_word = dirty_word
        
        return words, line_scores
    
    
###################################################################################################
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
# End Statistical NER Model
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
###################################################################################################    










###################################################################################################
# /////////////////////////////////////////////////////////////////////////////////////////////////
# Begin Deep Learning NER Model
# /////////////////////////////////////////////////////////////////////////////////////////////////
###################################################################################################



class ImportNERData():
    def __init__(self, vocab_size=20000,source='conll',**kwargs):
        super().__init__()
        self.vocab_size=vocab_size
        self.source=source
        self.lookup_layer = None
        self.vocabulary = None
        
        for key, value in kwargs.items():
            setattr(self, key, value)        
            
    def export_to_file(self,export_file_path, data):
        with open(export_file_path, "w") as f:
            for record in data:
                ner_tags = record["ner_tags"]
                tokens = record["tokens"]
                if len(tokens) > 0:
                    binary_tags = self.per_tags(ner_tags)
                    f.write(
                        str(len(tokens))
                        + "\t"
                        + "\t".join(tokens)
                        + "\t"
                        + "\t".join(map(str, binary_tags))
                        + "\n"
                    )

    def per_tags(self,nertags):
        return [1 if x in (1,2) else 0 for x in nertags]

    def write_vocab_to_file(self,file_path,vocab):
        if not os.path.exists(file_path):
            with open(file_path, 'w') as file:
                file.write("This is a new file.")
            print(f"File {file_path} created successfully.")
        else:
            print(f"File {file_path} exists. Overwriting existing file.")

        with open(file_path, 'w', encoding='utf-8') as file:
            for item in vocab:
                file.write(item+"\n")

    def write_training_and_validation_data_to_file(self,training,validation):
        if os.path.exists("./data"):
            self.export_to_file("./data/train.txt", training)
            self.export_to_file("./data/val.txt", validation)
            print("training and validation data saved to ",os.path.join(current_directory,'data'))
        else:
            print("Creating directory for training and validation data: ",os.path.join(current_directory,'data'))
            os.mkdir("data")
            self.export_to_file("./data/train.txt", training)
            self.export_to_file("./data/val.txt", validation)
            print("training and validation data saved to ",os.path.join(current_directory,'data'))
            
    def lowercase_and_convert_to_ids(self, tokens):
        tokens = tf.strings.lower(tokens)
        tokens = self.lookup_layer(tokens)
        return tokens
            
    def map_record_to_training_data(self,record):
        record = tf.strings.split(record, sep="\t")
        length = tf.strings.to_number(record[0], out_type=tf.int32)
        tokens = record[1 : length + 1]
        tags = record[length + 1 :]
        tags = tf.strings.to_number(tags, out_type=tf.int64)
        tags += 1
        return tokens, tags   

    def import_data(self):
        if self.source == "conll":
            conll_data = load_dataset("conll2003")
            print("Importing conll data")
            self.write_training_and_validation_data_to_file(conll_data["train"],conll_data["validation"])
            train_data = tf.data.TextLineDataset("./data/train.txt")
            val_data = tf.data.TextLineDataset("./data/val.txt")
            all_tokens = sum(conll_data["train"]["tokens"], [])
            all_tokens_string = ' '.join(all_tokens)  # Join all tokens into a single string separated by spaces
            with open('./data/tokens.txt', 'w', encoding='utf-8') as f:
                f.write(all_tokens_string)
            all_tokens_array = np.array(list(map(str.lower, all_tokens)))
            counter = Counter(all_tokens_array)  
            vocabulary = [token for token, count in counter.most_common(self.vocab_size - 2)]  
            self.lookup_layer = keras.layers.StringLookup(vocabulary=vocabulary)
            self.write_vocab_to_file(VOCAB_PATH,vocabulary)

            batch_size = 128
            train_dataset = (
                train_data.map(self.map_record_to_training_data)
                .map(lambda x, y: (self.lowercase_and_convert_to_ids(x), y))
                .padded_batch(batch_size)
            )
            val_dataset = (
                val_data.map(self.map_record_to_training_data)
                .map(lambda x, y: (self.lowercase_and_convert_to_ids(x), y))
                .padded_batch(batch_size)
            )            
            # train_data = tf.data.TextLineDataset("./data/train.txt")
            # val_data = tf.data.TextLineDataset("./data/val.txt")            

        elif self.source == "existing":
            print("Importing preloaded training and validation data")
            with open('./data/tokens.txt', 'r', encoding='utf-8') as f:
                all_tokens_string = f.read()
                all_tokens = all_tokens_string.split(' ')
            all_tokens_array = np.array(list(map(str.lower, all_tokens)))
            counter = Counter(all_tokens_array)  
            vocabulary = [token for token, count in counter.most_common(self.vocab_size - 2)]  
            self.lookup_layer = keras.layers.StringLookup(vocabulary=vocabulary)
            self.write_vocab_to_file(VOCAB_PATH,vocabulary)
            train_data = tf.data.TextLineDataset("./data/train.txt")
            val_data = tf.data.TextLineDataset("./data/val.txt")  
            batch_size = 128
            train_dataset = (
                train_data.map(self.map_record_to_training_data)
                .map(lambda x, y: (self.lowercase_and_convert_to_ids(x), y))
                .padded_batch(batch_size)
            )
            val_dataset = (
                val_data.map(self.map_record_to_training_data)
                .map(lambda x, y: (self.lowercase_and_convert_to_ids(x), y))
                .padded_batch(batch_size)
            )            
            
#         if self.source == "gutenberg":

#             print("Importing gutenberg data")
#             train_data = tf.data.TextLineDataset("./data/train3.txt")
#             val_data = tf.data.TextLineDataset("./data/val3.txt")
            
#             print("Importing preloaded training and validation data")
#             with open('./data/vocab3_deduped.txt', 'r', encoding='utf-8') as f:
#                 all_tokens_string = f.read()
#                 all_tokens = all_tokens_string.split(' ')
#             all_tokens_array = np.array(list(map(str.lower, all_tokens)))
            
            
#             counter = Counter(all_tokens_array)  
#             vocabulary = [token for token, count in counter.most_common(self.vocab_size - 2)]  
#             self.lookup_layer = keras.layers.StringLookup(vocabulary=vocabulary)
#             self.write_vocab_to_file(VOCAB_PATH,vocabulary)

#             batch_size = 32
#             train_dataset = (
#                 train_data.map(self.map_record_to_training_data)
#                 .map(lambda x, y: (self.lowercase_and_convert_to_ids(x), y))
#                 .padded_batch(batch_size)
#             )
#             val_dataset = (
#                 val_data.map(self.map_record_to_training_data)
#                 .map(lambda x, y: (self.lowercase_and_convert_to_ids(x), y))
#                 .padded_batch(batch_size)
#             )            


        return train_dataset, val_dataset, self.lookup_layer


class NERNonPaddingTokenLoss(keras.losses.Loss):
    def __init__(self, reduction=tf.keras.losses.Reduction.AUTO, name="NERNonPaddingTokenLoss", **kwargs):
        super().__init__(reduction=reduction, name=name, **kwargs)

    def call(self, y_true, y_pred):
        try:
            loss_fn = keras.losses.SparseCategoricalCrossentropy(
                from_logits=False, reduction=keras.losses.Reduction.NONE
            )
            loss = loss_fn(y_true, y_pred)
            mask = tf.cast((y_true > 0), dtype=tf.float32)
            loss = loss * mask
            return tf.reduce_sum(loss) / tf.reduce_sum(mask)
        except Exception as e:
            print("Error in Loss: ",e)        


class NERTransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.ffn = keras.Sequential(
            [
                keras.layers.Dense(ff_dim, activation="gelu"),
                keras.layers.Dense(embed_dim),
            ]
        )
        self.layernorm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = keras.layers.Dropout(rate)
        self.dropout2 = keras.layers.Dropout(rate)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
    
    
class NERTokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super().__init__()
        self.token_emb = keras.layers.Embedding(
            input_dim=vocab_size, output_dim=embed_dim
        )
        self.pos_emb = keras.layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, inputs):
        try:
            maxlen = tf.shape(inputs)[-1]
            positions = tf.range(start=0, limit=maxlen, delta=1)
            position_embeddings = self.pos_emb(positions)
            token_embeddings = self.token_emb(inputs)
            return token_embeddings + position_embeddings    
        except Exception as e:
            print("Error in NERTokenAndPositionEmbedding: ",e)
    
    
    
class NERNamePrediction(keras.Model):
    def __init__(
        self, num_tags, vocab_size, maxlen=256, embed_dim=1024, num_heads=64, ff_dim=2048, lstm_units=1024
    ):
        super().__init__()
        self.embedding_layer = NERTokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
        self.transformer_block = NERTransformerBlock(embed_dim, num_heads, ff_dim)
        self.ff1 = layers.Dense(ff_dim, activation="gelu")
        self.bilstm = layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=True))        
        self.dropout1 = layers.Dropout(0.15)
        self.ff = layers.Dense(ff_dim, activation="gelu")
        self.dropout2 = layers.Dropout(0.15)
        self.ff_final = layers.Dense(num_tags, activation="softmax")

    def call(self, inputs, training=False):
        x = self.embedding_layer(inputs)
        x = self.transformer_block(x)
        x = self.ff1(x)
        x = self.dropout1(x, training=training)              
        x = self.bilstm(x)        
        x = self.ff(x)
        x = self.dropout2(x, training=training)
        x = self.ff_final(x)
        return x    
        
    
class NERSaveModel(tf.keras.callbacks.Callback):
    def __init__(self, save_freq, **kwargs):
        super(NERSaveModel, self).__init__()
        self.save_freq = save_freq
    
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.save_freq == 0:
            save_path = f"ner_saved_{epoch + 1}_" + FDATE
            self.model.save(save_path)
            print(f"Model saved at epoch: {epoch + 1}")    

    
class TrainNERNamePrediction:
    def __init__(self, num_tags=NUM_TAGS, vocab_size=20000, embed_dim=32, num_heads=4, 
                 ff_dim=64, **kwargs):
        self.num_tags = num_tags
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.ner_model = None
        self.history = None
        self.strategy = tf.distribute.get_strategy()
        self.callbacks = []
        
        
        for key, value in kwargs.items():
            setattr(self, key, value)                
        
    def add_early_stopping(self, monitor='val_accuracy', mode='min', patience = 5, verbose = 2):
        es = EarlyStopping(monitor=monitor, mode=mode, patience=patience, verbose=verbose)
        self.callbacks.append(es)     

    def save_model(self, save_path):
        self.ner_model.save(save_path)
        
    def setup_callbacks(self, save_freq=50):
        self.callbacks.append(NERSaveModel(save_freq=save_freq))  
        self.add_early_stopping()
        
    def build_model(self):
        with self.strategy.scope():
            self.ner_model = NERNamePrediction(self.num_tags, self.vocab_size, embed_dim=self.embed_dim,
                                      num_heads=self.num_heads, ff_dim=self.ff_dim)
        
    def compile_model(self, optimizer=Adam(learning_rate=0.001), loss=NERNonPaddingTokenLoss(), metrics=['accuracy']):
        if self.ner_model is None:
            raise ValueError("Model has not been built yet.")
        self.ner_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    
    def train(self, train_data, val_data, epochs=1, batch_size=1, save_freq=250):
        if self.ner_model is None:
            raise ValueError("Model has not been compiled yet.")
        self.setup_callbacks(save_freq)
        self.history = self.ner_model.fit(
            train_data, 
            validation_data=val_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=self.callbacks
        )
        return self.history    


###################################################################################################
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
# End Deep Learning NER Model
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
###################################################################################################    


















###################################################################################################
# /////////////////////////////////////////////////////////////////////////////////////////////////
# Begin CTC Model
# /////////////////////////////////////////////////////////////////////////////////////////////////
###################################################################################################
    
max_characters = 50
    
class CTCSetup(StatisticalNameData, QuotationIndicator):
    def __init__(self, **kwargs):
        super().__init__()
        self.num_characters = None
        self.name_to_token = {}

    def remove_posessive(self, name):
        name_cleaned = name    
        if name[-2:] == "'s":
            name_cleaned = name[:-2]
        return name_cleaned

    def create_character_list(self, text):
        str_persons_mentioned = []
        text_list, ner_prediction = StatisticalNameData.ner_process_line(self=self, text=text)
        list_words_w_ner_indicators = list(zip(text_list, ner_prediction))
        grouped_words_w_ner_indicators = QuotationIndicator.group_consecutive_words(self=self, text_w_word_indicators = list_words_w_ner_indicators)
        list_persons_mentioned = [item for item in grouped_words_w_ner_indicators if item[1] == '1']
        str_persons_mentioned = []
        for person in list_persons_mentioned:
            person_name = ' '.join(person[0])
            person_name = self.remove_posessive(person_name)
            person_name = AI574Project.remove_special_characters(person_name).strip()
            str_persons_mentioned.append(person_name)

        element_counts = Counter(str_persons_mentioned)
        sorted_str_persons_mentioned = sorted(str_persons_mentioned, key=lambda x: element_counts[x], reverse=True)
        unique_elements = set()
        uniq_str_persons_mentioned = [x for x in sorted_str_persons_mentioned if x not in unique_elements and (unique_elements.add(x) or True)]

        token_dict = {'Narrator': 0}
        tokenized_list = []
        unique_elements = set()

        for x in uniq_str_persons_mentioned:
            if x not in unique_elements:
                unique_elements.add(x)
                token = len(token_dict)
                token_dict[x] = token
                tokenized_list.append(token)
            
        self.num_characters = len(unique_elements)+1

        return token_dict, uniq_str_persons_mentioned

    def list_mentions(self, text):
        
        str_persons_mentioned = []
        text_list, ner_prediction = StatisticalNameData.ner_process_line(self=self, text=text)
        list_words_w_ner_indicators = list(zip(text_list, ner_prediction))
        grouped_words_w_ner_indicators = QuotationIndicator.group_consecutive_words(self=self, text_w_word_indicators = list_words_w_ner_indicators)
        list_persons_mentioned = [item for item in grouped_words_w_ner_indicators if item[1] == '1']
        str_persons_mentioned = []
        token_mentioned = []
        
        one_hot_mentions = np.zeros(AI574Project.max_characters)
        
        for person in list_persons_mentioned:
            person_name = ' '.join(person[0])
            person_name = self.remove_posessive(person_name)
            person_name = AI574Project.remove_special_characters(person_name).strip()
            person_token = self.name_to_token[person_name]
            str_persons_mentioned.append(person_name)            
            if person_token >= AI574Project.max_characters: # if person mentioned is greater than max then mark speaker as narrator
                person_token = 0
            one_hot_mentions[person_token] = 1
            token_mentioned.append(person_token)
                
                
        num_mentions = len(token_mentioned)
        padded_token_mentioned = []
        if num_mentions > 10:
            padded_token_mentioned = token_mentioned[-10:]
            attention_mask = [1] * 10
        else:
            padding_length = 10-num_mentions
            padded_token_mentioned = token_mentioned + [0] * padding_length
            attention_mask = [1] * num_mentions + [0] * padding_length

        token_mention_dict = {"input_ids": padded_token_mentioned,"attention_mask": attention_mask}

        return str_persons_mentioned, token_mention_dict, one_hot_mentions

    def convert_text_to_token(self, trimmed_segment):
        text_length = len(trimmed_segment)
        if text_length > 512:
            keep_right = trimmed_segment[:256]
            keep_left = trimmed_segment[text_length - 256:]
            trimmed_for_tokenizer = keep_right + keep_left
        else:
            trimmed_for_tokenizer = trimmed_segment
        padded_tokenized_segment = bert_tokenizer.encode_plus(trimmed_for_tokenizer, max_length=512, padding='max_length', truncation=True)
        return padded_tokenized_segment

    def process_text(self, data):
        items = []
        tokenized_text = []
        quotation_ind = []
        mentions = []
        speaker = []
        token_mentions = []
        one_hot_mentions = []
        most_recent_mentions = [1]
        self.name_to_token, character_list = self.create_character_list(data)

        split_text = re.split(r'(-->.*?<--)', data)
    
        for segment in split_text:
            trimmed_segment = segment.strip()

            if trimmed_segment:
                if trimmed_segment.startswith('-->') and trimmed_segment.endswith('<--'):
                    quotation_ind.append('1')
                    items.append(trimmed_segment[3:-3].strip())
                    names, tokens, one_hot_mentions_line = self.list_mentions(trimmed_segment)
                    if any(tokens['input_ids']) > 0:
                        most_recent_mentions = tokens['input_ids']
                    one_hot_mentions.append(one_hot_mentions_line)
                    # items.append(trimmed_segment)
                    padded_tokenized_segment = self.convert_text_to_token(trimmed_segment)
                    tokenized_text.append(padded_tokenized_segment)
                    mentions.append(names)
                    token_mentions.append(tokens)

                    # THIS IS A PLACEHOLDER TO CHOOSE A CHARACTER THATS BEEN MENTIONED RECENTLY
                    # IN THE TEXT BUT IS NOT NECESSARILY SPEAKING
                    # NEEDS TO BE REPLACED WITH CTC MODEL PREDICTION LOGIC
                    token_speaker = random.choice(most_recent_mentions) 
                    
                    speaker.append(token_speaker)
                else:
                    quotation_ind.append('0')
                    names, tokens, one_hot_mentions_line = self.list_mentions(trimmed_segment)
                    if any(tokens['input_ids']) > 0:
                        most_recent_mentions = tokens['input_ids']                
                    one_hot_mentions.append(one_hot_mentions_line)
                    items.append(trimmed_segment)
                    padded_tokenized_segment = self.convert_text_to_token(trimmed_segment)
                    tokenized_text.append(padded_tokenized_segment)
                    mentions.append(names)
                    token_mentions.append(tokens)
                    speaker.append(0)

        return items, tokenized_text, quotation_ind, mentions, token_mentions, one_hot_mentions, speaker, self.name_to_token    
    

class CTCSpeakerPrediction(tf.keras.Model):
    def __init__(self, vocab_size, output_labels, input_shape, 
                 strides=6, ff_dim=512, rnn_units=1024, lstm_units=512, dropout_rate=0.4):
        super(CTCSpeakerPrediction, self).__init__()
        self.strides = strides
        self.ff_dim = ff_dim
        self.rnn_units = rnn_units
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.conv1d = layers.Conv1D(
                filters=32,
                kernel_size=11,
                strides=self.strides,
                padding="same",
                use_bias=False,
                name="conv_1",
            )
        self.batnorm = layers.BatchNormalization(name="conv_1_bn")        
        self.relu = layers.ReLU(name="conv_2_relu")
        self.flatten = layers.Flatten()
        self.ff = layers.Dense(self.ff_dim, activation="relu")
        self.gru = layers.GRU(
            units=rnn_units,
            activation="tanh",
            recurrent_activation="sigmoid",
            use_bias=True,
            return_sequences=True,
            reset_after=True,
            name="gru",
        )
        self.bilstm = layers.Bidirectional(layers.LSTM(self.lstm_units, return_sequences=True))
        self.dropout = layers.Dropout(rate=self.dropout_rate)
        self.output_layer = layers.Dense(output_labels, activation='softmax')

    def call(self, inputs):
        x = inputs
        x = tf.expand_dims(x, axis=-1)
        x = self.conv1d(x)
        x = self.batnorm(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = layers.Reshape((-1, x.shape[-2] * x.shape[-1]))(x)
        x = self.gru(x)
        x = self.bilstm(x)
        x = self.ff(x)
        x = self.dropout(x)
        x = self.output_layer(x)
        return x
    
    
class SaveCTCSpeakerPrediction(tf.keras.callbacks.Callback):
    def __init__(self, save_freq, **kwargs):
        super(SaveCTCSpeakerPrediction, self).__init__()
        self.save_freq = save_freq
    
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.save_freq == 0:
            save_path = f"ctc_saved_{epoch + 1}_" + FDATE
            self.model.save(save_path)
            print(f"Model saved at epoch: {epoch + 1}")


class TrainCTCSpeakerPrediction:
    def __init__(self, input_shape, output_labels=AI574Project.max_characters, vocab_size=AI574Project.bert_vocab_size, **kwargs):
        self.output_labels = output_labels
        self.vocab_size = vocab_size
        self.ctc_model = None
        self.history = None
        self.strategy = tf.distribute.get_strategy()
        self.callbacks = []
        self.input_shape = input_shape
        
        for key, value in kwargs.items():
            setattr(self, key, value)                
        
    def add_early_stopping(self, monitor='val_accuracy', mode='min', patience = 5, verbose = 2):
        es = EarlyStopping(monitor=monitor, mode=mode, patience=patience, verbose=verbose)
        self.callbacks.append(es)     

    def setup_callbacks(self, save_freq=50):
        self.callbacks.append(SaveCTCSpeakerPrediction(save_freq=save_freq))  
        self.add_early_stopping()        
        
    def save_model(self, save_path):
        self.ctc_model.save(save_path)
        
    def build_model(self):
        with self.strategy.scope():
            self.ctc_model = CTCSpeakerPrediction(output_labels = self.output_labels, 
                                                  vocab_size = self.vocab_size, 
                                                  input_shape = self.input_shape,
                                                  )
        
    def compile_model(self, 
                      optimizer=Adam(learning_rate=0.0001), 
                      loss='sparse_categorical_crossentropy', 
                      metrics=['accuracy']):
        if self.ctc_model is None:
            raise ValueError("Model has not been built yet.")
        with self.strategy.scope():
            self.ctc_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    
    def train(self, train_data, val_data, epochs=1, batch_size=1, save_freq=250):
        if self.ctc_model is None:
            raise ValueError("Model has not been compiled yet.")
        self.setup_callbacks(save_freq)
        self.history = self.ctc_model.fit(
            train_data, 
            validation_data=val_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=self.callbacks
        )
        return self.history        
    
    
hobbit_characters = [
'Bilbo',
'Bilbo Baggins',
'Balin',
'Bard',
'Beorn',
'Bifur',
'Bofur',
'Bolg',
'Bombur',
'Carc',
'Chief of the guards',
'Councilors',
'Dori',
'Dwalin',
'Dain II',
'Elrond',
'Fili',
'Kili',
'Galion',
'Gandalf',
'Gloin',
'Golfimbul',
'Gollum',
'Great Goblin',
'Great grey chief wolf',
'Master of Lake-town',
'Nori',
'Ori',
'Radagast',
'Roac',
'Smaug',
'Thorin',
'Thorin II',
'Thranduil',
'Thrain II',
'Thror',
'Tom',
'Bert',
'William',
'Bandobras',
'Bandobras Took',
'Oin',
'Dwarves'
]    
    
hobbit_chap1_scored = (
'Narrator',
'Narrator',
'Narrator',
'Narrator',
'Bilbo',
'Narrator',
'Gandalf',
'Narrator',
'Gandalf',
'Bilbo',
'Narrator',
'Bilbo',
'Narrator',
'Gandalf',
'Narrator',
'Gandalf',
'Narrator',
'Bilbo',
'Narrator',
'Bilbo',
'Narrator',
'Gandalf',
'Narrator',
'Gandalf',
'Bilbo',
'Gandalf',
'Bilbo',
'Narrator',
'Bilbo',
'Narrator',
'Bilbo',
'Gandalf',
'Narrator',
'Gandalf',
'Bilbo',
'Gandalf',
'Bilbo',
'Narrator',
'Bilbo',
'Narrator',
'Bilbo',
'Narrator',
'Dwalin',
'Narrator',
'Bilbo',
'Narrator',
'Bilbo',
'Narrator',
'Bilbo',
'Narrator',
'Bilbo',
'Narrator',
'Balin',
'Narrator',
'Balin',
'Narrator',
'Bilbo',
'Narrator',
'Bilbo',
'Narrator',
'Balin',
'Narrator',
'Balin',
'Bilbo',
'Narrator',
'Bilbo',
'Narrator',
'Bilbo',
'Narrator',
'Kili',
'Narrator',
'Fili',
'Narrator',
'Bilbo',
'Narrator',
'Kili',
'Narrator',
'Kili',
'Bilbo',
'Narrator',
'Bilbo',
'Narrator',
'Bilbo',
'Narrator',
'Fili',
'Narrator',
'Fili',
'Narrator',
'Narrator',
'Narrator',
'Gandalf',
'Narrator',
'Gandalf',
'Bifur',
'Narrator',
'Bilbo',
'Narrator',
'Gandalf',
'Narrator',
'Gandalf',
'Thorin',
'Narrator',
'Bifur',
'Narrator',
'Bofu',
'Narrator',
'Bombur',
'Narrator',
'Narrator',
'Narrator',
'Gandalf',
'Narrator',
'Gandalf',
'Bilbo',
'Narrator',
'Bilbo',
'Narrator',
'Bilbo',
'Narrator',
'Bilbo',
'Narrator',
'Thorin',
'Narrator',
'Thorin',
'Narrator',
'Bilbo',
'Bilbo',
'Bilbo',
'Narrator',
'Dwarves',
'Narrator',
'Thorin',
'Narrator',
'Thorin',
'Narrator',
'Dwalin',
'Thorin',
'Narrator',
'Dwarves',
'Narrator',
'Thorin',
'Narrator',
'Bilbo',
'Narrator',
'Dwarves',
'Narrator',
'Dwarves',
'Bilbo',
'Narrator',
'Gandalf',
'Narrator',
'Gandalf',
'Narrator',
'Thorin',
'Narrator',
'Thorin',
'Narrator',
'Bilbo',
'Narrator',
'Gandalf',
'Narrator',
'Gandalf',
'Narrator',
'Gloin',
'Narrator',
'Gloin',
'Narrator',
'Bilbo',
'Bilbo',
'Narrator',
'Bilbo',
'Narrator',
'Bilbo',
'Gloin',
'Narrator',
'Gloin',
'Gandalf',
'Narrator',
'Gandalf',
'Narrator',
'Gandalf',
'Narrator',
'Gandalf',
'Narrator',
'Gandalf',
'Thorin',
'Narrator',
'Thorin',
'Balin',
'Gandalf',
'Narrator',
'Thorin',
'Narrator',
'Thorin',
'Narrator',
'Thorin',
'Bilbo',
'Narrator',
'Bilbo',
'Narrator',
'Gandalf',
'Narrator',
'Bilbo',
'Thorin',
'Narrator',
'Gandalf',
'Narrator',
'Gandalf',
'Narrator',
'Gandalf',
'Thorin',
'Narrator',
'Thorin',
'Gandalf',
'Narrator',
'Thorin',
'Narrator',
'Thorin',
'Gandalf',
'Narrator',
'Gandalf',
'Thorin',
'Narrator',
'Thorin',
'Narrator',
'Bilbo',
'Narrator',
'Bilbo',
'Thorin',
'Narrator',
'Thorin',
'Bilbo',
'Narrator',
'Bilbo',
'Narrator',
'Bilbo',
'Thorin',
'Narrator',
'Thorin',
'Narrator',
'Thorin',
'Gandalf',
'Narrator',
'Gandalf',
'Thorin',
'Narrator',
'Gandalf',
'Thorin',
'Narrator',
'Gandalf',
'Narrator',
'Thorin',
'Narrator',
'Gandalf',
'Narrator',
'Gandalf',
'Thorin',
'Narrator',
'Gandalf',
'Thorin',
'Narrator',
'Thorin',
'Gandalf',
'Bilbo',
'Narrator',
'Dwarves',
'Narrator',
'Bilbo',
'Dwarves',
'Narrator',
'Bilbo',
'Thorin',
'Narrator',
'Thorin',
'Narrator',
'Narrator'    
)

###################################################################################################
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
# End CTC Model
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
###################################################################################################    
