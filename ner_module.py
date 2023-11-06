import os
current_directory = os.getcwd()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'   
if current_directory == '/media/daniel/HDD1/AI574/Project':
    import tensorflow as tf
    tf.load_library("/etc/alternatives/libcudnn_so")
else:
    import tensorflow as tf
from tensorflow import keras    
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from datasets import load_dataset
from collections import Counter
from datetime import datetime
today = datetime.now()
from conlleval import evaluate
from ner_module import *


print("The NER model module importing from directory ",current_directory)


###################################################################################################
###################################################################################################
# GLOBAL Variables
###################################################################################################
###################################################################################################



VOCAB_PATH = os.path.join(current_directory,'vocabulary.txt')
FDATE = today.strftime("%Y%m%d")



###################################################################################################
###################################################################################################
# Define conll data characteristics
###################################################################################################
###################################################################################################



def conll_tag_lookup_table():
    iob_labels = ["B", "I"]
    ner_labels = ["PER", "ORG", "LOC", "MISC"]
    all_labels = [(label1, label2) for label2 in ner_labels for label1 in iob_labels]
    all_labels = ["-".join([a, b]) for a, b in all_labels]
    all_labels = ["[PAD]", "O"] + all_labels
    return dict(zip(range(0, len(all_labels) + 1), all_labels))

MAPPING = conll_tag_lookup_table()
NUM_TAGS = len(MAPPING)



##################################################################################################
###################################################################################################
# Data Handling
###################################################################################################
###################################################################################################



class ImportData():
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

            batch_size = 32
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
            train_data = tf.data.TextLineDataset("./data/train.txt")
            val_data = tf.data.TextLineDataset("./data/val.txt")            

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
            batch_size = 32
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


        return train_dataset, val_dataset, self.lookup_layer



###################################################################################################
###################################################################################################
# Define Loss Function
###################################################################################################
###################################################################################################

class CustomNonPaddingTokenLoss(keras.losses.Loss):
    def __init__(self, reduction=tf.keras.losses.Reduction.AUTO, name="CustomNonPaddingTokenLoss", **kwargs):
        super().__init__(reduction=reduction, name=name, **kwargs)

    def call(self, y_true, y_pred):
        loss_fn = keras.losses.SparseCategoricalCrossentropy(
            from_logits=False, reduction=keras.losses.Reduction.NONE
        )
        loss = loss_fn(y_true, y_pred)
        mask = tf.cast((y_true > 0), dtype=tf.float32)
        loss = loss * mask
        return tf.reduce_sum(loss) / tf.reduce_sum(mask)

    

###################################################################################################
###################################################################################################
# Define Model 
###################################################################################################
###################################################################################################



class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.ffn = keras.Sequential(
            [
                keras.layers.Dense(ff_dim, activation="relu"),
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
    
    
    
class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super().__init__()
        self.token_emb = keras.layers.Embedding(
            input_dim=vocab_size, output_dim=embed_dim
        )
        self.pos_emb = keras.layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, inputs):
        maxlen = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        position_embeddings = self.pos_emb(positions)
        token_embeddings = self.token_emb(inputs)
        return token_embeddings + position_embeddings    
    
    
    
class NERModel(keras.Model):
    def __init__(
        self, num_tags, vocab_size, maxlen=128, embed_dim=32, num_heads=2, ff_dim=32
    ):
        super().__init__()
        self.embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
        self.transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
        self.dropout1 = layers.Dropout(0.1)
        self.ff = layers.Dense(ff_dim, activation="relu")
        self.dropout2 = layers.Dropout(0.1)
        self.ff_final = layers.Dense(num_tags, activation="softmax")

    def call(self, inputs, training=False):
        x = self.embedding_layer(inputs)
        x = self.transformer_block(x)
        x = self.dropout1(x, training=training)
        x = self.ff(x)
        x = self.dropout2(x, training=training)
        x = self.ff_final(x)
        return x    
    
    
    
###################################################################################################
###################################################################################################
# Train Model
###################################################################################################
###################################################################################################   
    
    
    
class SaveModel(tf.keras.callbacks.Callback):
    def __init__(self, save_freq, **kwargs):
        super(SaveModel, self).__init__()
        self.save_freq = save_freq
    
    def on_epoch_end(self, epoch, logs=None):
        if (epoch + 1) % self.save_freq == 0:
            save_path = f"ner_saved_{epoch + 1}_" + FDATE
            self.model.save(save_path)
            print(f"Model saved at epoch: {epoch + 1}")    
 

    
class TrainModel:
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
        self.callbacks.append(SaveModel(save_freq=save_freq))  
        self.add_early_stopping()
        
    def build_model(self):
        with self.strategy.scope():
            self.ner_model = NERModel(self.num_tags, self.vocab_size, embed_dim=self.embed_dim,
                                      num_heads=self.num_heads, ff_dim=self.ff_dim)
        
    def compile_model(self, optimizer='adam', loss=CustomNonPaddingTokenLoss(), metrics=['accuracy']):
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