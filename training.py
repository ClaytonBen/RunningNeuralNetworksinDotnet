from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, concatenate, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
# Assuming a maximum length for input strings
max_length = 30  # You might need to adjust this based on your data

# Character-level tokenizer setup
# Assuming we're using ASCII characters + some special tokens for padding/start/end
num_characters = 128  # ASCII

# Model Architecture
def build_model(input_length, num_characters):
    # Input layers for both English and Pig Latin strings
    input_english = Input(shape=(input_length,), dtype='int32', name='input_english')
    input_piglatin = Input(shape=(input_length,), dtype='int32', name='input_piglatin')

    # Shared embedding layer
    embedding = Embedding(input_dim=num_characters, output_dim=50, input_length=input_length, name='shared_embedding')

    # LSTM layers for both inputs
    lstm_english = LSTM(64, return_sequences=False, name='lstm_english')(embedding(input_english))
    lstm_piglatin = LSTM(64, return_sequences=False, name='lstm_piglatin')(embedding(input_piglatin))

    # Concatenate both LSTM outputs
    merged = concatenate([lstm_english, lstm_piglatin], axis=-1)
    merged = Dropout(0.5)(merged)
    output = Dense(1, activation='sigmoid', name='output')(merged)

    # Model setup
    model = Model(inputs=[input_english, input_piglatin], outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model

model = build_model(max_length, num_characters)

# Model summary
model.summary()

# Load the data
df = pd.read_csv('pig_latin_test_data.csv')

# Preprocess the data
# Convert English and Pig Latin words to sequences of integers
tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(df['English'].values.tolist() + df['PigLatin'].values.tolist())

# Convert texts to sequences
english_sequences = tokenizer.texts_to_sequences(df['English'].values)
piglatin_sequences = tokenizer.texts_to_sequences(df['PigLatin'].values)

# Pad sequences to ensure uniform length
max_length = max(max(len(seq) for seq in english_sequences), max(len(seq) for seq in piglatin_sequences))
english_sequences_padded = pad_sequences(english_sequences, maxlen=max_length, padding='post')
piglatin_sequences_padded = pad_sequences(piglatin_sequences, maxlen=max_length, padding='post')

# Labels
labels = df['Label'].values

# Train-test split
X1_train, X1_test, X2_train, X2_test, y_train, y_test = train_test_split(
    english_sequences_padded, piglatin_sequences_padded, labels, test_size=0.2, random_state=42)