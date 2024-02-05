import tf2onnx
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, concatenate, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
import pandas as pd
from sklearn.model_selection import train_test_split

# Load and preprocess data
df = pd.read_csv('pig_latin_test_data.csv')
tokenizer = Tokenizer(char_level=True)
tokenizer.fit_on_texts(df['English'] + df['PigLatin'])
vocab_size = len(tokenizer.word_index) + 1

english_sequences = tokenizer.texts_to_sequences(df['English'])
piglatin_sequences = tokenizer.texts_to_sequences(df['PigLatin'])

max_length = max(max(len(seq) for seq in english_sequences), max(len(seq) for seq in piglatin_sequences))
english_padded = pad_sequences(english_sequences, maxlen=max_length, padding='post')
piglatin_padded = pad_sequences(piglatin_sequences, maxlen=max_length, padding='post')

labels = df['Label'].values

# Train-test split
X1_train, X1_test, X2_train, X2_test, y_train, y_test = train_test_split(english_padded, piglatin_padded, labels, test_size=0.2, random_state=42)

# Model architecture
input_english = Input(shape=(max_length,))
input_piglatin = Input(shape=(max_length,))
embedding = Embedding(input_dim=vocab_size, output_dim=50, input_length=max_length)

english_embedded = embedding(input_english)
piglatin_embedded = embedding(input_piglatin)

lstm_layer = LSTM(64, return_sequences=False)
english_lstm = lstm_layer(english_embedded)
piglatin_lstm = lstm_layer(piglatin_embedded)

concatenated = concatenate([english_lstm, piglatin_lstm])
output = Dense(1, activation='sigmoid')(concatenated)

model = Model(inputs=[input_english, input_piglatin], outputs=output)

# Compile the model
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Model summary
model.summary()

# Train the model
model.fit([X1_train, X2_train], y_train, batch_size=32, epochs=30, validation_split=0.1)

# Evaluate the model
loss, accuracy = model.evaluate([X1_test, X2_test], y_test)
print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')

tf2onnx.convert.from_keras(model, output_path="testing.onnx")