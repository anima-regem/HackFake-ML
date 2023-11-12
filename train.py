from transformers import TFGPT2LMHeadModel, GPT2Tokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization

tokenizer = GPT2Tokenizer.from_pretrained("ashiqabdulkhader/GPT2-Malayalam")

embedding_model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')
# embedding_model = AutoModelForCausalLM.from_pretrained("ashiqabdulkhader/GPT2-Malayalam", from_tf=True)df = pd.read_csv("data.csv", delimiter=",")
df = df[df['Text'].apply(lambda x: isinstance(x, str))]
df = df.drop(['Website','Date'],axis=1)
df = df.dropna()
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)


max_length = 500

train_encodings = tokenizer(train_df['Text'].tolist(), truncation=True, padding=True, max_length=max_length, return_tensors='tf')
# train_embeddings = embedding_model(**train_encodings).last_hidden_state[:, 0, :]
train_embeddings = embedding_model.encode(train_df['Text'].tolist(), show_progress_bar=True)

test_encodings = tokenizer(test_df['Text'].tolist(), truncation=True, padding=True, max_length=max_length, return_tensors='tf')
test_embeddings = embedding_model.encode(test_df['Text'].tolist(), show_progress_bar=True)
# test_embeddings = embedding_model(**test_encodings).last_hidden_state[:, 0, :]

train_labels = train_df[['Hate','Misleading', 'Disinformation', 'Rumour', 'Sensationalism']].to_numpy()
test_labels = test_df[['Hate','Misleading', 'Disinformation', 'Rumour', 'Sensationalism']].to_numpy()


model = Sequential([
    BatchNormalization(),
    Dense(512, activation='relu', input_dim=train_embeddings.shape[1]),
    BatchNormalization(),
    Dense(256, activation="relu"),
    BatchNormalization(),
    Dense(5, activation="sigmoid"),
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(train_embeddings, train_labels, epochs=30, batch_size=8, validation_split=0.1)
model.save("model.keras")

test_loss, test_accuracy = model.evaluate(test_embeddings, test_labels)
print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')