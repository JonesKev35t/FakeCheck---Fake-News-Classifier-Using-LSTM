import nltk
import pandas as pd
import numpy as np
import tensorflow as tf
from nltk.tokenize import sent_tokenize
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


data = pd.read_csv('/Users/kevinjones/Desktop/Mini Project/news.csv')

label_encoder = LabelEncoder()
data['label'] = label_encoder.fit_transform(data['label'])
data['text'] = data['text'].astype(str).fillna('')
data['label'] = pd.to_numeric(data['label'], errors='coerce')

data.dropna(subset=['label'], inplace=True)
data.sample(frac=1).reset_index(drop=True)

print(data.head(4))

texts = data['text'].values
labels = data['label'].values

nltk.download('punkt_tab')

text_list = [str(text) for text in texts]

sent = []
for text in text_list:
    sent.extend(sent_tokenize(text))

'''sent=sent_tokenize(texts)
print(sent)'''

newsent = tf.strings.regex_replace(tf.constant(sent), r'[^a-zA-Z\s]', '')
newsent_list = [str(item.numpy().decode('utf-8'))
                for item in newsent]
token = Tokenizer(num_words=5000)
token.fit_on_texts(text_list)
sequences = token.texts_to_sequences(text_list)
X = pad_sequences(sequences, maxlen=500)
y = np.array(labels, dtype=np.float32)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42, shuffle=True)



model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=100, input_length=500))
model.add(LSTM(100, dropout=0.6, recurrent_dropout=0.6))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer=Adam(), loss='binary_crossentropy',metrics=['accuracy'])



early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)

model.fit(X_train, y_train, epochs=20, batch_size=64,validation_data=(X_test, y_test),callbacks=[early_stopping, reduce_lr])
model.summary()
def pred_art(article, model, tokenizer, max_length=500):
    # Preprocess the article similarly to training data
    article = tf.strings.regex_replace(article, r'[^a-zA-Z\s]', '')  # Clean text
    article = article.numpy().decode('utf-8').lower()  # Decode and lowercase

    article_sequence = tokenizer.texts_to_sequences([article])
    article_padded = pad_sequences(article_sequence, maxlen=max_length)

    prediction = model.predict(article_padded)

    # Adjust threshold if necessary
    prediction_label = 'Real' if prediction[0][0] >= 0.5 else 'Fake'
    return prediction, prediction_label

new_article = """Bart Simpson elected mayor of LA town
The Simpsons star Nancy Cartwright has been elected mayor of Northridge in Los Angeles.

Cartwright, who voices tearaway child Bart Simpson in the cartoon series, is determined to make her mark following her election victory on Tuesday and plans to clamp down on rampaging youths immediately.

The 47-year-old mother-of-two says: "Everyone finds it funny that Bart is the new mayor. I can influence people because I'm Bart Simpson.

"I live in a nice neighbourhood. But down the road there's drugs and gangs, stealing and illiteracy."
"""

pred_val, result = pred_art(new_article, model, token)
print("prediction values: ", pred_val)
print("result: ", result)


