import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam

def load_data():
    # Sample data for binary sentiment classification
    texts = [
        "I love this product!",
        "This is terrible.",
        "Amazing experience!",
        "Disappointed with the service.",
        # Add more positive and negative samples...
    ]

    labels = ["positive", "negative", "positive", "negative"]

    return texts, labels

def preprocess(texts, labels):
    # Convert text data into numerical features
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(texts)
    
    # Encode labels to numerical values
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)

    return X.toarray(), y, vectorizer

def build_model(input_dim):
    model = Sequential()
    model.add(Dense(64, input_dim=input_dim))
    model.add(Activation('relu'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    return model

def train_model(X_train, y_train):
    model = build_model(X_train.shape[1])
    optimizer = Adam(learning_rate=0.001)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=10, batch_size=32)
    return model

def predict_sentiment(model, texts, vectorizer):
    X = vectorizer.transform(texts)
    y_pred_probs = model.predict(X)
    y_pred_classes = np.argmax(y_pred_probs, axis=1)
    return y_pred_classes

def main():
    texts, labels = load_data()
    X, y, vectorizer = preprocess(texts, labels)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = train_model(X_train, y_train)

    test_texts = [
        "I am happy with this product!",
        "The service was terrible.",
        "Great experience overall!",
        "The worst experience ever."
    ]

    y_pred = predict_sentiment(model, test_texts, vectorizer)
    sentiment_labels = ["negative", "positive"]
    predicted_sentiments = [sentiment_labels[p] for p in y_pred]

    for text, sentiment in zip(test_texts, predicted_sentiments):
        print(f"Text: '{text}'\tPredicted Sentiment: '{sentiment}'")

if __name__ == "__main__":
    main()
