def load_data(file_path):
    import pandas as pd
    data = pd.read_csv(file_path)
    return data

def clean_text(text):
    import re
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    return text

def preprocess_data(data):
    data['cleaned_text'] = data['text'].apply(clean_text)
    return data

def tokenize_text(data):
    from sklearn.feature_extraction.text import CountVectorizer
    vectorizer = CountVectorizer()
    features = vectorizer.fit_transform(data['cleaned_text'])
    return features, vectorizer

def split_data(data, test_size=0.2):
    from sklearn.model_selection import train_test_split
    train_data, test_data = train_test_split(data, test_size=test_size, random_state=42)
    return train_data, test_data