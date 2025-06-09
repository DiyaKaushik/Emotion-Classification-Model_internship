from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class EmotionClassifier:
    def __init__(self, model_name='distilbert-base-uncased-finetuned-sst-2-english'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

    def preprocess(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True)
        return inputs

    def predict(self, text):
        inputs = self.preprocess(text)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        predicted_class = logits.argmax(dim=-1).item()
        return predicted_class

    def classify(self, text):
        class_id = self.predict(text)
        return class_id  # You can map this to actual emotion labels as needed