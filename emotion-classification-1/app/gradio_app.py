from gradio import Interface
from src.model_sklearn import EmotionClassifier as SklearnClassifier
from src.model_tensorflow import EmotionClassifier as TensorflowClassifier
from src.model_pytorch import EmotionClassifier as PytorchClassifier
from src.model_huggingface import EmotionClassifier as HuggingFaceClassifier

def classify_emotion(text, model_type):
    if model_type == "Scikit-Learn":
        model = SklearnClassifier()
    elif model_type == "TensorFlow":
        model = TensorflowClassifier()
    elif model_type == "PyTorch":
        model = PytorchClassifier()
    elif model_type == "Hugging Face":
        model = HuggingFaceClassifier()
    else:
        return "Invalid model type selected."

    prediction = model.predict(text)
    return prediction

iface = Interface(
    fn=classify_emotion,
    inputs=["text", "dropdown"],
    outputs="text",
    title="Emotion Classification",
    description="Classify emotions from text using different models.",
    examples=[
        ["I am so happy today!", "Scikit-Learn"],
        ["I feel sad and lonely.", "TensorFlow"],
        ["This is amazing!", "PyTorch"],
        ["I am excited about the future.", "Hugging Face"]
    ]
)

if __name__ == "__main__":
    iface.launch()