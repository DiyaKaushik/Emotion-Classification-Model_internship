import streamlit as st
from src.data_preprocessing import load_data, preprocess_data
from src.model_sklearn import SklearnEmotionClassifier
from src.model_tensorflow import TensorFlowEmotionClassifier
from src.model_pytorch import PyTorchEmotionClassifier
from src.model_huggingface import HuggingFaceEmotionClassifier

st.title("Emotion Classification App")

# Load and preprocess data
data = load_data()
preprocessed_data = preprocess_data(data)

# Select model type
model_type = st.selectbox("Select Model Type", 
                           ["Scikit-Learn", "TensorFlow", "PyTorch", "Hugging Face"])

if model_type == "Scikit-Learn":
    model = SklearnEmotionClassifier()
elif model_type == "TensorFlow":
    model = TensorFlowEmotionClassifier()
elif model_type == "PyTorch":
    model = PyTorchEmotionClassifier()
else:
    model = HuggingFaceEmotionClassifier()

# Train the model
if st.button("Train Model"):
    model.train(preprocessed_data)
    st.success("Model trained successfully!")

# Input for prediction
user_input = st.text_area("Enter text for emotion classification:")

if st.button("Predict Emotion"):
    prediction = model.predict(user_input)
    st.write(f"Predicted Emotion: {prediction}")