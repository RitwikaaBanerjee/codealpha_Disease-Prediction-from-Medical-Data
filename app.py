import streamlit as st
import pickle
import pandas as pd
import numpy as np


def load_model():
    """
    Load the trained disease prediction model

    Returns:
        dict: Model, label encoder, and feature names
    """
    try:
        with open("models/disease_prediction_model.pkl", "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error("Model not found. Please train the model first.")
        return None


def predict_disease(model, label_encoder, feature_names, symptoms):
    """
    Predict disease based on input symptoms

    Args:
        model (RandomForestClassifier): Trained model
        label_encoder (LabelEncoder): Encoder for disease labels
        feature_names (list): List of all possible features
        symptoms (list): Selected symptoms

    Returns:
        str: Predicted disease
    """
    # Create input vector
    input_vector = [1 if symptom in symptoms else 0 for symptom in feature_names]
    input_array = np.array(input_vector).reshape(1, -1)

    # Predict
    prediction_index = model.predict(input_array)[0]
    prediction = label_encoder.inverse_transform([prediction_index])[0]

    return prediction


def main():
    st.title("Disease Prediction Model 🩺")

    # Load the model
    model_data = load_model()
    if not model_data:
        return

    model = model_data["model"]
    label_encoder = model_data["label_encoder"]
    feature_names = model_data["feature_names"]

    # Symptom selection
    st.header("Select Your Symptoms")

    # Dynamically create checkboxes for symptoms
    selected_symptoms = st.multiselect(
        "Choose your symptoms",
        feature_names,
        help="Select all symptoms you are experiencing",
    )

    # Prediction button
    if st.button("Predict Disease"):
        if not selected_symptoms:
            st.warning("Please select at least one symptom.")
        else:
            # Make prediction
            prediction = predict_disease(
                model, label_encoder, feature_names, selected_symptoms
            )

            # Display results
            st.success(f"Predicted Disease: {prediction}")

            # Additional advice (placeholder)
            st.info(
                "Note: This is a preliminary prediction. Always consult a healthcare professional."
            )


if __name__ == "__main__":
    main()
