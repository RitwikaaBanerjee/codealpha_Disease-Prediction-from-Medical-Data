# Disease Prediction Model 🩺

## Overview
This project implements a machine learning-based disease prediction model using Python, scikit-learn, and Streamlit. The model analyzes medical symptoms to predict potential diseases with high accuracy.

## 🚀 Project Structure
```
disease-prediction/
│
├── models/
│   ├── disease_prediction_model.pkl
│   ├── confusion_matrix.png
│   └── feature_importance.png
│
├── Training.csv        # Training dataset
├── Testing.csv         # Optional testing dataset
│
├── model_training.py   # Model training script
├── app.py              # Streamlit prediction interface
├── requirements.txt    # Project dependencies
└── README.md           # Project documentation
```

## 🔧 Prerequisites
- Python 3.8+
- Pip package manager

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/RitwikaaBanerjee/codealpha_Disease-Prediction-from-Medical-Data.git
cd disease-prediction
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 🧠 Model Training

To train the disease prediction model:
```bash
python model_training.py
```

This script will:
- Load training data
- Preprocess the dataset
- Train a Random Forest Classifier
- Evaluate model performance
- Save the trained model

## 🖥️ Running the Prediction App

Launch the Streamlit web application:
```bash
streamlit run app.py
```

## 📊 Model Features
- Uses Random Forest Classifier
- Supports multiple disease predictions
- Generates performance visualizations
- Interactive web interface

## 🔍 How to Use
1. Select symptoms from the web interface
2. Click "Predict Disease"
3. View the predicted disease
4. **Always consult a healthcare professional**

## 📈 Performance Metrics
- Accuracy visualization
- Confusion matrix
- Feature importance graph

## ⚠️ Disclaimer
This is a machine learning model for educational purposes. It should not replace professional medical advice.

## 🤝 Contributing
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📧 Contact
- GitHub: [@Prashant](https://github.com/prahants)
- GitHub: [@Ritwika](https://github.com/RitwikaaBanerjee)
## 📜 License
Distributed under the MIT License. See `LICENSE` for more information.
```

## 🙏 Acknowledgments
- Scikit-learn
- Streamlit
- Pandas
- Matplotlib
