import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle 

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


class DiseasePredictionModel:
    def __init__(
        self,
        training_data_path="/Users/prashant/Desktop/Disease Prediction From Medical Data/Testing.csv",
        testing_data_path="/Users/prashant/Desktop/Disease Prediction From Medical Data/Training.csv",
        output_dir="models",
    ):
        """
        Initialize the disease prediction model with configurable data paths

        Args:
            training_data_path (str): Path to the training dataset
            testing_data_path (str): Path to the testing dataset
            output_dir (str): Directory to save model outputs
        """
        # Validate input paths
        if not os.path.exists(training_data_path):
            raise FileNotFoundError(
                f"Training data file not found: {training_data_path}"
            )

        # Optional testing data validation
        if testing_data_path and not os.path.exists(testing_data_path):
            print(f"Warning: Testing data file not found: {testing_data_path}")

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Store file paths
        self.training_data_path = training_data_path
        self.testing_data_path = testing_data_path
        self.output_dir = output_dir

        # Model attributes
        self.train_data = None
        self.test_data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.label_encoder = LabelEncoder()

    def load_data(self):
        """
        Load training and optional testing datasets
        """
        # Load training data
        self.train_data = pd.read_csv(self.training_data_path)

        # Load testing data if path is provided
        if self.testing_data_path and os.path.exists(self.testing_data_path):
            self.test_data = pd.read_csv(self.testing_data_path)
            print("Using provided testing dataset.")
        else:
            print("No separate testing dataset. Will split training data.")

    def preprocess_data(self):
        """
        Preprocess the medical dataset
        """
        # Remove any unnamed columns from training data
        self.train_data = self.train_data.loc[
            :, ~self.train_data.columns.str.contains("^Unnamed")
        ]

        # Remove duplicates
        self.train_data.drop_duplicates(inplace=True)

        # Separate features and target from training data
        X = self.train_data.drop(columns=["prognosis"])
        y = self.train_data["prognosis"]

        # Encode target variable
        y_encoded = self.label_encoder.fit_transform(y)

        # If separate testing data is provided
        if self.test_data is not None:
            # Ensure test data has same preprocessing
            X_test = self.test_data.drop(columns=["prognosis"])
            y_test = self.test_data["prognosis"]

            # Encode test target
            y_test_encoded = self.label_encoder.transform(y_test)

            self.X_train, self.X_test = X, X_test
            self.y_train, self.y_test = y_encoded, y_test_encoded
        else:
            # Split training data
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
            )

    def train_model(self):
        """
        Train Random Forest Classifier
        """
        self.model = RandomForestClassifier(
            n_estimators=100, random_state=42, class_weight="balanced"
        )

        # Cross-validation
        cv_scores = cross_val_score(self.model, self.X_train, self.y_train, cv=5)

        # Train on full data
        self.model.fit(self.X_train, self.y_train)

        return {"cross_val_scores": cv_scores, "mean_cv_score": cv_scores.mean()}

    def evaluate_model(self):
        """
        Evaluate model performance
        """
        # Predictions
        y_pred = self.model.predict(self.X_test)

        # Performance metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        conf_matrix = confusion_matrix(self.y_test, y_pred)
        class_report = classification_report(
            self.y_test, y_pred, target_names=self.label_encoder.classes_
        )

        # Visualize confusion matrix
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            conf_matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.label_encoder.classes_,
            yticklabels=self.label_encoder.classes_,
        )
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.tight_layout()

        # Save confusion matrix
        matrix_path = os.path.join(self.output_dir, "confusion_matrix.png")
        plt.savefig(matrix_path)
        plt.close()

        return {
            "accuracy": accuracy,
            "confusion_matrix": conf_matrix,
            "classification_report": class_report,
        }

    def save_model(self):
        """
        Save trained model components
        """
        # Prepare save path
        model_path = os.path.join(self.output_dir, "disease_prediction_model.pkl")

        # Save model components
        with open(model_path, "wb") as f:
            pickle.dump(
                {
                    "model": self.model,
                    "label_encoder": self.label_encoder,
                    "feature_names": list(self.X_train.columns),
                },
                f,
            )

        print(f"Model saved to {model_path}")


def main():
    # Configurable paths
    training_path = "Training.csv"
    testing_path = "Testing.csv"  # Optional

    # Initialize model
    predictor = DiseasePredictionModel(
        training_data_path=training_path, testing_data_path=testing_path
    )

    # Load data
    predictor.load_data()

    # Preprocess
    predictor.preprocess_data()

    # Train model
    training_results = predictor.train_model()
    print("Cross-Validation Results:", training_results)

    # Evaluate
    evaluation_results = predictor.evaluate_model()
    print("Model Evaluation Metrics:", evaluation_results)

    # Save model
    predictor.save_model()


if __name__ == "__main__":
    main()
