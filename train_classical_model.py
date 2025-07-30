import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import joblib
import numpy as np

def main():
    # 1. Carica dati (feature aggregate)
    df = pd.read_csv("dataset/features_dataset.csv")  # aggiorna il path se serve
    print("Dataset shape:", df.shape)

    # 2. Preprocessing
    features = df.drop(columns=["label"])
    labels = df["label"]

    # Standardizza feature
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # Encode label (bot=0, human=1)
    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)

    # 3. Split train/test con stratificazione
    X_train, X_test, y_train, y_test = train_test_split(
        features_scaled, labels_encoded, test_size=0.2, random_state=42, stratify=labels_encoded
    )

    print("y_train class distribution:", dict(zip(*np.unique(y_train, return_counts=True))))
    print("y_test class distribution:", dict(zip(*np.unique(y_test, return_counts=True))))

    # 4. Modelli da allenare
    models = {
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "LDA": LinearDiscriminantAnalysis(),
        "SVM": SVC(kernel='linear', probability=True, random_state=42)
    }

    best_model_name = None
    best_model = None
    best_accuracy = 0

    # 5. Training e valutazione
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        print(f"Classification report for {name}:")
        print(classification_report(y_test, y_pred, target_names=le.classes_))

        accuracy = model.score(X_test, y_test)
        print(f"{name} Accuracy: {accuracy:.4f}")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model
            best_model_name = name

    # 6. Salva il miglior modello + label encoder + scaler
    print(f"\nBest model: {best_model_name} with accuracy {best_accuracy:.4f}")

    joblib.dump(best_model, "model.pkl")
    joblib.dump(le, "label_encoder.pkl")
    joblib.dump(scaler, "scaler.pkl")
    print("Modello, label encoder e scaler salvati.")

if __name__ == "__main__":
    main()

