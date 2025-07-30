import argparse
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import joblib

def main(input_file, model_output):
    # Carica dataset
    df = pd.read_csv(input_file)

    # Dividi X e y
    X = df.drop('label', axis=1)
    y = df['label']

    # Codifica le label (human -> 0, bot -> 1)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Salva il codificatore per dopo
    joblib.dump(le, 'label_encoder.pkl')

    # Dividi train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    # Inizializza modello
    clf = RandomForestClassifier(n_estimators=100, random_state=42)

    # Allenamento
    clf.fit(X_train, y_train)

    # Valutazione
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Salva modello
    joblib.dump(clf, model_output)
    print(f"\n✅ Modello salvato in: {model_output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', required=True, help="Path al file CSV delle feature")
    parser.add_argument('--model_output', required=True, help="Path dove salvare il modello")

    args = parser.parse_args()
    main(args.input_file, args.model_output)
