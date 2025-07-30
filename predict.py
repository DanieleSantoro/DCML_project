import argparse
import pandas as pd
import joblib

def main(input_file, model_path, encoder_path):
    # Carica dati
    df = pd.read_csv(input_file)

    # Carica modello e label encoder
    clf = joblib.load(model_path)
    le = joblib.load(encoder_path)

    # Predici
    prediction = clf.predict(df)

    # Decodifica etichetta
    predicted_label = le.inverse_transform(prediction)[0]

    print(f"\n Risultato: {predicted_label.upper()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', required=True, help="CSV con le feature da classificare")
    parser.add_argument('--model_path', default='model.pkl', help="Path al modello")
    parser.add_argument('--encoder_path', default='label_encoder.pkl', help="Path al label encoder")

    args = parser.parse_args()
    main(args.input_file, args.model_path, args.encoder_path)
