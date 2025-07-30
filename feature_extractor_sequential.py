import os
import pandas as pd

def process_file(filepath, output_dir):
    df = pd.read_csv(filepath)

    # Pulizia e codifica
    df['event'] = df['event'].str.strip().str.lower()
    df['event_code'] = df['event'].map({'keydown': 1, 'keyup': 0})
    df['key_code'] = df['key'].astype(str).astype('category').cat.codes

    # Ordina per timestamp
    df = df.sort_values(by='timestamp')

    # Seleziona colonne utili
    df_out = df[['event_code', 'key_code', 'timestamp', 'label']]

    # Salva il csv elaborato in output_dir con lo stesso nome file
    base = os.path.basename(filepath)
    out_path = os.path.join(output_dir, base)
    df_out.to_csv(out_path, index=False)
    print(f"Processed {filepath} -> {out_path}")

def main():
    input_dir = 'dataset'  # cartella con i raw csv
    output_dir = 'processed_dataset'  # cartella dove salvi dati puliti
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if filename.endswith('.csv'):
            filepath = os.path.join(input_dir, filename)
            process_file(filepath, output_dir)

if __name__ == "__main__":
    main()
