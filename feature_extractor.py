import pandas as pd
import numpy as np
import glob
import os

def extract_features_from_file(file_path):
    df = pd.read_csv(file_path)

    # Ordina per timestamp
    df = df.sort_values('timestamp')

    # Calcola durata pressione tasti: associa keydown e keyup per ogni tasto
    keydowns = df[df['event'] == 'keydown']
    keyups = df[df['event'] == 'keyup']

    durations = []
    intervals = []
    
    # Map keydowns by key + timestamp index
    keydown_times = {}
    for idx, row in keydowns.iterrows():
        keydown_times[(row['key'], idx)] = row['timestamp']

    # Per ogni keyup, trova il keydown corrispondente più vicino (precedente)
    for idx, row in keyups.iterrows():
        key = row['key']
        ts_up = row['timestamp']
        candidates = [(k, t) for (k, i), t in keydown_times.items() if k == key and i < idx]
        if not candidates:
            continue
        # Prendi il keydown più recente
        i_close, ts_down = max(candidates, key=lambda x: x[1])
        duration = ts_up - ts_down
        durations.append(duration)

    # Calcola intervalli tra keydown consecutivi
    keydown_times_list = keydowns['timestamp'].tolist()
    intervals = np.diff(keydown_times_list)

    features = {}

    if durations:
        features['avg_key_duration'] = np.mean(durations)
        features['std_key_duration'] = np.std(durations)
    else:
        features['avg_key_duration'] = 0
        features['std_key_duration'] = 0

    if len(intervals) > 0:
        features['avg_interval'] = np.mean(intervals)
        features['std_interval'] = np.std(intervals)
    else:
        features['avg_interval'] = 0
        features['std_interval'] = 0

    features['num_events'] = len(df)

    return features

def create_dataset_from_folder(folder_path):
    files = glob.glob(os.path.join(folder_path, "*.csv"))
    all_features = []
    labels = []

    for f in files:
        features = extract_features_from_file(f)
        df_tmp = pd.read_csv(f)
        label = df_tmp['label'].iloc[0] if 'label' in df_tmp.columns else 'unknown'
        all_features.append(features)
        labels.append(label)

    df_features = pd.DataFrame(all_features)
    df_features['label'] = labels
    return df_features

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract features from keyboard event CSVs")
    parser.add_argument('--input_dir', type=str, default='dataset', help="Folder with CSV files")
    parser.add_argument('--output_file', type=str, default='dataset/features_dataset.csv', help="Output CSV file with features")
    args = parser.parse_args()

    df_features = create_dataset_from_folder(args.input_dir)
    df_features.to_csv(args.output_file, index=False)
    print(f"[INFO] Features dataset saved to {args.output_file}")
