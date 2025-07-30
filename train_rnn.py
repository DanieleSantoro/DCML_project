import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import glob
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from RNN import KeystrokeRNN  # assicurati che questo file contenga la classe KeystrokeRNN

class KeystrokeDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = torch.tensor(np.array(sequences), dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

def create_sequences(features_df, labels_series, seq_length=10):
    sequences = []
    labels_out = []

    if len(features_df) <= seq_length:
        print("Not enough data to create sequences.")
        return sequences, labels_out

    for i in range(len(features_df) - seq_length):
        seq = features_df.iloc[i:i + seq_length].values
        label = labels_series.iloc[i + seq_length]
        sequences.append(seq)
        labels_out.append(label)

    return sequences, labels_out

def main():
    folder = "processed_dataset"
    all_files = glob.glob(f"{folder}/*.csv")

    dfs = []
    for file in all_files:
        df_tmp = pd.read_csv(file)
        dfs.append(df_tmp)

    df = pd.concat(dfs, ignore_index=True)
    print("Dataset shape:", df.shape)

    features = df.drop(columns=["label"])
    labels = df["label"]

    scaler = StandardScaler()
    features = pd.DataFrame(scaler.fit_transform(features), columns=features.columns)

    le = LabelEncoder()
    labels = pd.Series(le.fit_transform(labels))

    sequences, seq_labels = create_sequences(features, labels, seq_length=2)

    print("Num sequences:", len(sequences))
    if sequences:
        print("Example sequence shape:", len(sequences[0]), "x", len(sequences[0][0]))

    X_train, X_test, y_train, y_test = train_test_split(sequences, seq_labels, test_size=0.2, random_state=42)

    train_dataset = KeystrokeDataset(X_train, y_train)
    test_dataset = KeystrokeDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size = len(features.columns)
    model = KeystrokeRNN(input_size=input_size).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    epochs = 15
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss:.4f}")

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            _, predicted = torch.max(outputs.data, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()

    accuracy = 100 * correct / total if total > 0 else 0
    print(f"Test Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    main()

