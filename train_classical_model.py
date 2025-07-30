# train_classical_models.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# Caricamento dataset
df = pd.read_csv('dataset/features_dataset.csv')
X = df.drop(columns=['label'])
y = df['label']

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modelli
models = {
    "RandomForest": RandomForestClassifier(n_estimators=100),
    "LDA": LinearDiscriminantAnalysis(),
    "SVM": SVC()
}

# Addestramento e valutazione
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"=== {name} ===")
    print(classification_report(y_test, y_pred))
