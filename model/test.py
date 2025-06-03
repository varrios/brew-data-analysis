import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from constants import DATAFILE_PATH
import warnings

warnings.filterwarnings('ignore')

# Wczytanie danych
df = pd.read_csv(DATAFILE_PATH, encoding="ISO-8859-1")

# Usunięcie zbędnych kolumn
df.drop(columns=['BeerID', 'Name', 'URL', 'UserId', 'SugarScale', 'StyleID',
                 'PrimingAmount', 'PrimingMethod'], inplace=True, errors='ignore')

# Usunięcie kolumn z dużą liczbą braków
df.drop(columns=['BoilGravity', 'MashThickness', 'PitchRate', 'PrimaryTemp'], inplace=True, errors='ignore')

# Usunięcie wierszy z brakującą wartością 'Style'
df.dropna(subset=['Style'], inplace=True)

# Uzupełnienie braków medianą
df.fillna(df.median(numeric_only=True), inplace=True)

# Kodowanie zmiennej kategorycznej
df = pd.get_dummies(df, columns=['BrewMethod'], drop_first=True)

# Separacja cech i etykiet
X = df.drop('Style', axis=1)
y = df['Style']

# Standaryzacja
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Podział na dane treningowe i testowe
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, stratify=y, test_size=0.2, random_state=42)

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print("=== Random Forest ===")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Macro F1:", f1_score(y_test, y_pred_rf, average='macro'))
print(classification_report(y_test, y_pred_rf))

# Macierz pomyłek
plt.figure(figsize=(12, 6))
sns.heatmap(confusion_matrix(y_test, y_pred_rf), cmap='Blues', cbar=False)
plt.title('Random Forest - Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig("confusion_matrix_rf.png")
print("Macierz pomyłek zapisana jako confusion_matrix_rf.png")
