"""
PASSO 2: Treinar o modelo
--------------------------
Este script carrega o CSV gerado no passo 1, treina um classificador
Random Forest e salva o modelo treinado.

Como usar:
  Execute depois do passo 1: python 2_train_model.py
"""

import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# --- Carregar dados ---
print("Carregando dados...")
df = pd.read_csv("data/landmarks.csv")

# Separar features (X) do label (y)
X = df.drop("label", axis=1).values
y = df["label"].values

print(f"Total de amostras: {len(X)}")
print(f"Classes encontradas: {sorted(set(y))}")

# --- Dividir em treino e teste ---
# 80% treino, 20% teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y  # garante proporcao igual de cada letra
)

print(f"\nTreino: {len(X_train)} amostras | Teste: {len(X_test)} amostras")

# --- Treinar modelo ---
# n_estimators=100: usa 100 arvores de decisao
# n_jobs=-1: usa todos os nucleos do processador
print("\nTreinando modelo (pode demorar alguns minutos)...")
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# --- Avaliar ---
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n=== Resultados ===")
print(f"Acuracia geral: {accuracy * 100:.2f}%")
print("\nRelatorio por letra:")
print(classification_report(y_test, y_pred))

# --- Salvar modelo ---
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/asl_model.pkl")
print("Modelo salvo em 'model/asl_model.pkl'")

# --- Salvar matriz de confusao ---
labels_sorted = sorted(set(y))
cm = confusion_matrix(y_test, y_pred, labels=labels_sorted)

plt.figure(figsize=(16, 13))
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.title("Matriz de Confusao - ASL Alphabet", fontsize=16)
plt.colorbar()
plt.xticks(range(len(labels_sorted)), labels_sorted, fontsize=9)
plt.yticks(range(len(labels_sorted)), labels_sorted, fontsize=9)
plt.xlabel("Predicao", fontsize=12)
plt.ylabel("Valor Real", fontsize=12)
plt.tight_layout()
plt.savefig("model/confusion_matrix.png", dpi=150)
print("Matriz de confusao salva em 'model/confusion_matrix.png'")
