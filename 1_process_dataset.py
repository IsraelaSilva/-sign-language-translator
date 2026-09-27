"""
PASSO 1: Processar o dataset ASL
---------------------------------
Este script le as imagens coletadas, extrai os 21 pontos da mao
com o MediaPipe, e salva tudo em um arquivo CSV para treinar o modelo.

Como usar:
  python 1_process_dataset.py
"""

import os
import cv2
import csv
import urllib.request
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# --- Baixar modelo MediaPipe se nao existir ---
MODEL_PATH = "hand_landmarker.task"
MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

if not os.path.exists(MODEL_PATH):
    print("Baixando modelo MediaPipe (~7MB)...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Modelo baixado!")

# --- Inicializar detector ---
base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
options = mp_vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.3
)
detector = mp_vision.HandLandmarker.create_from_options(options)

# --- Configuracoes ---
DATASET_PATH = "data/asl_alphabet_train/asl_alphabet_train"
OUTPUT_CSV   = "data/landmarks.csv"

# --- Criar cabecalho do CSV ---
# 21 pontos, cada um com x, y, z = 63 colunas + coluna "label"
header = []
for i in range(21):
    header.extend([f"x{i}", f"y{i}", f"z{i}"])
header.append("label")

rows = []

# --- Processar cada pasta (letra) ---
labels = sorted(os.listdir(DATASET_PATH))
total_labels = len(labels)

for idx, label in enumerate(labels):
    label_path = os.path.join(DATASET_PATH, label)

    if not os.path.isdir(label_path):
        continue

    images = os.listdir(label_path)
    print(f"[{idx+1}/{total_labels}] Processando '{label}': {len(images)} imagens...")

    count = 0
    for img_file in images:
        img_path = os.path.join(label_path, img_file)

        # Ler imagem
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            continue

        # Converter para RGB (MediaPipe exige RGB)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Criar imagem MediaPipe e detectar
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        result = detector.detect(mp_image)

        # Se encontrou uma mao na imagem
        if result.hand_landmarks:
            landmarks = result.hand_landmarks[0]

            # Normalizar: subtrair posicao do pulso (landmark 0)
            wrist = landmarks[0]
            row = []
            for lm in landmarks:
                row.extend([
                    lm.x - wrist.x,
                    lm.y - wrist.y,
                    lm.z - wrist.z
                ])
            row.append(label)
            rows.append(row)
            count += 1

    print(f"   -> {count} maos detectadas")

# --- Salvar CSV ---
os.makedirs("data", exist_ok=True)
with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(rows)

print(f"\nConcluido! {len(rows)} amostras salvas em '{OUTPUT_CSV}'")

detector.close()
