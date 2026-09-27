"""
PASSO 3: Traducao a letra em tempo real
------------------------------------------------
Abre a webcam e traduz gestos para texto em tempo real.

Controles:
  - Segure um gesto por 1.5s para confirmar a letra
  - Pressione C para limpar a frase
  - Pressione Q para sair

Como usar:
  python 3_app.py
"""

import cv2
import numpy as np
import joblib
import time
import os
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

# --- Carregar modelo treinado ---
print("Carregando modelo...")
model = joblib.load("model/asl_model.pkl")
print("Modelo carregado!")

# --- Inicializar MediaPipe ---
base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
options = mp_vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=mp_vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_tracking_confidence=0.5
)
detector = mp_vision.HandLandmarker.create_from_options(options)

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17)
]

# --- Tema rosa ---
# (valores em BGR)
COR_FUNDO     = (45, 20, 50)       # roxo escuro
COR_ROSA      = (210, 160, 230)    # rosa claro
COR_ROSA_VIVO = (160,  80, 210)    # rosa mais vibrante
COR_BRANCO    = (255, 240, 248)    # branco rosado
COR_MUTED     = (160, 120, 170)    # rosa acinzentado
COR_BARRA     = (190, 120, 240)    # lilás/rosa para barra


def overlay(frame, x1, y1, x2, y2, cor, alpha=0.82):
    """Retangulo semitransparente."""
    bg = frame.copy()
    cv2.rectangle(bg, (x1, y1), (x2, y2), cor, -1)
    cv2.addWeighted(bg, alpha, frame, 1 - alpha, 0, frame)


def barra(frame, x, y, larg, alt, prog, cor_bg, cor_fill):
    """Barra de progresso com pontas arredondadas."""
    r = alt // 2
    # fundo
    cv2.rectangle(frame, (x + r, y), (x + larg - r, y + alt), cor_bg, -1)
    cv2.circle(frame, (x + r,        y + r), r, cor_bg, -1)
    cv2.circle(frame, (x + larg - r, y + r), r, cor_bg, -1)
    # fill
    if prog > 0:
        fx = x + r + int(prog * (larg - 2 * r))
        fx = max(fx, x + r)
        cv2.rectangle(frame, (x + r, y), (fx, y + alt), cor_fill, -1)
        cv2.circle(frame, (x + r, y + r), r, cor_fill, -1)
        if fx > x + r:
            cv2.circle(frame, (fx, y + r), r, cor_fill, -1)


def draw_hand(frame, landmarks, h, w):
    """Desenha esqueleto da mao."""
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for s, e in HAND_CONNECTIONS:
        cv2.line(frame, pts[s], pts[e], COR_ROSA, 2)
    for px, py in pts:
        cv2.circle(frame, (px, py), 5, COR_BRANCO, -1)
        cv2.circle(frame, (px, py), 5, COR_ROSA_VIVO, 1)

# Nomes curtos para as classes especiais
ABREV = {"space": "ESP", "del": "DEL", "nothing": "-"}

def escala_que_cabe(texto, larg_max, escala, grossura):
    """Diminui a fonte ate o texto caber em larg_max pixels."""
    while escala > 0.3:
        (tw, _), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, escala, grossura)
        if tw <= larg_max:
            break
        escala -= 0.05
    return escala

# Estado
sentence         = ""
last_letter      = ""
last_letter_time = 0.0
HOLD_TIME        = 1.5
cursor_visible   = True
cursor_timer     = time.time()
progress         = 0.0

# Alturas fixas dos paineis (evita sobreposicao)
TOP_H    = 115   # altura do painel superior
BOT_H    = 70    # altura do painel inferior
SIDE_W   = 190   # largura do painel lateral (top 3)

cap = cv2.VideoCapture(0)
print("Webcam aberta. Q = sair, C = limpar.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    frame_rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    timestamp_ms = int(time.time() * 1000)
    mp_image     = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    result       = detector.detect_for_video(mp_image, timestamp_ms)

    predicted_letter = ""
    confidence       = 0.0
    top3             = []

    if result.hand_landmarks:
        landmarks = result.hand_landmarks[0]
        draw_hand(frame, landmarks, h, w)

        wrist = landmarks[0]
        row = []
        for lm in landmarks:
            row.extend([lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z])

        proba            = model.predict_proba([row])[0]
        confidence       = float(max(proba))
        predicted_letter = model.classes_[int(np.argmax(proba))]
        top3_idx         = np.argsort(proba)[::-1][:3]
        top3             = [(model.classes_[i], float(proba[i])) for i in top3_idx]

        current_time = time.time()
        if predicted_letter == last_letter:
            elapsed  = current_time - last_letter_time
            progress = min(elapsed / HOLD_TIME, 1.0)
            if elapsed >= HOLD_TIME:
                if predicted_letter == "space":
                    sentence += " "
                elif predicted_letter == "del":
                    sentence = sentence[:-1]
                elif predicted_letter != "nothing":
                    sentence += predicted_letter.upper()
                last_letter_time = current_time
        else:
            last_letter      = predicted_letter
            last_letter_time = current_time
            progress         = 0.0
    else:
        progress = 0.0

    # 1. PAINEL SUPERIOR (y: 0 -> TOP_H)
    overlay(frame, 0, 0, w, TOP_H, COR_FUNDO)
    cv2.line(frame, (0, TOP_H), (w, TOP_H), COR_ROSA, 1)

    if predicted_letter:
        # Fonte menor para palavras longas (space, del, nothing)
        if len(predicted_letter) == 1:
            escala, grossura, y_texto = 3.0, 5, 88
        else:
            escala, grossura, y_texto = 1.4, 3, 75

        display = predicted_letter.upper()
        cv2.putText(frame, display, (20, y_texto),
                    cv2.FONT_HERSHEY_SIMPLEX, escala, COR_ROSA, grossura)

        # Posicionar % logo apos o texto
        (tw, _), _ = cv2.getTextSize(display, cv2.FONT_HERSHEY_SIMPLEX, escala, grossura)
        cv2.putText(frame, f"{confidence*100:.0f}%", (25 + tw, y_texto - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, COR_MUTED, 2)
        barra(frame, 20, 100, 260, 10, progress, (80, 50, 85), COR_BARRA)
    else:
        cv2.putText(frame, "Mostre sua mao...", (20, 68),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, COR_MUTED, 2)

    # 2. PAINEL LATERAL - TOP 3 (canto direito, abaixo do painel superior)
    if top3:
        sx = w - SIDE_W
        sy = TOP_H + 10
        sy_end = TOP_H + 130
        overlay(frame, sx, sy, w, sy_end, COR_FUNDO, alpha=0.78)
        cv2.line(frame, (sx, sy), (sx, sy_end), COR_ROSA, 1)

        cv2.putText(frame, "TOP 3", (sx + 10, sy + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, COR_MUTED, 1)

        for i, (letra, prob) in enumerate(top3):
            ty  = sy + 52 + i * 30 
            cor = COR_ROSA if i == 0 else COR_MUTED
            label = ABREV.get(letra, letra.upper())
            esc   = escala_que_cabe(label, 45, 0.8, 2)   # coluna de 45px

            cv2.putText(frame, label, (sx + 8 , ty),
                        cv2.FONT_HERSHEY_SIMPLEX, esc, cor, 2)
            barra(frame, sx + 55, ty - 13, 100, 9, prob, (80, 50, 85), cor)
            cv2.putText(frame, f"{prob*100:.0f}%", (sx + 155, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, COR_MUTED, 1)

    # 3. PAINEL INFERIOR (y: h-BOT_H -> h)
    overlay(frame, 0, h - BOT_H, w, h, COR_FUNDO)
    cv2.line(frame, (0, h - BOT_H), (w, h - BOT_H), COR_ROSA, 1)

    # cursor piscando
    if time.time() - cursor_timer > 0.5:
        cursor_visible = not cursor_visible
        cursor_timer   = time.time()
    cursor = "|" if cursor_visible else " "

    cv2.putText(frame, f"{sentence}{cursor}", (15, h - 22),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, COR_BRANCO, 2)
    cv2.putText(frame, "Q sair   C limpar", (w - 195, h - 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, COR_MUTED, 1)

    cv2.imshow("ASL Sign Language Translator", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        sentence = ""

cap.release()
cv2.destroyAllWindows()
detector.close()
print("App encerrado.")
