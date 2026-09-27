"""
PASSO 0: Coletando dados
--------------------------------------
Abre a webcam e tira fotos dos seus gestos automaticamente.
Faz isso para cada letra do alfabeto ASL (A-Z + space + del + nothing).

Como usar:
  python 0_collect_data.py

Controles durante a coleta:
  - ESPACO: comecar a tirar fotos do gesto atual
  - Q: pular esta letra e ir para a proxima
  - ESC: encerrar o programa
"""

import cv2 #cv2 para abrir a webcam e tirar fotos
import os #os para criar pastas e salvar fotos
import time #time para contagem regressiva

# --- Configuracoes ---
DATASET_PATH     = "data/asl_alphabet_train/asl_alphabet_train"
PHOTOS_PER_CLASS = 200
COUNTDOWN        = 3

# Lista de gestos a coletar
CLASSES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["space", "del", "nothing"]
#space = mao aberta, del = mao fechada, nothing = mao relaxada na frente da camera

# --- Tema rosa ---
COR_FUNDO     = (45, 20, 50)
COR_ROSA      = (210, 160, 230)
COR_ROSA_VIVO = (160,  80, 210)
COR_BRANCO    = (255, 240, 248)
COR_MUTED     = (160, 120, 170)
COR_BARRA     = (190, 120, 240)


def overlay(frame, x1, y1, x2, y2, cor, alpha=0.82):
    """Retangulo semitransparente."""
    bg = frame.copy()
    cv2.rectangle(bg, (x1, y1), (x2, y2), cor, -1)
    cv2.addWeighted(bg, alpha, frame, 1 - alpha, 0, frame)


def barra(frame, x, y, larg, alt, prog, cor_bg, cor_fill):
    """Barra de progresso com pontas arredondadas."""
    r = alt // 2
    cv2.rectangle(frame, (x + r, y), (x + larg - r, y + alt), cor_bg, -1)
    cv2.circle(frame, (x + r,        y + r), r, cor_bg, -1)
    cv2.circle(frame, (x + larg - r, y + r), r, cor_bg, -1)
    if prog > 0:
        fx = x + r + int(prog * (larg - 2 * r))
        fx = max(fx, x + r)
        cv2.rectangle(frame, (x + r, y), (fx, y + alt), cor_fill, -1)
        cv2.circle(frame, (x + r, y + r), r, cor_fill, -1)
        if fx > x + r:
            cv2.circle(frame, (fx, y + r), r, cor_fill, -1)


# --- Criar pastas ---
for label in CLASSES:
    os.makedirs(os.path.join(DATASET_PATH, label), exist_ok=True)

# --- Abrir webcam ---
cap = cv2.VideoCapture(0)
print("Webcam aberta!")
print(f"Vamos coletar {PHOTOS_PER_CLASS} fotos para cada um dos {len(CLASSES)} gestos.")
print("Pressione ESPACO para comecar cada gesto, Q para pular, ESC para sair.\n")

TOP_H = 110   # altura painel superior
BOT_H = 55    # altura painel inferior

for label in CLASSES:
    label_path = os.path.join(DATASET_PATH, label)

    existing = len(os.listdir(label_path))
    if existing >= PHOTOS_PER_CLASS:
        print(f"'{label}' ja tem {existing} fotos. Pulando...")
        continue

    idx_atual = CLASSES.index(label) + 1
    total     = len(CLASSES)
    print(f"\n--- [{idx_atual}/{total}] Proximo gesto: '{label}' ---")

    # --- Tela de espera ---
    waiting = True
    while waiting:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]

        # Painel superior
        overlay(frame, 0, 0, w, TOP_H, COR_FUNDO)
        cv2.line(frame, (0, TOP_H), (w, TOP_H), COR_ROSA, 1)

        # Barra de progresso geral
        prog_geral = (idx_atual - 1) / total
        barra(frame, 15, 12, w - 30, 8, prog_geral, (80, 50, 85), COR_ROSA)
        cv2.putText(frame, f"{idx_atual}/{total}", (w - 60, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COR_MUTED, 1)

        # Nome do gesto
        cv2.putText(frame, label, (20, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, COR_ROSA, 4)

        # Painel inferior com instrucoes
        overlay(frame, 0, h - BOT_H, w, h, COR_FUNDO)
        cv2.line(frame, (0, h - BOT_H), (w, h - BOT_H), COR_ROSA, 1)
        cv2.putText(frame, "ESPACO = comecar    Q = pular    ESC = sair",
                    (15, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.58, COR_MUTED, 1)

        cv2.imshow("Coletor de Dados - ASL", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):
            waiting = False
        elif key == ord('q'):
            print(f"Pulando '{label}'...")
            waiting = False
            label = None
            break
        elif key == 27:
            print("Encerrando coleta.")
            cap.release()
            cv2.destroyAllWindows()
            exit()

    if label is None:
        continue

    # --- Contagem regressiva ---
    cores_num = [(160, 80, 210), (190, 120, 240), (210, 160, 230)]  # 3=escuro, 2=medio, 1=claro
    for i in range(COUNTDOWN, 0, -1):
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]

        overlay(frame, 0, 0, w, TOP_H, COR_FUNDO)
        cv2.line(frame, (0, TOP_H), (w, TOP_H), COR_ROSA, 1)
        cv2.putText(frame, label, (20, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, COR_ROSA, 4)

        # Numero centralizado
        cor_num   = cores_num[COUNTDOWN - i]
        txt       = str(i)
        ts        = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 7, 12)[0]
        cx        = (w - ts[0]) // 2
        cy        = (h + ts[1]) // 2
        cv2.putText(frame, txt, (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 7, cor_num, 12)

        cv2.imshow("Coletor de Dados - ASL", frame)
        cv2.waitKey(1)
        time.sleep(1)

    # --- Coletar fotos ---
    count = existing
    print(f"Coletando fotos para '{label}'... ({PHOTOS_PER_CLASS - existing} restantes)")

    while count < PHOTOS_PER_CLASS:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]

        img_path = os.path.join(label_path, f"{label}_{count}.jpg")
        cv2.imwrite(img_path, frame)
        count += 1

        prog = count / PHOTOS_PER_CLASS
        pct  = int(prog * 100)

        # Painel superior
        overlay(frame, 0, 0, w, TOP_H, COR_FUNDO)
        cv2.line(frame, (0, TOP_H), (w, TOP_H), COR_ROSA, 1)
        cv2.putText(frame, label, (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, COR_ROSA, 4)
        cv2.putText(frame, f"{count}/{PHOTOS_PER_CLASS} fotos", (20, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COR_MUTED, 1)

        # Painel inferior com barra de progresso
        overlay(frame, 0, h - BOT_H, w, h, COR_FUNDO)
        cv2.line(frame, (0, h - BOT_H), (w, h - BOT_H), COR_ROSA, 1)
        barra(frame, 15, h - 38, w - 80, 16, prog, (80, 50, 85), COR_BARRA)
        cv2.putText(frame, f"{pct}%", (w - 58, h - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, COR_BRANCO, 2)

        cv2.imshow("Coletor de Dados - ASL", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            print("Coleta interrompida. Voce pode continuar depois.")
            cap.release()
            cv2.destroyAllWindows()
            exit()

    print(f"'{label}' concluido! {count} fotos salvas.")

# --- Encerrar ---
cap.release()
cv2.destroyAllWindows()
print("\nColeta concluida!")
print(f"Agora rode: python 1_process_dataset.py")
