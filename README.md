# Sign Language Translator

Real-time hand gesture recognition that translates ASL alphabet into text using MediaPipe and Machine Learning.

![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python)
![MediaPipe](https://img.shields.io/badge/MediaPipe-1.0-green)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-red)
![scikit-learn](https://img.shields.io/badge/scikit--learn-RandomForest-orange)

---

## Como funciona

1. A webcam captura o video em tempo real
2. O **MediaPipe** detecta 21 pontos da mao em cada frame
3. Os pontos sao normalizados e enviados ao modelo
4. Um **Random Forest** treinado nas suas proprias maos preve a letra
5. Segure o gesto por 1.5s para confirmar e montar a frase

---

## Resultados

- **Acuracia no teste:** 99.91%
- **Dataset:** coleta propria via webcam (5.787 amostras, 29 classes)
- **Tempo de resposta:** tempo real (~30fps)

---

## Instalacao

```bash
git clone https://github.com/IsraelaSilva/-sign-language-translator.git
cd -sign-language-translator
pip install -r requirements.txt
```

---

## Como usar

### 1. Coletar seus proprios dados
```bash
python 0_collect_data.py
```
Faz gestos na frente da webcam. O script tira 200 fotos por letra automaticamente.

### 2. Processar o dataset
```bash
python 1_process_dataset.py
```
Extrai os 21 pontos da mao de cada foto e salva em CSV.

### 3. Treinar o modelo
```bash
python 2_train_model.py
```
Treina o Random Forest e salva o modelo + matriz de confusao.

### 4. Rodar o app
```bash
python 3_app.py
```
Abre a webcam. Segure um gesto por 1.5s para confirmar a letra.

**Controles:**
- `C` — limpar a frase
- `Q` — sair

---

## Stack

| Biblioteca | Uso |
|---|---|
| MediaPipe | Deteccao dos 21 pontos da mao |
| OpenCV | Webcam e interface visual |
| scikit-learn | Treinamento do modelo (Random Forest) |
| NumPy | Processamento dos dados |
| Matplotlib | Matriz de confusao |
