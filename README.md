# ZAD5_WSI – Sztuczne Sieci Neuronowe (MLP od podstaw)

**Autorzy:** Mikołaj Wróbel, Kacper Maciejko  
**Kurs:** Wstęp do Sztucznej Inteligencji (WSI) – Laboratorium 5  
**Data:** 2026

---

## Opis projektu

Implementacja wielowarstwowego perceptronu (MLP) zbudowana **wyłącznie z NumPy** (bez PyTorch / TensorFlow). Projekt obejmuje:

- Pełną implementację sieci neuronowej (warstwy, neurony, forward pass, backpropagation)
- Mini-batch SGD z propagacją wsteczną
- Augmentację danych treningowych (×5)
- Bayesowską optymalizację hiperparametrów (Gaussian Process + UCB)
- Zapis i wczytywanie wytrenowanych modeli do pliku `.npz`
- Interaktywne wizualizacje Plotly (architektura 3D, krzywe uczenia, macierz pomyłek)

---

## Struktura repozytorium

```
ZAD5_WSI/
├── Code/
│   ├── Import_training_data/
│   │   └── import_traing_data.py    # DataLoader: augmentacja, normalizacja, podział danych
│   ├── main/
│   │   └── main.py                  # Punkt wejścia (tryb manualny i automatyczny)
│   ├── Optymalize_Model_Size/
│   │   └── Bayes_optymalization.py  # Bayesowska optymalizacja (GP + UCB)
│   ├── train/
│   │   ├── model.py                 # Klasy: Node, Layer, Model (forward, save, load)
│   │   └── training.py              # TrainModel: backprop, mini-batch SGD
│   └── Visualization/
│       └── network_visualization.py # Wizualizacje Plotly (2D, 3D, krzywe uczenia)
├── saved_models/
│   ├── bayes_best_model.npz         # Najlepszy model z optymalizacji Bayesowskiej
│   └── manual_best_model.npz        # Najlepszy model z trybu manualnego
├── report.ipynb                     # Raport z eksperymentami porównawczymi
├── vizualize_choosed_model.ipynb    # Interaktywna wizualizacja zapisanego modelu
└── README.md
```

---

## Dane

Zbiór **Digits** (sklearn) — 1797 obrazów cyfr 0–9, rozdzielczość 8×8 pikseli, wartości 0–16.

### Podział danych

| Zbiór       | Udział |
|-------------|--------|
| Treningowy  | 70%    |
| Walidacyjny | 15%    |
| Testowy     | 15%    |

### Augmentacja treningowa (×5)

Każdy obraz treningowy jest rozszerzany do 5 wersji:

1. Oryginał
2. Pixel dropout (losowe wyzerowanie 2 jasnych pikseli)
3. Pogrubienie (maximum filter 2×2)
4. Przesunięcie (losowy shift o 1 piksel w osi X i/lub Y)
5. Szum Gaussowski (σ = 1.5)

Normalizacja: wartości pikseli z zakresu 0–16 są dzielone przez 16 → zakres **[0, 1]**.

---

## Architektura sieci

- Klasy: `Model → [Layer] → [Node]`
- Konfigurowalny: dowolna liczba warstw ukrytych i neuronów na warstwę
- Aktywacja warstw ukrytych: **ReLU**
- Aktywacja warstwy wyjściowej: **Softmax**
- Funkcja straty: **entropia krzyżowa** (cross-entropy)
- Optimizer: **Mini-batch SGD** z propagacją wsteczną

---

## Uruchomienie

```bash
python ZAD5_WSI/Code/main/main.py
```

### Tryb manualny (1)

Użytkownik ręcznie podaje:
- Liczbę warstw ukrytych i neurony na warstwę (np. `128 64` dla dwóch warstw)
- Liczbę epok, współczynnik uczenia η, rozmiar mini-batcha
- Liczbę powtórzeń treningu — zachowywany jest model z najwyższą **val accuracy**
- Opcjonalnie: zapis modelu do `saved_models/manual_best_model.npz`

### Tryb automatyczny – Bayesowska optymalizacja (2)

Optimizer przeszukuje przestrzeń hiperparametrów za pomocą procesu Gaussa (GP) z funkcją akwizycji UCB:

| Parametr               | Zakres     |
|------------------------|------------|
| Liczba warstw ukrytych | 1 – 4      |
| Neurony na warstwę     | 16 – 256   |
| Współczynnik uczenia η | 0.001 – 0.1|
| Liczba epok            | 50 – 200   |
| Rozmiar mini-batcha    | 16 – 128   |

Po znalezieniu najlepszej konfiguracji model jest trenowany 5 razy; zachowywany jest ten z najwyższą **val accuracy**.  
Opcjonalnie: zapis do `saved_models/bayes_best_model.npz`.

---

## Zapis i wczytywanie modelu

```python
# Zapis
model.save('saved_models/my_model')          # tworzy my_model.npz

# Wczytanie
from Code.train.model import Model
model = Model.load('saved_models/my_model')  # wczytuje wagi + architekturę
```

Pliki `.npz` przechowują wagi, biasy oraz pełną definicję architektury (rozmiary warstw i funkcje aktywacji).

---

## Wizualizacja zapisanego modelu

Otwórz `vizualize_choosed_model.ipynb` i ustaw `MODEL_PATH` na ścieżkę do wybranego pliku `.npz`.  
Notebook udostępnia:

- Architekturę sieci w widoku 3D (obracanie, zoom)
- Macierz pomyłek na zbiorze testowym
- Interaktywny podgląd predykcji — wybierz cyfrę z dropdown i losuj próbki z zestawu testowego
