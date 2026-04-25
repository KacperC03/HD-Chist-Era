## YOLO Crop Classification Prototype

Projekt prototypowy wykorzystujący model YOLO (wersja 8 - dla lepszego hardware niż co ja miałem można podbić) do klasyfikacji upraw roślin na podstawie obrazów lotniczych. System dzieli duże obrazy na mniejsze kafelki (tiles), trenuje model klasyfikacyjny i ewaluuje jego wydajność.

### Instalacja

1. Sklonuj repozytorium:
```bash
git clone <repository-url>
cd YOLO-prototype
```

2. Utwórz wirtualne środowisko:
```bash
python -m venv venv
venv\Scripts\activate  # Windows
# lub
source venv/bin/activate  # Linux/Mac
```

3. Zainstaluj zależności:
```bash
pip install -r requirements.txt
```

### Dataset

https://drive.google.com/file/d/1RbZB62FqgN1b5acDF7bDR2A-fo_EhplA/view?usp=drive_link

### Konfiguracja

Utwórz plik `.env` w głównym katalogu projektu z następującymi zmiennymi lub przekopiuj je z env.example:

```env
RAW_DATA_DIR=data
TILED_DATA_DIR=tiled_data
EXPERIMENT_NAME=yolo_model_v1
SPLIT_STRATEGY=sequential
SEQUENCE_GAP=5
```

- `RAW_DATA_DIR`: Ścieżka do surowych danych obrazów
- `TILED_DATA_DIR`: Ścieżka wyjściowa dla podzielonych obrazów
- `EXPERIMENT_NAME`: Nazwa eksperymentu (katalog w runs/)
- `SPLIT_STRATEGY`: Strategia podziału danych ('random' lub 'sequential')
- `SEQUENCE_GAP`: Przerwa między zbiorami dla strategii sekwencyjnej

### Przygotowanie danych

1. Umieść surowe obrazy w katalogu `RAW_DATA_DIR` (domyślnie `data/`)

2. Uruchom przygotowanie danych:
```bash
python data_prep.py
```

Skrypt automatycznie:
- Podzieli obrazy na kafelki 640x640 pikseli
- Zastosuje inteligentne nakładanie dla klas z niedoborem danych
- Podzieli dane na zbiory: treningowy (80%), walidacyjny (10%), testowy (10%)

### Trening modelu

Uruchom trening:
```bash
python train.py
```

Parametry treningu (w train.py):
- Model: yolov8s-cls.pt
- Epoki: 30
- Rozmiar obrazów: 640x640
- Batch size: 4 (na spokojnie można podbić do 16 przy 8GB VRAM)
- Device: GPU (jeśli dostępne)

Model zostanie zapisany w `runs/classify/crop_classification/{EXPERIMENT_NAME}/`

### Ewaluacja

Uruchom ewaluację na zbiorze testowym:
```bash
python evaluate.py
```

Wyniki zostaną zapisane w `evaluation_results/` z metrykami i raportem.

### Narzędzia deweloperskie

W katalogu `dev-tools/` dostępne są dodatkowe skrypty:

- `check_device.py`: Sprawdzenie dostępności GPU
- `count_dataset.py`: Analiza rozkładu klas w dataset
- `test_prep.py`: Testowanie przygotowania danych

### Wyniki eksperymentów

Aktualne wyniki treningów dostępne w `runs/classify/crop_classification/`:
- yolo_model_v1: Pierwsza wersja modelu
- yolo_model_v1-2: Udoskonalona wersja

Raporty ewaluacji w `evaluation_results/` zawierają:
- Metryki dokładności
- Macierz pomyłek
- Rozkład klas

### Struktura projektu

```
YOLO-prototype/
├── data_prep.py         # Przygotowanie danych
├── train.py             # Trening modelu
├── evaluate.py          # Ewaluacja
├── requirements.txt     # Zależności Python
├── README.md            # Ten plik
├── yolo26n.pt           # Model YOLOv8 nano
├── yolov8s-cls.pt       # Model klasyfikacyjny
├── dev-tools/           # Narzędzia deweloperskie
├── evaluation_results/  # Wyniki ewaluacji
├── runs/                # Wyniki treningów
├── test_results/        # Wyniki testów
├── .env.example         # Przykładowa konfiguracja
└── .env                 # Konfiguracja (utwórz własny)
```

