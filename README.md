mAP before augmentation: 0.45257
mAP after augmentation: 0.51702
mAP after augmentation with tuning:0.54587

# Output of train37

Validating traffic_training/train37/weights/best.pt...
Ultralytics 8.3.153 🚀 Python-3.11.7 torch-2.6.0+cu124 CUDA:0 (NVIDIA A100 80GB PCIe, 81154MiB)
YOLO11n summary (fused): 100 layers, 2,585,662 parameters, 0 gradients, 6.3 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 10/10 [00:01<00:00,  6.25it/s]
                   all        300       2568      0.644      0.415       0.54      0.303
               bicycle         30         32      0.646      0.312      0.366      0.175
                   bus        220        425      0.774       0.53      0.643      0.418
                   car        239        922      0.768       0.74      0.791       0.52
               minibus          2          2      0.341        0.5      0.497      0.397
               minivan         87        110      0.486      0.436      0.408      0.307
             motorbike        166        335       0.68      0.513      0.559      0.205
                pickup        105        142      0.625      0.246      0.378      0.233
              rickshaw         62        192       0.74      0.609      0.675      0.428
               scooter          1          1          1          0      0.995      0.199
  three wheelers -CNG-        148        252      0.834      0.595      0.702      0.465
                 truck         53         84      0.545       0.56      0.524      0.341
                   van         52         62      0.264      0.248      0.224       0.14
           wheelbarrow          9          9      0.674      0.111      0.257      0.108
Speed: 0.0ms preprocess, 0.4ms inference, 0.0ms loss, 0.6ms postprocess per image
Results saved to traffic_training/train37
💡 Learn more at https://docs.ultralytics.com/modes/train
Saved traffic_training/yolo11n_MAX8/tune_scatter_plots.png
Saved traffic_training/yolo11n_MAX8/tune_fitness.png

Tuner: 3/10 iterations complete ✅ (1794.50s)
Tuner: Results saved to traffic_training/yolo11n_MAX8
Tuner: Best fitness=0.32668 observed at iteration 3
Tuner: Best fitness metrics are {'metrics/precision(B)': 0.64461, 'metrics/recall(B)': 0.41534, 'metrics/mAP50(B)': 0.54009, 'metrics/mAP50-95(B)': 0.30297, 'val/box_loss': 1.37739, 'val/cls_loss': 0.99681, 'val/dfl_loss': 1.15922, 'fitness': 0.32668}
Tuner: Best fitness model is traffic_training/train37
Tuner: Best fitness hyperparameters are printed below.

Printing 'traffic_training/yolo11n_MAX8/best_hyperparameters.yaml'

lr0: 0.00955
lrf: 0.00998
momentum: 0.97997
weight_decay: 0.00054
warmup_epochs: 2.54319
warmup_momentum: 0.94159
box: 7.99327
cls: 0.42895
dfl: 1.61663
hsv_h: 0.01156
hsv_s: 0.75561
hsv_v: 0.31895
degrees: 0.0
translate: 0.10405
scale: 0.4834
shear: 0.0
perspective: 0.0
flipud: 0.0
fliplr: 0.40892
bgr: 0.0
mosaic: 1.0
mixup: 0.0
cutmix: 0.0
copy_paste: 0.0

# Final Output from Tuner

Tuner: 10/10 iterations complete ✅ (5806.02s)
Tuner: Results saved to traffic_training/yolo11n_MAX8
Tuner: Best fitness=0.32668 observed at iteration 3
Tuner: Best fitness metrics are {'metrics/precision(B)': 0.64461, 'metrics/recall(B)': 0.41534, 'metrics/mAP50(B)': 0.54009, 'metrics/mAP50-95(B)': 0.30297, 'val/box_loss': 1.37739, 'val/cls_loss': 0.99681, 'val/dfl_loss': 1.15922, 'fitness': 0.32668}
Tuner: Best fitness model is traffic_training/train37
Tuner: Best fitness hyperparameters are printed below.

Printing 'traffic_training/yolo11n_MAX8/best_hyperparameters.yaml'

lr0: 0.00955
lrf: 0.00998
momentum: 0.97997
weight_decay: 0.00054
warmup_epochs: 2.54319
warmup_momentum: 0.94159
box: 7.99327
cls: 0.42895
dfl: 1.61663
hsv_h: 0.01156
hsv_s: 0.75561
hsv_v: 0.31895
degrees: 0.0
translate: 0.10405
scale: 0.4834
shear: 0.0
perspective: 0.0
flipud: 0.0
fliplr: 0.40892
bgr: 0.0
mosaic: 1.0
mixup: 0.0
cutmix: 0.0
copy_paste: 0.0

# Function oversampling with cmd
for label in labels/*.txt; do
    if grep -E '^(0|1|6|7|8|13|17)[[:space:]]' "$label"; then
        base=$(basename "$label" .txt)
        cp "images/$base.jpg" oversampled/images/"$base"_copy.jpg
        cp "$label" oversampled/labels/"$base"_copy.txt
    fi
done

duplicated classes 0|1|6|7|8|13|17

# Preproc Step
removed taxi, police car and suv from classes and mapped to car (5) reduced total classes from 21 to 18

# Yolo11n Model summary
YOLO11n summary: 181 layers, 2,593,350 parameters, 2,593,334 gradients, 6.5 GFLOPs

# Used Optimizer
optimizer: AdamW(lr=0.0004, momentum=0.9) with parameter groups 57 weight(decay=0.0), 64 weight(decay=0.0005), 63 bias(decay=0.0)

# Color Schema for Graphs
Colors:

darkest blue #032a4d
med dark blue #00457a
medium blue #0458a5
base blue #0476df
light blue #50b1ff
lightest blue #88cbff
grey #9e9e9e

# Dataset Summary
for train:

Total images: 2704
Images with NO label file: 0
Images with EMPTY label file: 2
Total unlabeled images: 2

# Begründung Parameter
Optimal Training Parameters
Once CUDA is enabled, here are the optimal model.train() parameters for your RTX 2060, Ryzen 7 2700X, and traffic object detection use case. These balance accuracy, training speed, and VRAM constraints.
Core Parameters
data (data="../trafic_data/data_1.yaml"):
Purpose: Specifies the dataset configuration (paths, 21 classes).

Optimization: Ensure your YAML is correct (as provided):
yaml

path: /Users/furka/PycharmProjects/Autonome-Fahrzeugszenen-Segmentierung/trafic_data
train: train/images
val: valid/images
nc: 21
names: ['ambulance', 'army vehicle', 'auto rickshaw', 'bicycle', 'bus', 'car', 'garbagevan', 'human hauler', 'minibus', 'minivan', 'motorbike', 'pickup', 'policecar', 'rickshaw', 'scooter', 'suv', 'taxi', 'three wheelers -CNG-', 'truck', 'van', 'wheelbarrow']

Recommendation: Verify paths exist and labels are in YOLO format (<class_id> <x_center> <y_center> <width> <height>).

epochs (epochs=50):
Purpose: Number of passes through the dataset.

Optimization: 50 epochs is reasonable for ~10,000 images but may be excessive if the model converges earlier.

Recommendation: Set epochs=50 with patience=20 (early stopping if no improvement for 20 epochs) to save time (~2–2.5 hours on RTX 2060).

imgsz (imgsz=640):
Purpose: Input image size (640x640 pixels).

Optimization: 640 is ideal for traffic detection, balancing accuracy (detects small objects like bicycles) and VRAM usage (~3–4GB on RTX 2060).

Recommendation: Keep imgsz=640. If VRAM errors occur, reduce to imgsz=416 (~1.5–2 hours, slightly lower accuracy).

batch (batch=16):
Purpose: Number of images per iteration.

Optimization: batch=16 fits within 6GB VRAM for yolov8n.pt with imgsz=640 and mixed precision (AMP). Larger batches (e.g., 32) may cause out-of-memory errors.

Recommendation: Use batch=16. If VRAM issues arise, reduce to batch=8 (~3 hours).

device (device=0):
Purpose: Specifies GPU (0 for RTX 2060).

Optimization: Explicitly set to ensure GPU usage once CUDA is enabled.

Recommendation: Use device=0.

workers (workers=8):
Purpose: Number of CPU threads for data loading.

Optimization: Your Ryzen 7 2700X (8 cores, 16 threads) handles workers=8 well, maximizing data loading without CPU bottlenecks.

Recommendation: Set workers=8. If you notice high CPU usage but low GPU utilization, reduce to workers=4.

Optimization Parameters
optimizer (optimizer="auto"):
Purpose: Selects the optimizer (SGD, Adam, AdamW).

Optimization: "auto" chooses AdamW for yolov8n.pt, which is fast and stable for small models.

Recommendation: Use optimizer="auto".

lr0 (lr0=0.001):
Purpose: Initial learning rate.

Optimization: Default (0.001) works well for yolov8n.pt with 21 classes. Lower values (e.g., 0.0005) may stabilize training for small datasets (<1,000 images).

Recommendation: Use lr0=0.001. If loss spikes, try lr0=0.0005.

patience (patience=20):
Purpose: Stops training if validation metrics (e.g., mAP) don’t improve for 20 epochs.

Optimization: Prevents overfitting and saves time (~30–40% reduction if convergence occurs early).

Recommendation: Set patience=20.

amp (amp=True):
Purpose: Enables Automatic Mixed Precision for faster training and lower VRAM usage.

Optimization: Critical for RTX 2060 to reduce memory footprint (30% less VRAM) and speed up training (20% faster).

Recommendation: Use amp=True (default).

Data Augmentation Parameters
Augmentations enhance robustness for traffic scenes (e.g., varying lighting, angles).
hsv_h, hsv_s, hsv_v (hsv_h=0.015, hsv_s=0.7, hsv_v=0.4):
Purpose: Adjusts hue, saturation, value for color jittering.

Optimization: Defaults are ideal for traffic datasets, handling lighting variations (e.g., day/night).

Recommendation: Keep defaults.

degrees (degrees=5.0):
Purpose: Random rotation angle.

Optimization: Slight rotations (±5°) mimic vehicle orientations without distorting context.

Recommendation: Set degrees=5.0 (default is 0.0).

translate (translate=0.1):
Purpose: Random image shifts.

Optimization: Default is good for small object displacement (e.g., bicycles, pedestrians).

Recommendation: Keep translate=0.1.

scale (scale=0.5):
Purpose: Random zooming.

Optimization: Default enhances robustness to object size variations.

Recommendation: Keep scale=0.5.

fliplr (fliplr=0.5):
Purpose: Probability of horizontal flipping.

Optimization: Ideal for traffic scenes, as vehicles can appear from either side.

Recommendation: Keep fliplr=0.5.

mosaic (mosaic=1.0):
Purpose: Combines 4 images into one for complex scenes.

Optimization: Boosts performance in crowded traffic scenes (e.g., intersections).

Recommendation: Use mosaic=1.0. Disable (mosaic=0.0) for the last 10 epochs if overfitting occurs.

Additional Parameters
project (project="runs/train"):
Purpose: Directory for training outputs.

Recommendation: Keep default or set project="traffic_training".

name (name="traffic_yolov8n"):
Purpose: Run name for organization.

Recommendation: Use name="traffic_yolov8n" for clarity.

save (save=True):
Purpose: Saves checkpoints (best.pt, last.pt).

Recommendation: Keep save=True.

plots (plots=True):
Purpose: Generates training plots (loss, mAP).

Recommendation: Keep plots=True for analysis.



# Autonome-Fahrzeugszenen-Segmentierung

Train data set size: 2704

Val data set size: 300

https://www.kaggle.com/datasets/ashfakyeafi/road-vehicle-images-dataset

## 1. Problemverständnis (Business Understanding)
**Ziel:** Klärung des Projekthintergrunds und der Motivation.

- **Forschungsfrage / Zielsetzung:** Was soll erreicht werden?  
  *Beispiel:* „Segmentierung von Fahrzeugszenen in relevante Objekte (Fahrzeuge, Fußgänger, Fahrbahnmarkierungen)“

- **Kontext:** Warum ist das Problem interessant? Gibt es reale Analogien?  
  *Beispiel:* Sicherheit autonomer Fahrzeuge, Reaktion auf komplexe Umgebungen.

- **Relevanz des neuronalen Netzes:** Warum erscheint ein neuronales Netz als geeigneter Ansatz?  
  *Beispiel:* Komplexe visuelle Muster, hohe Generalisierungsfähigkeit.

## 2. Datenverständnis (Data Understanding)
**Ziel:** Erste Exploration und Beschreibung der Daten.

- **Datenquelle(n):** Herkunft der Daten (z. B. Cityscapes, KITTI, eigene Aufnahmen)
- **Variablenbeschreibung:** Bilddaten, Klassenmasken, ggf. Zusatzinformationen (z. B. Lichtverhältnisse)
- **Erste Statistiken & Visualisierungen:** Klassenverteilung, Beispielbilder, Pixelverteilung
- **Zielgröße(n):** Segmentierungsmasken pro Bild

## 3. Datenvorbereitung (Data Preparation)
**Ziel:** Bereinigung und Transformation der Daten zur Modellierung.

- **Bereinigung:** Ausschluss unvollständiger Datensätze, Formatvereinheitlichung
- **Feature Engineering:** Farbraumtransformationen, eventuell Zusatzkanäle (z. B. Kanten, Tiefenbilder)
- **Train-Test-Split:** Aufteilung mit ggf. zusätzlichem Validierungsset
- **Normierung/Skalierung:** Pixelnormalisierung (z. B. 0–1 oder -1 bis 1)

## 4. Modellierung (Modeling)
**Ziel:** Aufbau und Training des neuronalen Netzes.

- **Architektur des Netzwerks:** z. B. U-Net, SegNet, DeepLab
- **Hyperparameter:** Lernrate, Batchgröße, Anzahl Epochen, Optimierungsverfahren
- **Trainingsprozess:** Trainingsverlauf (Loss-Kurve), Early Stopping, Augmentation
- **Vergleich mit Baseline:** z. B. einfache Schwellenwertverfahren, Random Classifier

## 5. Evaluierung (Evaluation)
**Ziel:** Bewertung der Modellgüte auf Testdaten.

- **Metriken:** Intersection over Union (IoU), Pixel Accuracy, Mean Accuracy
- **Visualisierungen:** Beispielhafte Vorher-Nachher-Darstellungen, Fehlerkarten
- **Interpretation:** Welche Klassen werden schlecht erkannt? Wo treten Fehler auf?
- **Vergleich mit Zielsetzung:** Entspricht das Ergebnis der ursprünglichen Fragestellung?

## ❌ 6. Deployment (entfällt)

---

## ✍️ Zusätzliche Hinweise für die Bewertung

- **Reflexion:** Wo lagen Schwierigkeiten? Was hätte man anders machen können?
- **Reproduzierbarkeit:** Code und Ergebnisse nachvollziehbar dokumentiert?
- **Saubere Argumentation:** Begründung der methodischen Entscheidungen?
