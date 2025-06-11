Colors:

darkest blue #032a4d
med dark blue #00457a
medium blue #0458a5
base blue #0476df
light blue #50b1ff
lightest blue #88cbff
grey #9e9e9e

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
