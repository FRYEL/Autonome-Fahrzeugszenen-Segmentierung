# Import Libraries


```python
from ultralytics import YOLO
import torch
```

# Training of initial YOLO11n Baseline Model


```python
model = YOLO("../yolo11n.pt")

# Training with default settings as a fair baseline
model.train(
    data="../traffic_data/data_1.yaml",
    epochs=40,
    imgsz=640,
    batch=16,
    device=0,
    project="traffic_training",
    name="yolo11n_baseline",
    save=True,
    plots=True
)
```

# Preprocessing the Dataset

## Remapping Classes

Classes that occur very rarely but are similar to the car class, like taxi, suv and polic car are mapped to the car class.


```python
import os

label_dirs = ["../traffic_data/train/labels", "../traffic_data/valid/labels"]
remap = {12: 5, 15: 5, 16: 5}  # old → new class indices

for label_dir in label_dirs:
    for file in os.listdir(label_dir):
        if not file.endswith(".txt"):
            continue

        path = os.path.join(label_dir, file)
        new_lines = []

        with open(path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                cls = int(parts[0])
                if cls in remap:
                    cls = remap[cls]
                new_line = " ".join([str(cls)] + parts[1:])
                new_lines.append(new_line)


        with open(path, "w") as f:
            f.write("\n".join(new_lines))

```

## Shifting Classes > 12 due to removed classes


```python
# Classes that were removed (already mapped to class 5)
removed_classes = [12, 15, 16]

def final_shift(cls):
    # Skip if already merged to 5
    if cls == 5:
        return 5
    # Shift down if above removed ones
    shift = sum(1 for r in removed_classes if cls > r)
    return cls - shift

for label_dir in label_dirs:
    for file in os.listdir(label_dir):
        if not file.endswith(".txt"):
            continue
        path = os.path.join(label_dir, file)
        new_lines = []
        with open(path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                cls = int(parts[0])
                new_cls = final_shift(cls)
                new_line = " ".join([str(new_cls)] + parts[1:])
                new_lines.append(new_line)
        with open(path, "w") as f:
            f.write("\n".join(new_lines))

```

## Data Augmentation

All images containing classes 0, 1, 6, 7,8, 13 and 17 will be oversampled by copying them to a new directory. This is done to balance the dataset and improve model performance on these classes. The copied images will have the suffix "_copy" added to their filenames, and the corresponding label files will also be copied with the same suffix. Then the copied images and labels will be added to the training data.

```bash
for label in labels/*.txt; do
    if grep -E '^(0|1|6|7|8|13|17)[[:space:]]' "$label"; then
        base=$(basename "$label" .txt)
        cp "images/$base.jpg" oversampled/images/"$base"_copy.jpg
        cp "$label" oversampled/labels/"$base"_copy.txt
    fi
done
```

# Hyperparameter Tuning and Evolution Selection with YOLO11n tune() function

Further training after using the .tune() function in Ultralytics YOLO is not needed because each tuning iteration already performs a full training run using a distinct set of hyperparameters. The best model is selected based on its validation performance (e.g., mAP@50) across all iterations. This model, saved as best.pt,has already undergone full optimization and training under the best-found configuration. Re-training it would not improve performance unless new data, loss functions, or architectures are introduced. Therefore, the best.pt file from .tune() is directly usable for inference or evaluation without additional training.


```python
model = YOLO("../yolo11n.pt")

# hyperparameter evolution
model.tune(
    data="../traffic_data/data_1.yaml",
    epochs=40,
    iterations=10,
    device=0,
    imgsz=640,
    batch=16,
    optimizer="auto",
    project="traffic_training",
    name="yolo11n_MAX",
    cache=True
)
```

## 🔧 Hyperparameter Tuning Results

 **10/10 iterations complete** (⏱️ 5806.02 seconds)  <br>
Best fitness: `0.32668` (observed at iteration 3)   <br>
Best fitness model: `traffic_training/train37`  <br>

### Best Fitness Metrics:
- **Precision (B):** 0.64461
- **Recall (B):** 0.41534
- **mAP@50 (B):** 0.54009
- **mAP@50–95 (B):** 0.30297
- **Box Loss:** 1.37739
- **Class Loss:** 0.99681
- **DFL Loss:** 1.15922
- **Fitness Score:** 0.32668

---

### Best Hyperparameters:

| Parameter           | Value     |
|---------------------|-----------|
| `lr0`               | 0.00955   |
| `lrf`               | 0.00998   |
| `momentum`          | 0.97997   |
| `weight_decay`      | 0.00054   |
| `warmup_epochs`     | 2.54319   |
| `warmup_momentum`   | 0.94159   |
| `box`               | 7.99327   |
| `cls`               | 0.42895   |
| `dfl`               | 1.61663   |
| `hsv_h`             | 0.01156   |
| `hsv_s`             | 0.75561   |
| `hsv_v`             | 0.31895   |
| `degrees`           | 0.0       |
| `translate`         | 0.10405   |
| `scale`             | 0.4834    |
| `shear`             | 0.0       |
| `perspective`       | 0.0       |
| `flipud`            | 0.0       |
| `fliplr`            | 0.40892   |
| `bgr`               | 0.0       |
| `mosaic`            | 1.0       |
| `mixup`             | 0.0       |
| `cutmix`            | 0.0       |
| `copy_paste`        | 0.0       |



```python

```

# Visual Comparison on Validation Set


```python
# Best Model trained by .tune() function
model = YOLO("../traffic_training/best_tuning/train37/weights/best.pt")
results = model.predict(source="../../traffic_data/valid/images/Pias--289-_jpg.rf.a6cb4e0f81f99b7c803bd9c9832163da.jpg", conf=0.25, iou=0.45)
for result in results:
    result.show()
    result.save()
```


```python
# Basic YOLO11n Baseline Model trained without .tune() function
model2 = YOLO("../traffic_training/yolo11n_baseline/weights/best.pt")
results2 = model2.predict(source="../../traffic_data/valid/images/Pias--289-_jpg.rf.a6cb4e0f81f99b7c803bd9c9832163da.jpg", conf=0.25, iou=0.45)
for result in results2:
    result.show()
    result.save()
```

# Visualizations


```python
import os
import pandas as pd
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
```


```python

# dir paths
train_label_dir = "../../traffic_data/train/labels"
val_label_dir = "../../traffic_data/valid/labels"
train_image_dir = "../../traffic_data/train/images"
output_dir = "../output_plots"

# custom colour palette
custom_palette = [
    "#032a4d", "#00457a", "#0458a5", "#0476df", "#50b1ff", "#88cbff", "#9e9e9e"
]

# clsses
class_names = [
'ambulance', 'army vehicle', 'auto rickshaw', 'bicycle', 'bus', 'car', 'garbagevan', 'human hauler', 'minibus', 'minivan', 'motorbike', 'pickup', 'policecar', 'rickshaw', 'scooter', 'suv', 'taxi', 'three wheelers -CNG-', 'truck', 'van', 'wheelbarrow'
]
```


```python
# Function to process label files and count classes
def process_labels(label_dir):
    class_counts = defaultdict(int)
    objects_per_image = []
    bbox_sizes = []

    for file in os.listdir(label_dir):
        if file.endswith(".txt"):
            with open(os.path.join(label_dir, file), "r") as f:
                lines = f.readlines()
                objects_per_image.append(len(lines))
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id = int(parts[0])
                        width = float(parts[3])
                        height = float(parts[4])
                        class_counts[class_names[class_id]] += 1
                        bbox_sizes.append(width * height)
    return class_counts, objects_per_image, bbox_sizes

```


```python
# Process labels
train_counts, train_objs, train_bbox = process_labels(train_label_dir)
val_counts, val_objs, val_bbox = process_labels(val_label_dir)

total_counts = train_counts.copy()
for k, v in val_counts.items():
    total_counts[k] += v
all_objects = train_objs + val_objs
```


```python
sns.set_theme(style="whitegrid")
os.makedirs(output_dir, exist_ok=True)
```


```python
df_counts = pd.DataFrame(total_counts.items(), columns=["Class", "Count"]).sort_values("Count", ascending=False)
```


```python

# Bar plot of distributiion
plt.figure(figsize=(12, 6))
sns.barplot(data=df_counts, x="Class", y="Count", palette=custom_palette * (len(df_counts) // len(custom_palette) + 1))
plt.xticks(rotation=45)
plt.title("Class Distribution (Train + Val)")
plt.tight_layout()

plt.savefig(os.path.join(output_dir, "SMALL_class_distribution.png"))
plt.close()
```


```python

# Pie chart distribution
top5 = df_counts.head(5)
plt.figure(figsize=(6, 6))
plt.pie(top5["Count"], labels=top5["Class"], colors=custom_palette[:5], autopct='%1.1f%%', startangle=140)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "top5_class_pie.png"))
plt.close()

```


```python
# Heatmap of bounding boxes

image_width = 640
image_height = 360
output_path = "./bbox_heatmap.png"

# extra palette for heatmap
custom_palette = [
    "#032a4d", "#00457a", "#0458a5", "#0476df", "#50b1ff", "#88cbff", "#9e9e9e"
]
custom_cmap = LinearSegmentedColormap.from_list("custom_blues", custom_palette)

# bbxos centers

x_centers = []
y_centers = []

for fname in os.listdir(train_label_dir):
    if not fname.endswith(".txt"):
        continue
    with open(os.path.join(train_label_dir, fname), "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            _, x, y, _, _ = map(float, parts)
            x_centers.append(int(x * image_width))
            y_centers.append(int(y * image_height))

# heatmap
heatmap, xedges, yedges = np.histogram2d(
    x_centers, y_centers,
    bins=(64, 36),
    range=[[0, image_width], [0, image_height]]
)

# Plot
plt.figure(figsize=(3.4, 2.5))  # Single-column LaTeX size
sns.set_style("white")
sns.heatmap(
    heatmap.T,
    cmap=custom_cmap,
    cbar=True,
    xticklabels=False,
    yticklabels=False
)

plt.xlabel("Image Width (px)", fontsize=8)
plt.ylabel("Image Height (px)", fontsize=8)
plt.tight_layout()
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.close()

```
