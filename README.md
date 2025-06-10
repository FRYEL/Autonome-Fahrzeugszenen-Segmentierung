# Autonome-Fahrzeugszenen-Segmentierung

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
