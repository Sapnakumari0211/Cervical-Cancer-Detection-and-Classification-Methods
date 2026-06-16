# Cervical Cancer Detection and Classification Techniques

This project is a modified and focused version of the cervical cancer image-classification codebase. It directly matches the title **Cervical Cancer Detection and Classification Techniques** by implementing two tasks:

1. **Detection:** classify a Pap smear/cervical cell image as **Normal** or **Abnormal**.
2. **Classification:** classify the image into five SIPaKMeD cervical cell categories.

## Dataset
Use the SIPaKMeD Pap smear dataset with the following raw folder structure:

```text
raw_sipakmed/
├── im_Dyskeratotic/
├── im_Koilocytotic/
├── im_Metaplastic/
├── im_Parabasal/
└── im_Superficial-Intermediate/
```

The five-class classification labels are:

- Dyskeratotic
- Koilocytotic
- Metaplastic
- Parabasal
- Superficial-Intermediate

For the detection task, the mapping used is:

| Classification Class | Detection Label |
|---|---|
| Dyskeratotic | Abnormal |
| Koilocytotic | Abnormal |
| Metaplastic | Normal |
| Parabasal | Normal |
| Superficial-Intermediate | Normal |

## Project Structure

```text
Cervical_Cancer_Detection_Classification_Techniques/
├── app.py
├── requirements.txt
├── README.md
├── src/
│   ├── config.py
│   ├── prepare_dataset.py
│   ├── model_utils.py
│   ├── train_detection.py
│   ├── train_classification.py
│   ├── predict.py
│   └── gradcam.py
├── models/
├── results/
└── sample_images/
```

## Step 1: Install Requirements

```bash
pip install -r requirements.txt
```

## Step 2: Prepare Dataset

```bash
cd src
python prepare_dataset.py --raw_dir "D:/dataset/raw_sipakmed" --output_dir "D:/dataset/processed_sipakmed"
```

This creates:

```text
processed_sipakmed/
├── detection/train|val|test/Normal|Abnormal
└── classification/train|val|test/<five classes>
```

## Step 3: Train Detection Model

```bash
cd src
python train_detection.py --data_dir "D:/dataset/processed_sipakmed" --epochs 20 --model_name resnet50
```

## Step 4: Train Classification Model

```bash
cd src
python train_classification.py --data_dir "D:/dataset/processed_sipakmed" --epochs 20 --model_name resnet50
```

## Step 5: Predict One Image

```bash
cd src
python predict.py --image "D:/dataset/sample.bmp" --detection_model "../models/detection_model_final.h5" --classification_model "../models/classification_model_final.h5"
```

Expected output:

```json
{
  "detection_result": "Abnormal",
  "detection_confidence": 0.94,
  "classification_result": "Dyskeratotic",
  "classification_confidence": 0.91
}
```

## Step 6: Run Web App

```bash
streamlit run app.py
```

## Research Methodology

The project follows this workflow:

```text
Pap Smear Image
↓
Preprocessing and Data Augmentation
↓
Binary Detection Model: Normal / Abnormal
↓
Multiclass Classification Model: 5 Cervical Cell Classes
↓
Evaluation: Accuracy, Precision, Recall, F1-score, Confusion Matrix
↓
Grad-CAM Explainability
```

## Internship Report Title

Use the same title as the certificate:

**Cervical Cancer Detection and Classification Techniques**

## Suggested Objectives

- To develop an automated cervical cancer screening system using Pap smear images.
- To detect whether a cervical cell image is normal or abnormal.
- To classify cervical cell images into five categories using transfer learning.
- To evaluate the proposed system using accuracy, precision, recall, F1-score, and confusion matrix.
- To provide explainable visualization using Grad-CAM.

## Models Used

- ResNet50 transfer learning
- EfficientNetB0 optional model
- Softmax classifier
- Grad-CAM visualization

## Important Note

This project does not claim medical diagnosis. It is an academic research prototype for automated image-based cervical cancer detection and classification.
