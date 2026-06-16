IMG_SIZE = (224, 224)
BATCH_SIZE = 16
SEED = 42

# SIPaKMeD five-class labels
CLASS_NAMES = [
    "Dyskeratotic",
    "Koilocytotic",
    "Metaplastic",
    "Parabasal",
    "Superficial-Intermediate",
]

# Binary detection mapping: abnormal/cancer-suspicious vs normal/benign cell categories
BINARY_MAP = {
    "Dyskeratotic": "Abnormal",
    "Koilocytotic": "Abnormal",
    "Metaplastic": "Normal",
    "Parabasal": "Normal",
    "Superficial-Intermediate": "Normal",
    "im_Dyskeratotic": "Abnormal",
    "im_Koilocytotic": "Abnormal",
    "im_Metaplastic": "Normal",
    "im_Parabasal": "Normal",
    "im_Superficial-Intermediate": "Normal",
}
