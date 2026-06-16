import kagglehub

path = kagglehub.dataset_download(
    "prahladmehandiratta/cervical-cancer-largest-dataset-sipakmed"
)

print("Path to dataset files:", path)