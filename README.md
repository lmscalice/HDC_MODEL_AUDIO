# HD_ISOLET
HD_ISOLET: Speech recognition using Hyperdimensional Computing

## Instructions

### Dataset
 - https://www.dropbox.com/s/4jegzwp7p4epdkd/isolet.pkl?dl=0

### Background
 - https://archive.ics.uci.edu/ml/datasets/isolet

### Installing prerequisites.
 - pip install -r requirement.txt

### Run the application.
 - python HDC_model.py --app_ ./spoken-digit-dataset/recordings/ --iter_ 20 --dimension_ 10000

 - Different hyperparameters
    - 1.app_: Training and testing dataset
    -  2.iter_: Number of epochs for retraining
    -   3.dimension_: Dimension of hyper-vector using in HDC model
