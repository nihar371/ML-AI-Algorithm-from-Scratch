import os
import zipfile
import pandas as pd

DATA_DIR = "./data"
ZIP_FILE = os.path.join(DATA_DIR, "wine+quality.zip")
EXTRACTED_DIR = os.path.join(DATA_DIR, "winequality")

# Extract files if not already extracted
if not os.path.exists(EXTRACTED_DIR):
    os.makedirs(EXTRACTED_DIR, exist_ok=True)
    with zipfile.ZipFile(ZIP_FILE, 'r') as zip_ref:
        zip_ref.extractall(EXTRACTED_DIR)

def data_load_decorator(func):
    def wrapper(*args, **kwargs):
        print("==================== Fetching the Data ====================")
        result = func(*args, **kwargs)
        return result
    return wrapper

@data_load_decorator
class WineRegressionData:
    """
    Prepares regression data using winequality-red.csv
    """
    def __init__(self):
        red_path = os.path.join(EXTRACTED_DIR, "winequality-red.csv")
        self.data = pd.read_csv(red_path, sep=';')
        self.X = self.data.drop('quality', axis=1)
        self.y = self.data[['quality']]

    def get_features_targets(self):
        return self.X.copy(), self.y.copy()

@data_load_decorator
class WineClassificationData:
    """
    Prepares classification data by combining red and white wine datasets,
    adding a 'wine_type' column to distinguish between them.
    """
    def __init__(self):
        red_path = os.path.join(EXTRACTED_DIR, "winequality-red.csv")
        white_path = os.path.join(EXTRACTED_DIR, "winequality-white.csv")
        red = pd.read_csv(red_path, sep=';')
        white = pd.read_csv(white_path, sep=';')
        red['wine_type'] = 'red'
        white['wine_type'] = 'white'
        self.data = pd.concat([red, white], ignore_index=True)
        self.X = self.data.drop('wine_type', axis=1)
        self.y = self.data[['wine_type']]

    def get_features_targets(self):
        return self.X.copy(), self.y.copy()