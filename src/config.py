import os

# Root directory of the project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Path to the processed dataset
DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed', 'HR_Sailfort_dataset_Processed.xls')

# Path to directory where output figures will be saved
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs', 'figures')

# Path to directory where trained models will be saved
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Model Configurations
RANDOM_STATE = 42
TEST_SIZE = 0.20
CV_SPLITS = 5
TARGET_COLUMN = 'left'

# Visual Configurations
STATUS_PALETTE = {1: "#ff6e54", 0: "#4e79a7"}
