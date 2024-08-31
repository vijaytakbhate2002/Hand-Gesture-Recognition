import pathlib
import os

# REQUIRED PATHS
ROOT_PATH = pathlib.Path('\\'.join(__file__.split('\\')[:-1])).resolve().parent
DATASETS_PATH = os.path.join(ROOT_PATH, 'Data')
TRAINED_MODELS_PATH = os.path.join(ROOT_PATH, 'Trained Models')
