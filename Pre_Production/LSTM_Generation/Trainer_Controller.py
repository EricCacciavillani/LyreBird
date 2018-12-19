import os
import sys

import shelve

# Import shared files
sys.path.insert(0, '/home/eric/Desktop/LyreBird/Shared_Files')
from Constants import *

DATASET_DIRECTORY_PATH = "/home/eric/Desktop/LyreBird/Datasets"
DATASETS_PATHS = [x[0] for x in os.walk(DATASET_DIRECTORY_PATH)]
DATASETS_PATHS.pop(0)

i = 0
for dataset_pth in DATASETS_PATHS:
    i += 1
    if i <= 2:
        os.system('python ' + CATEGORICAL_MODEL_GENERATION
                  + 'Categorical_Notes_Model_Train.py'
                  + ' ' + dataset_pth)