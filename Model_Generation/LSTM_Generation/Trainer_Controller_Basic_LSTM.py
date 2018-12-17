import os

DATASET_DIRECTORY_PATH = "/home/eric/Desktop/LyreBird/Datasets"
DATASETS_PATHS = [x[0] for x in os.walk(DATASET_DIRECTORY_PATH)]
DATASETS_PATHS.pop(0)

i = 0
for dataset_pth in DATASETS_PATHS:
    i += 1
    if i <= 4:
        os.system('python Basic_LSTM_Model_Trainer.py' + ' '
                  + dataset_pth + ' '
                  + '/home/eric/Desktop/LyreBird/Main_Production/Models/LSTM_Models')