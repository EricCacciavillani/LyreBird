import os

__PROJECT_ABSOLUTE_PATH = ''.join(os.getcwd().partition('LyreBird')[0:2])


def enum(**enums):
    return type('Enum', (), enums)


ABS_PATHS = enum(PROJECT_ABSOLUTE_PATH=__PROJECT_ABSOLUTE_PATH,
                 TRAINING_DATASET_DIRECTORY_PATH=__PROJECT_ABSOLUTE_PATH + "/Datasets/",
                 DATA_DUMP_PATH=__PROJECT_ABSOLUTE_PATH + "/Data_Dump/",
                 MODELS_PATH=__PROJECT_ABSOLUTE_PATH + "/Data_Dump/Models/",
                 SHELVES_PATH="/Data_Dump/Shelves")

# Set path directory
MIDI_CONSTANTS = enum(INPUT_SEQUENCE_LEN=50,
                      TRACK_INDEX=0)

# Init 'const' variables to be shared
SECONDS_TO_COOL_GPU = 150
