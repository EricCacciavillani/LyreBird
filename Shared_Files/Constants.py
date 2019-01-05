import os

__PROJECT_ABSOLUTE_PATH = ''.join(os.getcwd().partition('LyreBird')[0:2])


def enum(**enums):
    return type('Enum', (), enums)


ABS_PATHS = enum(PROJECT_ABSOLUTE_PATH=__PROJECT_ABSOLUTE_PATH,
                 TRAINING_DATASET_DIRECTORY_PATH=__PROJECT_ABSOLUTE_PATH + "/Datasets/",
                 DATA_DUMP_PATH=__PROJECT_ABSOLUTE_PATH + "/Data_Dump/",
                 SAVED_MODELS_PATH=__PROJECT_ABSOLUTE_PATH + "/Data_Dump/Saved_Models/",
                 SAVED_WEIGHTS_PATH=__PROJECT_ABSOLUTE_PATH + "/Data_Dump/Saved_Weights/",
                 SHELVES_PATH=__PROJECT_ABSOLUTE_PATH + "/Data_Dump/Shelves/")
SHELVE_NAMES = enum(PRE_PROCESSOR="Midi_Pre_Processor",
                    MODELS_HISTORY="Models_History")

MIDI_CONSTANTS = enum(FLAT_INPUT_SEQUENCE_LEN=50)

HARDWARE_RELATED_CONSTANTS = enum(SECONDS_TO_COOL_GPU=150,
                                  THREAD_POOL_AMOUNT=2000)

INSTRUMENT_NOTE_SPLITTER = enum(STR="-:-")

DEFAULT_NOTE_CONSTANTS = enum(OFFSET=.43,
                              VELOCITY=100)

VELOCITY_CONSTANTS = enum(LIST=[60, 80, 100])

# Init 'const' variables to be shared


# from keras.models import load_model
#
# rnn_model = load_model(filepath,
#                        custom_objects={
#                            'focal_loss_fixed': focal_loss(alpha=.25, gamma=2)})