import os

__PROJECT_ABSOLUTE_PATH = ''.join(os.getcwd().partition('LyreBird')[0:2])


def enum(**enums):
    return type('Enum', (), enums)


ABS_PATHS = enum(PROJECT_ABSOLUTE_PATH=__PROJECT_ABSOLUTE_PATH,
                 DATA_DUMP_PATH=__PROJECT_ABSOLUTE_PATH + "/Data_Dump/",
                 MATRIX_ROW_CLEANING=__PROJECT_ABSOLUTE_PATH + "/Data_Dump/Matrix_Row_Cleaning/",
                 TRAINING_DATASET_DIRECTORY_PATH=__PROJECT_ABSOLUTE_PATH + "/Data_Dump/Datasets/",
                 SAVED_MODELS_PATH=__PROJECT_ABSOLUTE_PATH + "/Data_Dump/Saved_Models/",
                 SAVED_MODELS_PATH_LYREBIRD_TN=__PROJECT_ABSOLUTE_PATH + "/Data_Dump/Saved_Models/LyreBird_TN",
                 SAVED_MODELS_PATH_LYREBIRD_TN_BEST=__PROJECT_ABSOLUTE_PATH + "/Data_Dump/Saved_Models/LyreBird_TN_Best",
                 SAVED_WEIGHTS_PATH=__PROJECT_ABSOLUTE_PATH + "/Data_Dump/Saved_Weights/",
                 SHELVES_PATH=__PROJECT_ABSOLUTE_PATH + "/Data_Dump/Shelves/",
                 AUDIO_CACHE_PATH=__PROJECT_ABSOLUTE_PATH + "/Data_Dump/Audio_Cache/")

SHELVE_NAMES = enum(PRE_PROCESSOR="Midi_Pre_Processor",
                    WAVE_FORMS="Wave_Forms",
                    MODELS_HISTORY="Models_History",
                    MODEL_EXTRAS="Model_Extras")


MIDI_CONSTANTS = enum(SMALL_FILE_CHECK=0)

HARDWARE_RELATED_CONSTANTS = enum(SECONDS_TO_COOL_GPU=150,
                                  THREAD_POOL_AMOUNT=2000)


FLUID_SYNTH_CONSTANTS = enum(SAMPLING_RATE=8000)

INSTRUMENT_NOTE_SPLITTER = enum(STR="-:-")
PARAMETER_VAL_SPLITTER = enum(STR=":")

DEFAULT_NOTE_CONSTANTS = enum(OFFSET=.2499,
                              VELOCITY=100)

UNIVERSAL_MUSIC_TRANSLATOR = enum(MU=256,
                                  LATENT_DIM=64,
                                  POOL_SIZE=400,
                                  BATCH_SIZE=3)


# VELOCITY_CONSTANTS = enum(LIST=[60, 80, 100])