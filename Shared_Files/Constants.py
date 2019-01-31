import os

__PROJECT_ABSOLUTE_PATH = ''.join(os.getcwd().partition('LyreBird')[0:2])


def enum(**enums):
    return type('Enum', (), enums)


ABS_PATHS = enum(PROJECT_ABSOLUTE_PATH=__PROJECT_ABSOLUTE_PATH,
                 DATA_DUMP_PATH=__PROJECT_ABSOLUTE_PATH + "/Data_Dump/",
                 TRAINING_DATASET_DIRECTORY_PATH=__PROJECT_ABSOLUTE_PATH + "/Data_Dump/Datasets/",
                 SAVED_MODELS_PATH=__PROJECT_ABSOLUTE_PATH + "/Data_Dump/Saved_Models/",
                 SAVED_WEIGHTS_PATH=__PROJECT_ABSOLUTE_PATH + "/Data_Dump/Saved_Weights/",
                 SHELVES_PATH=__PROJECT_ABSOLUTE_PATH + "/Data_Dump/Shelves/")

SHELVE_NAMES = enum(PRE_PROCESSOR="Midi_Pre_Processor",
                    WAVE_FORMS="Wave_Forms",
                    MODELS_HISTORY="Models_History")

MIDI_CONSTANTS = enum(SMALL_FILE_CHECK=0)

HARDWARE_RELATED_CONSTANTS = enum(SECONDS_TO_COOL_GPU=150,
                                  THREAD_POOL_AMOUNT=2000)


FLUID_SYNTH_CONSTANTS = enum(SAMPLING_RATE=44100)

INSTRUMENT_NOTE_SPLITTER = enum(STR="-:-")
PARAMETER_VAL_SPLITTER = enum(STR=":")

DEFAULT_NOTE_CONSTANTS = enum(OFFSET=.43,
                              VELOCITY=100)

UNIVERSAL_MUSIC_TRANSLATOR = enum(STEPS_PER_EPOCH=10000,
                                  MU=256,
                                  LATENT_DIM=64,
                                  POOL_SIZE=400,
                                  BATCH_SIZE=3)


# VELOCITY_CONSTANTS = enum(LIST=[60, 80, 100])