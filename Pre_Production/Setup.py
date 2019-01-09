import shelve
import sys
sys.path.append('..')

from Pre_Production.Midi_Pre_Processor import *
from Pre_Production.Model_Generator import *
from Shared_Files.Music_Pallete import *


pre_processor_shelve = shelve.open(ABS_PATHS.SHELVES_PATH
                                   + SHELVE_NAMES.PRE_PROCESSOR)

pre_processor_obj = pre_processor_shelve["pre_processor"]

MusicPallete(pre_processor_obj)



# create_pretty_midi_object(genre_file_dict["Funk_Music"]["/home/eric/Desktop/LyreBird/Datasets/Funk_Music_Midi/sofunky.mid"]["flat_instr_note_seq"])













# pre_processor_obj = None
#
# pre_processor_shelve = shelve.open(ABS_PATHS.SHELVES_PATH
#                                    + SHELVE_NAMES.PRE_PROCESSOR)

#
# if "pre_processor" not in pre_processor_shelve.keys():
#     pre_processor_obj = MidiPreProcessor(
#         ABS_PATHS.TRAINING_DATASET_DIRECTORY_PATH)
#
#     pre_processor_shelve["pre_processor"] = pre_processor_obj
#
# else:
#     pre_processor_obj = pre_processor_shelve["pre_processor"]
#
# pre_processor_shelve.close()
#
# model_generator = ModelGenerator(pre_processor_obj=pre_processor_obj,
#                                  twilo_account=True)
#
# model_generator.train_category_rnn_model()
#
#
