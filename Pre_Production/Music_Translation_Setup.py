from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib

matplotlib.use('Agg')

import sys
sys.path.append('..')

from Pre_Production.Music_Translation import *
from Pre_Production.Midi_Pre_Processor import *


pre_processor_obj = None
pre_processor_shelve = shelve.open(ABS_PATHS.SHELVES_PATH
                                   + SHELVE_NAMES.PRE_PROCESSOR)

# Check to see if the object already exists
if "pre_processor" in pre_processor_shelve.keys():
    print("Found stored pre processor!")
    pre_processor_obj = pre_processor_shelve["pre_processor"]

# Pre-processor not found generate pre-processor
else:

    print("Generating pre processor!")
    pre_processor_obj = MidiPreProcessor(
        ABS_PATHS.TRAINING_DATASET_DIRECTORY_PATH)

    pre_processor_shelve["pre_processor"] = pre_processor_obj

pre_processor_shelve.close()

instrument_name_contains = {"Guitar": False,
                            "Piano": False,
                            "Brass": False,
                            "Drums": True}


all_instruments = pre_processor_obj.return_all_instruments()
instr_note_pairs_dict = pre_processor_obj.return_instr_note_pairs_dict()


print("Synthesizing wanted instr/note pairs...")
instr_wave_forms = get_instr_wave_forms(instrument_name_contains=instrument_name_contains,
                                        all_instruments=all_instruments,
                                        instr_note_pairs_dict=instr_note_pairs_dict,
                                        unique_matrix=True,
                                        normalize=True,
                                        remove_rows_with_files=True)

for instr, waves in instr_wave_forms.items():
    print("instr:{0} Matrix_Shape: {1}".format(instr, waves.shape))


lyre_bird_model = MusicTranslationModelGenerator(instr_wave_forms=instr_wave_forms,
                                                 twilo_account=True)

lyre_bird_model.create_model(load_model_path=None,
                             training_sessions=15,
                             steps_per_training_session=27000)
