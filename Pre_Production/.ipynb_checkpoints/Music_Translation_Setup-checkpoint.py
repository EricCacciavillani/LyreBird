from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import shelve
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import librosa
from tqdm import tqdm
import pretty_midi

from os import listdir
from os.path import isfile, join
import sys
sys.path.append('..')

from Pre_Production.Midi_Pre_Processor import *
from Pre_Production.Model_Generator import *
from Shared_Files.Music_Pallete import *
from Pre_Production.Model_Generator import *
from Pre_Production.Music_Translation import *


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
                            "Synth": False,
                            "Drums": True}

all_instruments = pre_processor_obj.return_all_instruments()
instr_note_pairs_dict = pre_processor_obj.return_instr_note_pairs_dict()


print("Synthesizing wanted instr/note pairs...")
instr_wave_forms = get_instr_wave_forms(instrument_name_contains=instrument_name_contains,
                                        all_instruments=all_instruments,
                                        instr_note_pairs_dict=instr_note_pairs_dict,
                                        unique_matrix=True)

for instr, waves in instr_wave_forms.items():
    print("instr:{0} Matrix_Shape: {1}".format(instr, waves.shape))


lyre_bird_model = MusicTranslationModelGenerator(instr_wave_forms=instr_wave_forms,
                                                 models_gen_count=3,
                                                 twilo_account=True)

lyre_bird_model.create_train_model(model_number="119")


lyre_bird_model.generate_music()