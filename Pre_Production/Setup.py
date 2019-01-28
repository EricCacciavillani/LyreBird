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

from os import listdir
from os.path import isfile, join
import sys
sys.path.append('..')

from Pre_Production.Midi_Pre_Processor import *
from Pre_Production.Model_Generator import *
from Shared_Files.Music_Pallete import *
from Pre_Production.Model_Generator import *
from Pre_Production.Music_Translation import *


# Init pre-processor in scope; create/grab shelf from given directory
pre_processor_obj = None
pre_processor_shelve = shelve.open(ABS_PATHS.SHELVES_PATH
                                   + SHELVE_NAMES.PRE_PROCESSOR)

pre_processor_obj = MidiPreProcessor(
    ABS_PATHS.TRAINING_DATASET_DIRECTORY_PATH, 2)

pre_processor_shelve["pre_processor"] = pre_processor_obj

# Check to see if the object already exists
if "pre_processor" in pre_processor_shelve.keys():
    print("Found stored pre processor!")

    pre_processor_obj = pre_processor_shelve["pre_processor"]

# Pre-processor not found generate pre-processor
else:

    print("Generating pre processor!")
    pre_processor_obj = MidiPreProcessor(
        ABS_PATHS.TRAINING_DATASET_DIRECTORY_PATH, 20)

    pre_processor_shelve["pre_processor"] = pre_processor_obj

pre_processor_shelve.close()


a = MusicTranslation(pre_processor_obj)
a.create_model(model_path="Testing")