import shelve
from keras.utils import np_utils
import sys
sys.path.append('..')
from Pre_Production.Midi_Pre_Processor import *
from Pre_Production.Model_Structures import *
from Shared_Files.Global_Util import *
from Shared_Files.Constants import *
import numpy as np


test_obj = MidiPreProcessor(ABS_PATHS.TRAINING_DATASET_DIRECTORY_PATH)

exit(-1)

pre_processor_shelve = shelve.open(ABS_PATHS.SHELVES_PATH + "/Midi_Pre_Processor")
# pre_processor_shelve["pre_processor"] = test_obj

pre_processor_obj = pre_processor_shelve["pre_processor"]

test_attr_dict = pre_processor_obj.return_core_atributes()


X_train, y_train, X_test, y_test = pre_processor_obj.train_test_split()


X_train = np.reshape(X_train, (len(X_train), MIDI_CONSTANTS.INPUT_SEQUENCE_LEN, 1))
y_train = np_utils.to_categorical(y_train)
print(y_train.shape)
print(X_train.shape)

print()


# RNN_model = create_rnn_model(pre_processor_obj)
#
# history = RNN_model.fit(X_train, y_train,
#                         batch_size=200,
#                         epochs=1000)

