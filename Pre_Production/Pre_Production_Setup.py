import shelve


import sys
sys.path.append('..')
from Pre_Production.Midi_Pre_Processor import *
from Pre_Production.Model_Structures import *
from Shared_Files.Global_Util import *
from Shared_Files.Constants import *


# test_obj = MidiPreProcessor(ABS_PATHS.TRAINING_DATASET_DIRECTORY_PATH,
#                             genre_sub_sample_set=40,
#                             validation_set=True)

pre_processor_shelve = shelve.open(ABS_PATHS.SHELVES_PATH + "/Midi_Pre_Processor")

pre_processor_obj = pre_processor_shelve["pre_processor"]

test_attr_dict = pre_processor_obj.return_core_atributes()


X_train, y_train, X_test, y_test = pre_processor_obj.train_test_split()

print("Old X_train.shape: ", X_train.shape)
X_train = np.reshape(X_train, (X_train.shape[0],1,X_train.shape[1],))
print("New X_train.shape: ", X_train.shape)
print(X_train)
print()
print("Old y_train.shape: ", y_train.shape)
y_train = np.reshape(y_train, ((y_train.shape[0]), 1))
print(y_train.shape)
print("New y_train.shape: ", y_train.shape)

print(y_train.shape)
print(y_train)

history = RNN_model.fit(X_train, y_train,
                        batch_size=1000,
                        epochs=500)