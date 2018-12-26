from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Activation
from keras.optimizers import RMSprop


# Import shared files
import sys
sys.path.append('..')
from Shared_Files.Constants import *
from Shared_Files.Focal_Loss import *
from Shared_Files.Focal_Loss import *


def create_rnn_model(pre_processor_obj):

    rnn_model = Sequential()
    rnn_model.add(LSTM(512, return_sequences=True,
                       input_shape=(MIDI_CONSTANTS.INPUT_SEQUENCE_LEN, 1)))
    rnn_model.add(Dropout(0.27))
    rnn_model.add(LSTM(512, return_sequences=True))
    rnn_model.add(LSTM(512))
    rnn_model.add(Dense(256))
    rnn_model.add(Dropout(0.23))
    rnn_model.add(Dense(len(pre_processor_obj.return_core_atributes()["all_possible_notes"])))
    rnn_model.add(Activation('softmax'))
    rnn_model.compile(loss=[focal_loss(alpha=.25,
                                       gamma=2)],
                                       optimizer=RMSprop(lr=.001),
                                       metrics=['acc'])

    return rnn_model