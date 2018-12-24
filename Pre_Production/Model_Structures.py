from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Activation
from keras.optimizers import RMSprop


# Import shared files
import sys
sys.path.append('..')
from Shared_Files.Constants import *
from Shared_Files.Focal_Loss import *
from Shared_Files.Focal_Loss import *

RNN_model = Sequential()
RNN_model.add(LSTM(128, return_sequences=True, input_shape=(1, MIDI_CONSTANTS.INPUT_SEQUENCE_LEN)))
RNN_model.add(Dropout(0.23))
RNN_model.add(LSTM(128, return_sequences=False))
RNN_model.add(Dropout(0.23))
RNN_model.add(Dense(1))
RNN_model.add(Activation('softmax'))


focal_loss_alpha = .25
focal_loss_gamma = 2.

RNN_model.compile(loss=[focal_loss(alpha=focal_loss_alpha,
                                   gamma=focal_loss_gamma)],
                                   optimizer=RMSprop(lr=.001),
                                   metrics=['acc'])