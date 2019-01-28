import time
from collections import Counter
import shelve

from keras.models import Sequential
from keras.layers import Dense, LSTM, GRU, Dropout, Activation
from keras.optimizers import RMSprop, Adam
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, LambdaCallback

import sys
sys.path.append('..')
from Pre_Production.Midi_Pre_Processor import *
from Pre_Production.Model_Generator import *
from Shared_Files.Global_Util import *
from Shared_Files.Focal_Loss import *
from Shared_Files.Constants import *

class ModelGenerator:

    def __init__(self,
                 pre_processor_obj,
                 twilo_account=False,
                 model_path=""):


        # Setting to true allows auto messaging to personal sms through twilo
        self.__twilo_account = twilo_account

        # Extract needed data from pre_processor. (Currently gives everything just in case.)
        self.__all_possible_instr_note_pairs = pre_processor_obj.return_all_possible_instr_note_pairs()

        self.__genre_instr_note_counters = pre_processor_obj.return_genre_instr_note_counters()

        self.__all_possible_instr_note_pairs_counter = pre_processor_obj.return_all_possible_instr_note_pairs_counter()

        self.__all_instruments = pre_processor_obj.return_all_instruments()

        self.__blacklisted_files_validation = pre_processor_obj.return_blacklisted_files_validation()

        self.__genre_file_dict = pre_processor_obj.return_genre_file_dict()

        self.__corrupted_files_paths = pre_processor_obj.return_corrupted_files_paths()

        self.__small_files_paths = pre_processor_obj.return_small_files_paths()

        self.__master_instr_note_encoder = pre_processor_obj.return_master_instr_note_encoder()

        self.__master_instr_note_decoder = pre_processor_obj.return_master_instr_note_decoder()

        self.__master_instr_encoder = pre_processor_obj.return_master_instr_encoder()

        self.__master_instr_decoder = pre_processor_obj.return_master_instr_decoder()

        self.__master_genre_encoder = pre_processor_obj.return_master_genre_encoder()

        self.__master_genre_decoder = pre_processor_obj.return_master_genre_decoder()



    def generate_train_test_on_flat_seq(self,
                                          validation_set_required=True):
        """
            Returns back a training set and test set based on the input
            sizes of the models with a sequence of instr/notes with
        """

        # Init train/test; input/output for models
        train_note_sequences = []
        train_output = []

        test_note_sequences = []
        test_output = []

        # Iterate through given genres and associated meta data on each file
        for genre_name, genre_file_dict in self.__genre_file_dict.items():

            for file_path, file_dict in genre_file_dict.items():

                encoded_notes = self.encode_instr_note_seq(
                    file_dict["flat_instr_note_seq"])

                # Iterate by the input sequence length for each note sequence;
                # Eliminate sequences that are under the input
                for i in range(0, len(encoded_notes) + 1,
                               MIDI_CONSTANTS.FLAT_INPUT_SEQUENCE_LEN):
                    note_sequence = encoded_notes[
                                    i: i + MIDI_CONSTANTS.FLAT_INPUT_SEQUENCE_LEN]

                    if len(
                            note_sequence) == MIDI_CONSTANTS.FLAT_INPUT_SEQUENCE_LEN:

                        try:

                            # Validation file found
                            if validation_set_required \
                                    and file_path in self.__blacklisted_files_validation:

                                # Target will be the next note

                                test_output.append(encoded_notes[i +
                                                                 MIDI_CONSTANTS.FLAT_INPUT_SEQUENCE_LEN])

                                test_note_sequences.append(note_sequence)

                            # Must be a training file
                            else:

                                train_output.append(encoded_notes[i +
                                                                  MIDI_CONSTANTS.FLAT_INPUT_SEQUENCE_LEN])

                                train_note_sequences.append(note_sequence)

                        # Input sequence fit but there was no target
                        except IndexError:
                            continue
                    else:
                        break

        return train_note_sequences, train_output, test_note_sequences, test_output

    def train_category_rnn_model(self,
                                 saved_name="Category_Based_RNN_v12"):

        # Create testing/training data splits
        X_train, y_train, X_test, y_test = self.generate_train_test_on_flat_seq()

        num_classes = len(self.__all_possible_instr_note_pairs)

        # Change shape of input and output to accommodate with model designs
        X_train = np.reshape(X_train, (len(X_train),
                                       MIDI_CONSTANTS.FLAT_INPUT_SEQUENCE_LEN, 1))
        y_train = np_utils.to_categorical(y_train, num_classes=num_classes)

        # ---
        X_test = np.reshape(X_test, (len(X_test), MIDI_CONSTANTS.FLAT_INPUT_SEQUENCE_LEN, 1))
        y_test = np_utils.to_categorical(y_test, num_classes=num_classes)

        # Create RNN Model
        rnn_model = Sequential()
        rnn_model.add(LSTM(300, return_sequences=True, input_shape=(MIDI_CONSTANTS.FLAT_INPUT_SEQUENCE_LEN, 1)))
        rnn_model.add(Dropout(0.35))
        rnn_model.add(LSTM(600, return_sequences=True))
        rnn_model.add(Dropout(0.35))
        rnn_model.add(GRU(300))
        rnn_model.add(Dropout(0.35))
        rnn_model.add(Dense(300))
        rnn_model.add(Dropout(0.25))
        rnn_model.add(Dense(num_classes))
        rnn_model.add(Activation('softmax'))
        rnn_model.compile(loss=[focal_loss(alpha=.25,
                                           gamma=2)],
                          optimizer=Adam(lr=.001),
                          metrics=['acc'])

        # Init callbacks for model fitting
        epoc_cool_gpu = LambdaCallback(on_epoch_begin=
                                       lambda e, l:
                                       time.sleep(HARDWARE_RELATED_CONSTANTS.SECONDS_TO_COOL_GPU)
                                       if e != 0 and e % 50 == 0 else None)

        filepath = str(ABS_PATHS.SAVED_MODELS_PATH) + saved_name\
                                                    + "_Model.hdf5"
        checkpoint = ModelCheckpoint(
            filepath,
            monitor='val_acc',
            verbose=0,
            save_best_only=True,
            mode='max'
        )
        reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=.9,
                                      patience=50, min_lr=0.0001)
        callbacks_list = [checkpoint, reduce_lr, epoc_cool_gpu]

        # ----
        history = rnn_model.fit(X_train, y_train,
                                validation_data=(X_test, y_test),
                                callbacks=callbacks_list,
                                batch_size=650,
                                epochs=2000)

        # Send text message
        if self.__twilo_account:
            send_sms_to_me("Max found train_acc {0:.3f}% and test acc {1:.3f}%.\n"
                           "Max_found test_acc {2:.3f}% and train acc {3:.3f}%"
                           "".format(max(history.history["acc"]) * 100,
                           history.history["val_acc"][np.argmax(history.history["acc"])] * 100,

                           max(history.history["val_acc"]) * 100,
                           history.history["acc"][np.argmax(history.history["val_acc"])] * 100))

        # Shelve up model history data
        model_hist_shelve = shelve.open(ABS_PATHS.SHELVES_PATH + SHELVE_NAMES.MODELS_HISTORY)

        model_hist_shelve[saved_name] = {"Train_Accuracy": history.history["acc"],
                                         "Test_Accuracy": history.history["val_acc"]}

        model_hist_shelve.close()


    # --------------- Basic Functionality ---------------
    def encode_instr_note(self, instr_note_str):
        return self.__master_instr_note_encoder[instr_note_str]

    def encode_instr_note_seq(self, instr_note_seq):
        return [self.__master_instr_note_encoder[instr_note_pair]
                for instr_note_pair in instr_note_seq]

    # ----
    def decode_instr_note(self, instr_note_num):
        return self.__master_instr_note_decoder[instr_note_num]

    def decode_instr_note_seq(self, instr_note_seq):
        return [self.__master_instr_note_decoder[instr_note_pair]
                for instr_note_pair in instr_note_seq]

    # ----
    def encode_instr(self, instr_str):
        return self.__master_instr_encoder[instr_str]

    def decode_instr(self, instr_num):
        return self.__master_instr_decoder[instr_num]

    # ----
    def encode_genre(self, genre_str):
        return self.__master_genre_encoder[genre_str]

    def decode_genre(self, genre_num):
        return self.__master_genre_decoder[genre_num]
