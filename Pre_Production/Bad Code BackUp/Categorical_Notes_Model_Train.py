from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Activation
from keras.optimizers import RMSprop
from music21 import converter, note, chord
from random import shuffle
from collections import defaultdict
from keras.models import load_model
import numpy as np
import sys
import os
import glob
import shelve
import time

# Import shared files
sys.path.insert(0, '/home/eric/Desktop/LyreBird/Shared_Files')
from Global_Util import *
from Constants import *
from Focal_Loss import *

# Create constants
TRACK_INDEX = 0

TRAINING_DATASET_PTH = sys.argv[1]

GENRE_NAME = (TRAINING_DATASET_PTH.split('/', -1)[-1]).replace('_Midi', '')

# Grab files full path for each midi file in the training set
train_files = [TRAINING_DATASET_PTH + "/" + file
               for file in glob.glob1(TRAINING_DATASET_PTH, "*.mid")]

TOTAL_TRAIN_FILES_LEN = len(train_files)

# Empty directory check
if not train_files:
    exception_output \
        = "ERROR: No Midi files were extracted from the following directory:",\
           TRAINING_DATASET_PTH
    raise Exception(exception_output)

# We shuffle the entire set so each training batch for every genre varies
shuffle(train_files)
song_count = 0

# Master loop; keep iterating until no files for that genre are left
while train_files:

    sampled_genre_notes = []
    for i, file in enumerate(train_files):

        # Removing files already seen so the next genre sample will be
        train_files.remove(file)

        # ---
        try:
            midi_song = converter.parse(file)
        except:
            continue

        song_count += 1

        # Use the first track on the midi file (Multiple ones)
        midi_song = midi_song[TRACK_INDEX]

        # Extract notes from the song
        notes_to_parse = midi_song.flat.notes

        # Convert and append notes to our list
        for note_element in notes_to_parse:

            if isinstance(note_element, note.Note):
                sampled_genre_notes.append(str(note_element.pitch))

            elif isinstance(note_element, chord.Chord):

                # Get the numerical representation of the cord
                sampled_genre_notes.append('.'.join(str(n)
                                                    for n in note_element.normalOrder))

        # Display song stats
        print("*" * 50)
        print("Song_Count: {0}:\nGenre_name: {1}\nFile_name: {2}\nNote_Count: {3}".format(
            i+1,GENRE_NAME,file.split('/', -1)[-1],
            len(notes_to_parse)))
        print("*" * 50)

        # Break if we get to many notes for our sub-sample
        if len(sampled_genre_notes) >= MAX_TRAINING_NOTES:
            break

    print("DONE LOADING SONGS")

    # Un-Shelve data from file on directory path for encoded data
    label_encoder_shelve = shelve.open(LSTM_DATA_DUMP_PATH + 'Label_Encoder_Tone_Data')
    pitches = label_encoder_shelve["all_possible_tones"]
    length_all_possible_tones = label_encoder_shelve["length_all_possible_tones"]
    master_note_encoder = label_encoder_shelve["master_note_encoder"]
    label_encoder_shelve.close()

    # Pre-processing on data
    num_training = len(sampled_genre_notes) - INPUT_SEQUENCE_LEN
    input_notes = np.zeros((num_training, INPUT_SEQUENCE_LEN, length_all_possible_tones))
    output_notes = np.zeros((num_training, length_all_possible_tones))

    # Set input and output for neural network;
    # Using the encoder to one hot encode each tone in a set positional
    # Input: Sequence of tones/notes found within sampled notes
    # Output: A single note
    # Summary: Predict a note with a given sequence
    for i in range(0, num_training):
        input_sequence = sampled_genre_notes[i: i + INPUT_SEQUENCE_LEN]
        output_note = sampled_genre_notes[i + INPUT_SEQUENCE_LEN]
        for j, tone in enumerate(input_sequence):
            input_notes[i][j][master_note_encoder[tone]] = 1
        output_notes[i][master_note_encoder[output_note]] = 1

    # Model generation
    model = None

    focal_loss_alpha = .25
    foca_loss_gamma = 2.

    # Check if the model already exists
    if os.path.isfile(LSTM_MODELS_PATH + "Categorical_LSTM_Model.h5"):
        model = load_model((LSTM_MODELS_PATH + "Categorical_LSTM_Model.h5"),
                           custom_objects={'FocalLoss': focal_loss,
                                           'focal_loss_fixed': focal_loss(alpha=focal_loss_alpha,
                                                                          gamma=foca_loss_gamma)})

    # Model doesn't exists currently; generate it
    else:

        model = Sequential()
        model.add(LSTM(128, return_sequences=True, input_shape=(INPUT_SEQUENCE_LEN,
                                                                length_all_possible_tones)))
        model.add(Dropout(0.23))
        model.add(LSTM(128, return_sequences=False))
        model.add(Dropout(0.23))
        model.add(Dense(length_all_possible_tones))
        model.add(Activation('softmax'))

        model.compile(loss=[focal_loss(alpha=focal_loss_alpha,
                                       gamma=foca_loss_gamma)],
                      optimizer=RMSprop(lr=0.001),
                      metrics=['acc'])
    history = model.fit(input_notes, output_notes,
                        batch_size=int(MAX_TRAINING_NOTES/2),
                        epochs=2)

    max_accur = max(history.history['acc']) * 100
    min_accur = min(history.history['acc']) * 100
    accur_growth = max_accur - min_accur

    # Save model on absolute path
    model.save(LSTM_MODELS_PATH + "Categorical_LSTM_Model.h5")

    # Send SMS to personal phone to keep updates
    # send_sms_to_me("{0} sub sampled data."
    #                "\nStarting Accuracy {1:.2f}%"
    #                "\nFinished Accuracy {2:.2f}%"
    #                "\nOverall Growth:{3:.2f}%".format(
    #     GENRE_NAME, min_accur, max_accur, accur_growth))

    # Store data on model accuracy on set genres for later visualization
    model_accur_shelf = shelve.open(LSTM_DATA_DUMP_PATH +
                                    'Categorical_LSTM_Model_Accuracy')

    # Variable has yet to be stored yet ever before
    if "model_accur_by_genre" not in model_accur_shelf.keys():
        model_accur_shelf["model_accur_by_genre"] = defaultdict(dict)

    model_accur_dict = model_accur_shelf["model_accur_by_genre"]

    # Init new genre if needed
    if GENRE_NAME not in model_accur_dict.keys():
        model_accur_dict[GENRE_NAME] = {"max_accur": list(),
                                        "min_accur": list(),
                                        "accur_growth": list()}

    model_accur_dict[GENRE_NAME]["max_accur"].append(max_accur)
    model_accur_dict[GENRE_NAME]["min_accur"].append(min_accur)
    model_accur_dict[GENRE_NAME]["accur_growth"].append(accur_growth)
    model_accur_shelf["model_accur_by_genre"] = model_accur_dict

    model_accur_shelf.close()

    # Let the GPU cool per cycle
    print("Let the GPU cool off...")
    # time.sleep(SECONDS_TO_COOL_GPU)