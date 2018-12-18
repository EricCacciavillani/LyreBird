from music21 import converter, note, chord
import glob
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Activation
from keras.optimizers import RMSprop
from random import shuffle
from collections import defaultdict
import sys
import os
from keras.models import load_model
import shelve

sys.path.insert(0, '/home/eric/Desktop/LyreBird/Shared_Files')
from Global_Util import *
from Constants import *
from Focal_Loss import *

# --- Preprocessing
# Grab notes and cords for training our model to predict.

TRACK_INDEX = 0

TRAINING_DATASET_PTH = sys.argv[1]

GENRE_NAME = (TRAINING_DATASET_PTH.split('/', -1)[-1]).replace('_Midi','')

train_files = [TRAINING_DATASET_PTH + "/" + file
               for file in glob.glob1(TRAINING_DATASET_PTH, "*.mid")]


if not train_files:
    exception_output \
        = "ERROR: No Midi files were extracted from the following directory:",\
           TRAINING_DATASET_PTH
    raise Exception(exception_output)

shuffle(train_files)

while train_files:

    genre_notes = []
    for i, file in enumerate(train_files):

        try:
            midi = converter.parse(file)
        except:
            continue

        # Use the first track on the midi file (Multiple ones)
        midi = midi[TRACK_INDEX]

        # Parse the midi file by the notes it contains
        notes_to_parse = midi.flat.notes

        for element in notes_to_parse:

            if isinstance(element, note.Note):
                genre_notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                # Get the numerical representation
                genre_notes.append('.'.join(str(n) for n in element.normalOrder))

        print("*" * 50)
        print("Song_Count: {0}:\nGenre_name: {1}\nFile_name: {2}\nNote_Count: {3}".format(
            i+1,GENRE_NAME,file.split('/', -1)[-1],
            len(notes_to_parse)))
        print("*" * 50)

        train_files.remove(file)

        if len(genre_notes) >= MAX_TRAINING_NOTES:
            break

    print("DONE LOADING SONGS")

    # Get all pitch names
    label_encoder_shelve = shelve.open(LSTM_DATA_DUMP_PATH + 'Label_Encoder_Tone_Data')
    pitches = label_encoder_shelve["all_possible_tones"]
    length_all_possible_tones = label_encoder_shelve["length_all_possible_tones"]
    master_note_decoder = label_encoder_shelve["master_note_decoder"]
    master_note_encoder = label_encoder_shelve["master_note_encoder"]
    label_encoder_shelve.close()

    # Get all pitch names
    vocab_length = length_all_possible_tones
    number_notes = len(genre_notes)

    # Now we must get these notes in a usable form for our LSTM. Let's construct sequences that can be grouped together to predict the next note in groups of 10 notes.

    # Let's use One Hot Encoding for each of the notes and create an array as such of sequences.
    #Let's first assign an index to each of the possible notes

    #print(note_dict)

    # Now let's construct sequences. Taking each note and encoding it as a numpy array with a 1 in the position of the note it has
    sequence_length = 50

    # Lets make a numpy array with the number of training examples, sequence length, and the length of the one-hot-encoding
    num_training = number_notes - sequence_length

    input_notes = np.zeros((num_training, sequence_length, vocab_length))
    output_notes = np.zeros((num_training, vocab_length))

    for i in range(0, num_training):
        # Here, i is the training example, j is the note in the sequence for a specific training example
        input_sequence = genre_notes[i: i+sequence_length]
        output_note = genre_notes[i+sequence_length]
        for j, tone in enumerate(input_sequence):
            input_notes[i][j][master_note_encoder[tone]] = 1
        output_notes[i][master_note_encoder[output_note]] = 1


    model = None
    print(LSTM_MODELS_PATH + "Categorical_LSTM_Model.h5")

    if os.path.isfile(LSTM_MODELS_PATH + "Categorical_LSTM_Model.h5"):
        model = load_model((LSTM_MODELS_PATH + "Categorical_LSTM_Model.h5"))
    else:
        model = Sequential()
        model.add(LSTM(128, return_sequences=True, input_shape=(sequence_length,
                                                                vocab_length)))
        model.add(Dropout(0.23))
        model.add(LSTM(128, return_sequences=False))
        model.add(Dropout(0.23))
        model.add(Dense(vocab_length))
        model.add(Activation('softmax'))

        # [focal_loss(alpha=.25, gamma=2)]
        model.compile(loss="categorical_crossentropy",
                      optimizer=RMSprop(lr=0.001),
                      metrics=['acc'])
    history = model.fit(input_notes, output_notes,
                        batch_size=3000, nb_epoch=2)


    #### Visualizing the Model's Results
    # The models accuracy can be seen here increasing, as it learns the sequences over the course of 200 epochs.
    # summarize history for accuracy

    # ### Generating New Music

    # Make a dictionary going backwards (with index as key and the note as the value)

    max_accur = max(history.history['acc']) * 100
    min_accur = min(history.history['acc']) * 100
    accur_growth = max_accur - min_accur

    # model.save(LSTM_MODELS_PATH + "Categorical_LSTM_Model.h5")
    # send_sms_to_me("{0} sub sampled data."
    #                "\nStarting Accuracy {1:.2f}%"
    #                "\nFinished Accuracy {2:.2f}%"
    #                "\nOverall Growth:{3:.2f}%".format(
    #     GENRE_NAME, min_accur, max_accur, accur_growth))

    model_accur_shelf = shelve.open(LSTM_DATA_DUMP_PATH +
                                    '/Categorical_LSTM_Model_Accuracy')


    if "model_accur_by_genre" not in model_accur_shelf.keys():
        model_accur_shelf["model_accur_by_genre"] = defaultdict(dict)

    if GENRE_NAME not in model_accur_shelf["model_accur_by_genre"].keys():
        model_accur_shelf["model_accur_by_genre"][GENRE_NAME] = "Testing shit!!!!!"
        print("jnjnjj")


    print("Catch:")
    print(model_accur_shelf["model_accur_by_genre"][GENRE_NAME])
    print("End:")
    # model_accur_shelf["model_accur_by_genre"][GENRE_NAME]["max_accur"].append(
    #     max_accur)
    #
    # model_accur_shelf["model_accur_by_genre"][GENRE_NAME]["min_accur"].append(
    #     min_accur)
    #
    # model_accur_shelf["model_accur_by_genre"][GENRE_NAME]["accur_growth"].append(
    #     accur_growth)

    model_accur_shelf.close()

    print(len(train_files))