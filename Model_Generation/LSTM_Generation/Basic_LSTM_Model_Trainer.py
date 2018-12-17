from music21 import converter, note, chord
import glob
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Activation
from random import shuffle
import sys
import os
import _pickle as cPickle
import zlib

sys.path.insert(0, '/home/eric/Desktop/LyreBird/Model_Generation')
from util import send_sms_to_me
import shelve

# ### Preprocessing
# First we must load the data from the songs.
# To do this, we'll go through all the songs in our training data of MIDI files
# We parse them with music21 to get the individual notes.
# If the element is a chord, then it is converted to it's numerical val.
# After this step we will have all of the notes/chords that appear in string
# Corresponding vocabulary as a set of them all.


notes = []
track = 0

NUM_TRAIN_FILES = 2
EPOCHS = 2
TRAINING_DATASET_PTH = sys.argv[1]
MODEL_PTH = sys.argv[2]
LSTM_DATA_DUMP_PTH = "/home/eric/Desktop/LyreBird/Main_Production/Models/LSTM_Models/LSTM_Data_Dump/"

model_name = (TRAINING_DATASET_PTH.split('/', -1)[-1]).replace('_Midi','_LSTM_Model')

train_files = [TRAINING_DATASET_PTH + "/" + file
               for file in glob.glob1(TRAINING_DATASET_PTH, "*.mid")]


if not train_files:
    exception_output \
        = "ERROR: No Midi files were extracted from the following directory:",\
           TRAINING_DATASET_PTH
    raise Exception(exception_output)

shuffle(train_files)
non_corrupted_files = 0

for i, file in enumerate(train_files):
    try:
        midi = converter.parse(file)
    except:
        continue

    non_corrupted_files += 1

    # There are multiple tracks in the MIDI file, so we'll use the first one
    midi = midi[track]
    notes_to_parse = None

    # Parse the midi file by the notes it contains
    notes_to_parse = midi.flat.notes

    for element in notes_to_parse:
        if isinstance(element, note.Note):
            notes.append(str(element.pitch))
        elif isinstance(element, chord.Chord):
            # get's the normal order (numerical representation) of the chord
            notes.append('.'.join(str(n) for n in element.normalOrder))
    print("Song {} Loaded".format(i+1))

    if non_corrupted_files == NUM_TRAIN_FILES:
        break

print("DONE LOADING SONGS")
# Get all pitch names
pitches = sorted(set(item for item in notes))
# Get all pitch names
vocab_length = len(pitches)
number_notes = len(notes)
print(vocab_length)


# Now we must get these notes in a usable form for our LSTM. Let's construct sequences that can be grouped together to predict the next note in groups of 10 notes.

# Let's use One Hot Encoding for each of the notes and create an array as such of sequences.
#Let's first assign an index to each of the possible notes
note_dict = dict()
for i, tone in enumerate(pitches):
    note_dict[tone] = i
#print(note_dict)

# Now let's construct sequences. Taking each note and encoding it as a numpy array with a 1 in the position of the note it has
sequence_length = 50

# Lets make a numpy array with the number of training examples, sequence length, and the length of the one-hot-encoding
num_training = number_notes - sequence_length

input_notes = np.zeros((num_training, sequence_length, vocab_length))
output_notes = np.zeros((num_training, vocab_length))

for i in range(0, num_training):
    # Here, i is the training example, j is the note in the sequence for a specific training example
    input_sequence = notes[i: i+sequence_length]
    output_note = notes[i+sequence_length]
    for j, note in enumerate(input_sequence):
        input_notes[i][j][note_dict[note]] = 1
    output_notes[i][note_dict[output_note]] = 1


model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape=(sequence_length, vocab_length)))
model.add(Dropout(0.23))
model.add(LSTM(128, return_sequences=False))
model.add(Dropout(0.23))
model.add(Dense(vocab_length))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

history = model.fit(input_notes, output_notes, batch_size=1600, nb_epoch=EPOCHS)

if not os.path.exists(MODEL_PTH):
    os.makedirs(MODEL_PTH)

# Creates a HDF5 file 'my_model.h5'
model.save(MODEL_PTH + '/' + model_name + '.h5')

#### Visualizing the Model's Results
# The models accuracy can be seen here increasing, as it learns the sequences over the course of 200 epochs.
# summarize history for accuracy

# ### Generating New Music

# Make a dictionary going backwards (with index as key and the note as the value)
backward_dict = dict()
for note in note_dict.keys():
    index = note_dict[note]
    backward_dict[index] = note

send_sms_to_me("{0} has finished training with a max accuracy of {1:.2f}%.".format(model_name, max(history.history['acc']) * 100))


# create a "shelf"
shelf = shelve.open(LSTM_DATA_DUMP_PTH + model_name)
shelf['max_history_acc'] = max(history.history['acc'])
shelf['backward_dict'] = backward_dict
shelf['input_notes'] = zlib.compress(cPickle.dumps(input_notes))
shelf['vocab_length'] = vocab_length
shelf['sequence_length'] = sequence_length
shelf.close()