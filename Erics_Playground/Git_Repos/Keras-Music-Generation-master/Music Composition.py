from music21 import converter, instrument, note, chord, midi, stream
import glob
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Input, LSTM, Dropout, Activation
from keras.callbacks import EarlyStopping, ModelCheckpoint
from random import shuffle


# ### Preprocessing
# First we must load the data from the songs.
# To do this, we'll go through all the songs in our training data of MIDI files
# We parse them with music21 to get the individual notes.
# If the element is a chord, then it is converted to it's numerical val.
# After this step we will have all of the notes/chords that appear in string
# Corresponding vocabulary as a set of them all.


notes = []
track = 0

train_files = [file for file in glob.glob("trainedOn/*.mid")]
shuffle(train_files)
train_files = train_files[:80]

print(len(train_files))

for i, file in enumerate(train_files):
    midi = converter.parse(file)

    # There are multiple tracks in the MIDI file, so we'll use the first one
    midi = midi[track]
    print(file)
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

print(num_training)
print(sequence_length)
print(vocab_length)

input_notes = np.zeros((num_training, sequence_length, vocab_length))
output_notes = np.zeros((num_training, vocab_length))

for i in range(0, num_training):
    # Here, i is the training example, j is the note in the sequence for a specific training example
    input_sequence = notes[i: i+sequence_length]
    output_note = notes[i+sequence_length]
    for j, note in enumerate(input_sequence):
        input_notes[i][j][note_dict[note]] = 1
    output_notes[i][note_dict[output_note]] = 1


# In[ ]:


model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape=(sequence_length, vocab_length)))
model.add(Dropout(0.2))
model.add(LSTM(128, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(vocab_length))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])

history = model.fit(input_notes, output_notes, batch_size=1600, nb_epoch=600)


# #### Visualizing the Model's Results
# The models accuracy can be seen here increasing, as it learns the sequences over the course of 200 epochs.

# summarize history for accuracy
print(history.history['acc'])

# ### Generating New Music

# Make a dictionary going backwards (with index as key and the note as the value)
backward_dict = dict()
for note in note_dict.keys():
    index = note_dict[note]
    backward_dict[index] = note

# pick a random sequence from the input as a starting point for the prediction
n = np.random.randint(0, len(input_notes)-1)
sequence = input_notes[n]
start_sequence = sequence.reshape(1, sequence_length, vocab_length)
output = []

# Let's generate a song of 100 notes
for i in range(0, 100):
    newNote = model.predict(start_sequence, verbose=0)
    # Get the position with the highest probability
    index = np.argmax(newNote)
    encoded_note = np.zeros((vocab_length))
    encoded_note[index] = 1
    output.append(encoded_note)
    sequence = start_sequence[0][1:]
    start_sequence = np.concatenate((sequence, encoded_note.reshape(1, vocab_length)))
    start_sequence = start_sequence.reshape(1, sequence_length, vocab_length)
    

# Now output is populated with notes in their string form
for element in output:
    print(element)


# ### Convert to MIDI format
# Code here to output to MIDI files taken
# from github repo https://github.com/Skuldur/Classical-Piano-Composer.



from music21 import converter, instrument, note, chord, midi, stream

finalNotes = [] 
for element in output:
    index = list(element).index(1)
    finalNotes.append(backward_dict[index])
    
offset = 0
output_notes = []
    
# create note and chord objects based on the values generated by the model
for pattern in finalNotes:
    # pattern is a chord
    if ('.' in pattern) or pattern.isdigit():
        notes_in_chord = pattern.split('.')
        notes = []
        for current_note in notes_in_chord:
            new_note = note.Note(int(current_note))
            new_note.storedInstrument = instrument.Piano()
            notes.append(new_note)
        new_chord = chord.Chord(notes)
        new_chord.offset = offset
        output_notes.append(new_chord)
    # pattern is a note
    else:
        print(pattern)
        new_note = note.Note(pattern)
        new_note.offset = offset
        new_note.storedInstrument = instrument.Piano()
        output_notes.append(new_note)

    # increase offset each iteration so that notes do not stack
    offset += 0.5

midi_stream = stream.Stream(output_notes)

midi_stream.write('midi', fp='test_output.mid')

# Creates a HDF5 file 'my_model.h5'
model.save('basic_LSTM.h5')