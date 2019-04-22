import sys
sys.path.append('..')

from Shared_Files.Constants import *
from Shared_Files.Global_Util import *

import pickle
import numpy
from music21 import instrument, note, stream, chord
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Activation


# Main function to generate the music
def generate():

    # Open and read notes from file
    with open(ABS_PATHS.SAVED_NOTES_PATH + "Classical_Notes", 'rb') as filepath:
        notes = pickle.load(filepath)

    pitchnames = sorted(set(item for item in notes))
    n_vocab = len(set(notes))

    # ----
    network_input, normalized_input = prepare_sequences(notes, pitchnames, n_vocab)
    model = create_network(normalized_input, n_vocab)
    prediction_output = generate_notes(model, network_input, pitchnames, n_vocab)
    create_midi(prediction_output)

def prepare_sequences(notes,
                      pitchnames,
                      n_vocab):

    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    sequence_length = 100
    network_input = []
    output = []
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        output.append(note_to_int[sequence_out])

    n_patterns = len(network_input)

    normalized_input = numpy.reshape(network_input, (n_patterns, sequence_length, 1))
    normalized_input = normalized_input / float(n_vocab)

    return (network_input, normalized_input)

# Recreate the model from train
def create_network(network_input, n_vocab):
    model = Sequential()
    model.add(LSTM(
        512,
        input_shape=(network_input.shape[1], network_input.shape[2]),
        return_sequences=True
    ))
    model.add(Dropout(0.3))
    model.add(LSTM(512, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(512))
    model.add(Dense(256))
    model.add(Dropout(0.3))
    model.add(Dense(n_vocab))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    model.load_weights(ABS_PATHS.SAVED_WEIGHTS_PATH + 'Classic_Music_Weights.hdf5')

    return model

# Predict notes with given input
def generate_notes(model,
                   network_input,
                   pitchnames,
                   n_vocab):

    start = numpy.random.randint(0, len(network_input)-1)

    int_to_note = dict((number, note) for number, note in enumerate(pitchnames))

    pattern = network_input[start]
    prediction_output = []

    for note_index in range(200):
        prediction_input = numpy.reshape(pattern, (1, len(pattern), 1))
        prediction_input = prediction_input / float(n_vocab)

        prediction = model.predict(prediction_input, verbose=0)

        index = numpy.argmax(prediction)
        result = int_to_note[index]
        prediction_output.append(result)

        pattern.append(index)
        pattern = pattern[1:len(pattern)]

    return prediction_output

# Generate music
def create_midi(prediction_output):

    offset = 0
    output_notes = []

    # Iterate list of notes to generate music
    for pattern in prediction_output:

        # Must be a Chord
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
        # Must be a note
        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)

        # Offset next note by .5 of a second
        offset += 0.5

    midi_stream = stream.Stream(output_notes)

    midi_stream.write('midi', fp=ABS_PATHS.AUDIO_CACHE_PATH + 'Output.mid')


if __name__ == '__main__':
    generate()
