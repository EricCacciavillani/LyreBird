from music21 import converter, note, chord
import glob
import numpy as np
import sys
import os
import shelve
from collections import Counter

sys.path.insert(0, '/home/eric/Desktop/LyreBird/Shared_Files')
from Global_Util import *
from Constants import *

DATASETS_PATHS = [x[0] for x in os.walk(DATASET_DIRECTORY_PATH)]
DATASETS_PATHS.pop(0)

all_possible_tones = set()
i = 0
corrupted_files = []
genre_tone_dict = Counter()
for dataset_pth in DATASETS_PATHS:

    train_files = [dataset_pth + "/" + file
                   for file in glob.glob1(dataset_pth, "*.mid")]

    dataset_genre = extract_genre_name_from_dir(dataset_pth)

    genre_tone_dict[dataset_genre] = Counter()

    for i, file in enumerate(train_files):

        try:
            midi = converter.parse(file)
        except:
            corrupted_files.append(file)
            continue

        # Use the first track on the midi file (Multiple ones)
        midi = midi[TRACK_INDEX]

        # Parse the midi file by the notes it contains
        notes_to_parse = midi.flat.notes

        for element in notes_to_parse:

            if isinstance(element, note.Note):
                tone = str(element.pitch)
                all_possible_tones.add(tone)

                if tone in genre_tone_dict[dataset_genre].keys():
                    genre_tone_dict[dataset_genre][tone] += 1
                else:
                    genre_tone_dict[dataset_genre][tone] = 1

            elif isinstance(element, chord.Chord):
                tone = '.'.join(str(n) for n in element.normalOrder)
                # Get the numerical representation
                all_possible_tones.add(tone)

                if tone in genre_tone_dict[dataset_genre].keys():
                    genre_tone_dict[dataset_genre][tone] += 1
                else:
                    genre_tone_dict[dataset_genre][tone] = 1

        print("*" * 50)
        print("Song_Count: {0}:"
              "\nGenre_name: {1}"
              "\nFile_name: {2}"
              "\nNote_Count: {3}".format(i + 1,
                                         dataset_genre,
                                         file.split('/', -1)[-1],
                                         len(notes_to_parse)))
        print("*" * 50)

for song in corrupted_files:
    os.remove(song)

master_note_encoder = dict()
for i, tone in enumerate(all_possible_tones):
    master_note_encoder[tone] = i

master_note_decoder = dict()
for note in master_note_decoder.keys():
    index = master_note_decoder[note]
    master_note_encoder[index] = note


label_encoder_shelve = shelve.open(LSTM_DATA_DUMP_PATH + '/Label_Encoder_Tone_Data')
label_encoder_shelve["genre_tone_dict"] = genre_tone_dict
label_encoder_shelve["all_possible_tones"] = all_possible_tones
label_encoder_shelve["length_all_possible_tones"] = len(all_possible_tones)
label_encoder_shelve["master_note_decoder"] = master_note_decoder
label_encoder_shelve["master_note_encoder"] = master_note_encoder
label_encoder_shelve.close()
