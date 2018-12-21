from music21 import converter, note, chord
import glob
import numpy as np
import sys
import os
import shelve
from collections import Counter
from multiprocessing.dummy import Pool as ThreadPool

# Import shared files
sys.path.insert(0, '/home/eric/Desktop/LyreBird/Shared_Files')
from Global_Util import *
from Constants import *

from tqdm import tqdm
from random import random, randint
from concurrent.futures import ProcessPoolExecutor, as_completed


pbar = tqdm(total=80)


def genre_files_tone_extractor(genre_train_files):

    GENRE_NAME = (genre_train_files[0].split('/', -1)[6]).replace('_Midi', '')

    all_genre_notes = []
    file_note_count_dict = dict()
    corrupted_files = []

    for song_count, file in enumerate(genre_train_files):

        pbar.update(1)
        pbar.set_postfix_str(s=file.split("/",4)[4][:50], refresh=True)

        # Attempt to parse midi file
        try:
            midi = converter.parse(file)
        except ...:

            # Midi file couldn't be opened; Remove file
            corrupted_files.append(file)
            # print(file)
            # os.remove(file)
            continue

        # Use the first track on the midi file (Multiple ones)
        midi = midi[TRACK_INDEX]

        # Parse the midi file by the notes it contains
        notes_to_parse = midi.flat.notes

        note_count = 0

        # Convert and append notes to our list; store tone count
        for element in notes_to_parse:

            if isinstance(element, note.Note):
                tone = str(element.pitch)
                all_genre_notes.append(tone)

                note_count += 1

            elif isinstance(element, chord.Chord):

                # Get the numerical representation
                tone = '.'.join(str(n) for n in element.normalOrder)
                all_genre_notes.append(tone)

                note_count += 1

        file_note_count_dict[file] = note_count

        # Print stats so far
        # print("*" * 50)
        # print("Song_Count: {0}:"
        #       "\nGenre_name: {1}"
        #       "\nFile_name: {2}"
        #       "\nNote_Count: {3}".format(song_count + 1,
        #                                  DATASET_GENRE,
        #                                  file.split('/', -1)[-1],
        #                                  len(notes_to_parse)))
        # print("*" * 50)

    return {"genre_name": GENRE_NAME,
            "file_note_count_dict": file_note_count_dict,
            "genre_note_counter": {GENRE_NAME: Counter(all_genre_notes)},
            "corrupted_files": corrupted_files,
            "total_genre_notes": len(all_genre_notes)}

# Grab all datasets full path
datasets_abs_paths = [x[0] for x in os.walk(DATASET_DIRECTORY_PATH)]
datasets_abs_paths.pop(0)
# ---
all_possible_tones = set()

# Catch any files that can't be opened
corrupted_files = []

# Store all possible files by
all_possible_files = list()

for dataset_pth in datasets_abs_paths[:4]:

    all_possible_files.append([dataset_pth + "/" + file
                               for file in glob.glob1(dataset_pth, "*.mid")][:20])



# all_results = parallel_process(array=all_possible_files, function=genre_files_tone_extractor, n_jobs=4, use_kwargs=False, front_num=20)
# pool = ThreadPool(2)
#
# all_results = pool.map(genre_files_tone_extractor,
#                    all_possible_files)

pool = ThreadPool(4)

all_results = pool.imap_unordered(genre_files_tone_extractor, all_possible_files)
pool.close()
pool.join()
pbar.close()

all_corrupted_files = []
all_file_note_count_dict = {}
total_note_count = 0
all_genre_note_counter = {}
all_note_counter = Counter()
for genre_dataset_result in all_results:

    all_file_note_count_dict = {**all_file_note_count_dict,
                                **genre_dataset_result["file_note_count_dict"]}
    all_corrupted_files += genre_dataset_result["corrupted_files"]
    all_genre_note_counter = {**all_genre_note_counter,
                              **genre_dataset_result["genre_note_counter"]}

    all_note_counter += genre_dataset_result["genre_note_counter"][genre_dataset_result["genre_name"]]
    total_note_count += genre_dataset_result["total_genre_notes"]


print("all_file_note_count_dict:", all_file_note_count_dict)
print("\nall_corrupted_files: ", all_corrupted_files)
print("\nall_genre_note_counter: ", all_genre_note_counter)
print("\ntotal_note_count: ", total_note_count)
print("\nall_note_counter: ", all_note_counter)
print("\nall_notes: ", list(all_note_counter.keys()))


# # Remove corrupted files
# for song in corrupted_files:
#     os.remove(song)
#
# # Encode nodes
# master_note_encoder = dict()
# for i, tone in enumerate(all_possible_tones):
#     master_note_encoder[tone] = i
#
# # Decode nodes
# master_note_decoder = dict()
# for note in master_note_decoder.keys():
#     index = master_note_decoder[note]
#     master_note_encoder[index] = note
#


# # Store variables in shelf
# label_encoder_shelve = shelve.open(LSTM_DATA_DUMP_PATH + 'Label_Encoder_Tone_Data')
# label_encoder_shelve["genre_tone_dict"] = genre_tone_dict
# label_encoder_shelve["all_possible_tones"] = all_possible_tones
# label_encoder_shelve["length_all_possible_tones"] = len(all_possible_tones)
# label_encoder_shelve["master_note_decoder"] = master_note_decoder
# label_encoder_shelve["master_note_encoder"] = master_note_encoder
# label_encoder_shelve.close()