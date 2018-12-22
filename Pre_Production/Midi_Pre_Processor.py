from music21 import converter, note, chord
import glob
import numpy as np

import os
from collections import Counter
from multiprocessing.dummy import Pool as ThreadPool
from tqdm import tqdm

# Import shared files
import sys
sys.path.append('..')
from Shared_Files.Global_Util import *
from Shared_Files.Constants import *
from Shared_Files.Custom_Exception import *



class Midi_Pre_Processor:

    def __init__(self, path_to_full_data_set):

        # Progress-bar for threading-pool
        self.__pbar = None

        # Store all individual tones found
        self.__all_possible_notes = set()
        self.__all_possible_notes_counter = Counter()

        # Files to ignore for when splicing data into train/test
        self.__blacklisted_files_validation = set()

        # Stores all genres to another dict that stores
        # the corresponding file note size
        self.__all_genre_file_note_count_dict = dict()

        # Stores the genre names with all possible note and associated counts
        self.__all_genre_note_counters = dict()

        # Stores all corrupted files found
        self.__found_corrupted_file_paths = []

        # ---
        self.__total_file_count = 0
        self.__total_note_count = 0

        # Thread pool out reading multiple files of each dataset
        thread_pool_results = self.__thread_pool_datasets_reader(self.__genre_dataset_init,
                                                                 path_to_full_data_set)

        # Init data based on thread pool results
        for genre_dataset_result in thread_pool_results:

            self.__all_genre_file_note_count_dict = {**self.__all_genre_file_note_count_dict,
                                                     **genre_dataset_result["file_note_count_dict"]}
            self.__found_corrupted_file_paths += genre_dataset_result["corrupted_files"]
            self.__all_genre_note_counters = {**self.__all_genre_note_counters,
                                              **genre_dataset_result[
                                                  "genre_note_counter"]}

            self.__all_possible_notes_counter += genre_dataset_result["genre_note_counter"][
                genre_dataset_result["genre_name"]]
            self.__total_note_count += genre_dataset_result["total_genre_notes"]

        self.__all_possible_notes = sorted(set(self.__all_possible_notes_counter.keys()))

        # Corrupted files were found.
        if self.__found_corrupted_file_paths:
            print('There were corrupted files found in your training.')
            display_options_menu(menu_intro="Corrupted files found! "
                                            "It is suggested to get rid of them",
                                 menu_options={1: "Delete all corrupted.",
                                               2: "List all corrupted files",
                                               3: "Ignore"})

            user_input = input("\nInput")

            # Remove corrupted files
            if user_input == 1:
                for song in self.__found_corrupted_file_paths:
                    os.remove(song)

            elif user_input == 2:

                print("Displaying all corrupted songs:\n")
                for song in self.__found_corrupted_file_paths:
                    print(song.split("/", 7)[-1])
            else:
                pass

        # Encode dict for nodes
        self.__master_note_encoder = dict()
        for i, tone in enumerate(self.__all_possible_notes):
            self.__master_note_encoder[tone] = i + 1

        # Decode dict for nodes
        self.__master_note_decoder = {v: k for k, v
                                      in self.__master_note_encoder.items()}

        # Find files for best fit the value requirement
        for genre_name, note_counter in self.__all_genre_note_counters.items():

            genre_note_count = sum(note_counter.values())
            needed_valadation_note_count = int((genre_note_count / self.__total_note_count)\
                                    * genre_note_count)

            note_count_file_dict = {v: k for k, v
                                    in self.__all_genre_file_note_count_dict[genre_name].items()}

            note_count_file_list = list(note_count_file_dict.keys())

            while True and needed_valadation_note_count > 25:

                closest_note_count = find_nearest(numbers=note_count_file_list,
                                                  target=needed_valadation_note_count)
                needed_valadation_note_count -= closest_note_count

                self.__blacklisted_files_validation.add(
                    note_count_file_dict[closest_note_count])

                note_count_file_list.remove(closest_note_count)

            if not self.__blacklisted_files_validation:
                self.__blacklisted_files_validation.add(note_count_file_dict[find_nearest(note_count_file_list, needed_valadation_note_count)])


        print(self.__blacklisted_files_validation)

    # Displays core attributes
    def display_attributes(self):
        print("*" * 50)
        print("all_genre_file_note_count_dict: ",
              self.__all_genre_file_note_count_dict)
        print("\nall_corrupted_files: ", self.__found_corrupted_file_paths)
        print("\nall_genre_note_counter: ", self.__all_genre_note_counters)
        print("\ntotal_note_count: ", self.__total_note_count)
        print("\ntotal_file_count: ", self.__total_file_count)
        print("\nall_possible_notes: ", self.__all_possible_notes_counter)
        print("\nall_notes: ", self.__all_possible_notes)
        print("\nblacklist_files_validation: ", self.__blacklisted_files_validation)
        print("*" * 50)

    # Threads out each dataset by it's genre
    def __thread_pool_datasets_reader(self, function,
                                      path_to_full_data_set):

        # Get
        all_train_datasets_paths = [x[0] for x in os.walk(
            path_to_full_data_set)]
        all_train_datasets_paths.pop(0)

        files_by_genre = []

        for dataset_pth in all_train_datasets_paths[:4]:

            dataset_files = [dataset_pth + "/" + file for file in
                             glob.glob1(dataset_pth, "*.mid")][:6]

            if len(dataset_files):
                self.__total_file_count += len(dataset_files)

                files_by_genre.append(dataset_files)

        self.__pbar = tqdm(total=self.__total_file_count)
        pool = ThreadPool(len(files_by_genre))
        all_results = pool.imap_unordered(function,
                                          files_by_genre)

        pool.close()
        pool.join()
        self.__pbar.close()
        self.__pbar = None

        return all_results

    def read_midi_file(self, file):

        song_notes = []

        # Attempt to parse midi file
        try:
            midi = converter.parse(file)
        except Midi_Reading_Error as err:

            # Midi file couldn't be opened
            return {"song_notes": [],
                    "note_count": [],
                    "corrupted": True}

        # Use the first track on the midi file (Multiple ones)
        midi = midi[MIDI_CONSTANTS.TRACK_INDEX]

        # Parse the midi file by the notes it contains
        notes_to_parse = midi.flat.notes

        note_count = 0

        # Convert and append notes to our list; store tone count
        for element in notes_to_parse:

            if isinstance(element, note.Note):
                tone = str(element.pitch)
                song_notes.append(tone)

                note_count += 1

            elif isinstance(element, chord.Chord):

                # Get the numerical representation
                tone = '.'.join(str(n) for n in element.normalOrder)
                song_notes.append(tone)

                note_count += 1

        return {"song_notes": song_notes,
                "note_count": note_count,
                "corrupted": False}

    def __genre_dataset_init(self, genre_train_files):

        all_genre_notes = []
        file_note_count_dict = dict()
        corrupted_files = []

        genre_name = genre_train_files[0].split("/")[-2].replace('_Midi','')

        for _, file in enumerate(genre_train_files):

            self.__pbar.update(1)
            self.__pbar.set_postfix_str(s=file.split("/",-1)[-1][:20], refresh=True)
            midi_file_attr = self.read_midi_file(file)

            if not midi_file_attr["corrupted"]:
                file_note_count_dict[file] = midi_file_attr["note_count"]
                all_genre_notes += midi_file_attr["song_notes"]
            else:
                corrupted_files.append(file)

        return {"genre_name": genre_name,
                "file_note_count_dict": {genre_name: file_note_count_dict},
                "genre_note_counter": {genre_name: Counter(all_genre_notes)},
                "corrupted_files": corrupted_files,
                "total_genre_notes": len(all_genre_notes)}



a = Midi_Pre_Processor(ABS_PATHS.TRAINING_DATASET_DIRECTORY_PATH)