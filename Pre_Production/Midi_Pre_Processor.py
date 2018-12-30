import pretty_midi
import glob
import os
import copy
from collections import Counter
from multiprocessing.dummy import Pool as ThreadPool
from tqdm import tqdm

# Import shared files
import sys
sys.path.append('..')

import warnings
warnings.filterwarnings("ignore")

from Shared_Files.Global_Util import *
from Shared_Files.Constants import *


'''
    Reads across multiple datasets stores meta data on each data set and files and
    stores them to be used for later.
'''
class MidiPreProcessor:

    def __init__(self, path_to_full_data_set,
                 genre_sub_sample_set=sys.maxsize):

        # Progress-bar for threading-pool
        self.__pbar = None

        # ---
        self.__all_possible_instr_note_pairs = set()
        self.__all_possible_instr_note_pairs_counter = Counter()
        self.__all_instruments = set()

        # Files to ignore for when splicing data into train/test
        self.__blacklisted_files_validation = set()

        # Stores all genres to another dict that stores
        # the corresponding file note size
        self.__genre_file_dict = dict()

        self.__genre_instr_note_counters = dict()

        # Stores all corrupted files found
        self.__corrupted_files_paths = []

        # Store files that are to small (Determined by the input sequence)
        self.__small_files_paths = []

        # Init encoders and decoders
        self.__master_instr_note_encoder = dict()
        self.__master_instr_note_decoder = dict()

        self.__master_instr_encoder = dict()
        self.__master_instr_decoder = dict()

        self.__master_genre_encoder = dict()
        self.__master_genre_decoder = dict()
        # ---------------------------------

        # Numeric counts
        self.__total_file_count = 0
        self.__total_intr_note_pair_size = 0

        # Thread pool out reading multiple files of each dataset
        thread_pool_results = self.__thread_pool_datasets_reader(
            self.__genre_dataset_init, path_to_full_data_set, genre_sub_sample_set)

        # Init all data based on thread pool results
        for genre_count, genre_dataset_result in enumerate(thread_pool_results):

            # Add to set of all instr/note pairs
            self.__all_possible_instr_note_pairs |= genre_dataset_result["genre_instr_note_pairs"]

            # Add to set of all instruments
            self.__all_instruments |= genre_dataset_result["genre_instruments"]

            # Numeric value of non-unique total instr/note pairs
            self.__total_intr_note_pair_size += genre_dataset_result[
                "genre_size"]

            # Store files based on the genre of songs
            self.__genre_file_dict = {**self.__genre_file_dict,
                                      **genre_dataset_result["genre_file_meta_data"]}

            # Store counter object based on genre
            self.__genre_instr_note_counters[genre_dataset_result[
                "genre_name"]] = genre_dataset_result["genre_instr_note_pairs_counter"]

            # Counter object of all possible instr/note
            self.__all_possible_instr_note_pairs_counter += genre_dataset_result["genre_instr_note_pairs_counter"]

            # ---
            self.__corrupted_files_paths += genre_dataset_result[
                "corrupted_files"]
            self.__small_files_paths += genre_dataset_result["small_files"]

        # Sort all data before encoding for my own sanity
        self.__all_possible_instr_note_pairs = sorted(
            self.__all_possible_instr_note_pairs)
        self.__all_instruments = sorted(self.__all_instruments)

        # Begin creating label encoders and decoders

        # -----
        for label, (genre, _) in enumerate(
                self.__genre_instr_note_counters.items()):
            self.__master_genre_encoder[genre] = label + 1

        self.__master_genre_decoder = {v: k for k, v
                                       in self.__master_genre_encoder.items()}
        # -----
        for label, instr_note_pair in enumerate(
                self.__all_possible_instr_note_pairs):
            self.__master_instr_note_encoder[instr_note_pair] = label + 1

        self.__master_instr_note_decoder = {v: k for k, v
                                            in
                                            self.__master_instr_note_encoder.items()}
        # -----

        for label, instr in enumerate(
                self.__all_instruments):
            self.__master_instr_encoder[instr] = label + 1

        self.__master_instr_decoder = {v: k for k, v
                                       in self.__master_instr_encoder.items()}
        # -------------------------------------

        # Corrupted files were found.
        if self.__corrupted_files_paths:

            print("The Pre Processor found {0} corrupted files".format(len(self.__corrupted_files_paths)))
            print("Displaying all corrupted songs:\n")
            for song in self.__corrupted_files_paths:
                print("\t", song.split("/", 6)[-1])
            print()

            display_options_menu(menu_intro="Corrupted files found!\n"
                                            "\tIt's fine if you don't delete"
                                            " them.Just know the pre-processor"
                                            " will not use them at all.",
                                 menu_options={1: "Delete all corrupted files",
                                               2: "Ignore"})
            user_input = input("\nInput: ")

            # Remove corrupted files
            if user_input == "1":
                self.delete_corrupted_files()
            else:
                pass
        # ---------------------------------------------

        # Small files were found.
        if self.__small_files_paths:

            print("The Pre Processor found {0} files that"
                  " are smaller or equal to than {1} notes".format(
                len(self.__small_files_paths),
                MIDI_CONSTANTS.INPUT_SEQUENCE_LEN))

            print("Displaying all small songs:\n")
            for song in self.__small_files_paths:
                print("\t", song.split("/", 6)[-1])
            print()

            display_options_menu(menu_intro="Small files found!\n"
                                            "\tIt's fine if you don't delete"
                                            " them.Just know the pre-processor"
                                            " will not use them at all.",
                                 menu_options={1: "Delete all small files",
                                               2: "Ignore"})
            user_input = input("\nInput: ")

            # Remove small files
            if user_input == "1":
                self.delete_small_files()
            else:
                pass
        # ---------------------------------------------

        # Marks files to be selected for validation
        self.__generate_validation_files()

    def __thread_pool_datasets_reader(self, func,
                                      path_to_full_data_set,
                                      genre_sub_sample_set):
        """
            Thread pools out the dataset by genre
        """

        # Get all folder paths for each genre
        all_train_datasets_paths = [x[0] for x in os.walk(
            path_to_full_data_set)]
        all_train_datasets_paths.pop(0)

        all_files_by_genre = []

        for dataset_pth in all_train_datasets_paths:

            dataset_files = [dataset_pth + "/" + file for file in
                             glob.glob1(dataset_pth, "*.mid")][:genre_sub_sample_set]

            # Ensures files were actually extracted
            if len(dataset_files):
                self.__total_file_count += len(dataset_files)

                all_files_by_genre.append(dataset_files)

        # Init progress bar
        self.__pbar = tqdm(total=self.__total_file_count)

        # Begin threaded pool
        pool = ThreadPool(2000)
        all_results = pool.imap_unordered(func,
                                          all_files_by_genre)

        # End threaded pool
        pool.close()
        pool.join()
        self.__pbar.close()
        self.__pbar = None

        return all_results

    def read_midi_file(self, file):
        """
            Extract out the instruments/notes of the midi file.
        """
        # Attempt to parse midi file
        try:
            midi_data = pretty_midi.PrettyMIDI(file)

        # Midi file couldn't be opened; Raise flag; return dummy dict
        except:
            return {"flat_instr_note_seq": [],
                    "flat_instr_note_seq_len": 0,
                    "instruments": {},
                    "small_file_check": False,
                    "corrupted": True}

        # Stores instrument note pair
        flat_instr_note_seq = []

        file_instruments = set()

        # Move through midi file; store data on instrument/note relationship in
        # string
        for instr in midi_data.instruments:

            for note_obj in instr.notes:
                file_instruments.add("Program:" + str(instr.program) +
                                     INSTRUMENT_NOTE_SPLITTER.STR +
                                     "Is_Drum:" + str(instr.is_drum))

                flat_instr_note_seq.append(
                    ("Program:" + str(instr.program) + INSTRUMENT_NOTE_SPLITTER.STR
                     + "Is_Drum:" + str(instr.is_drum) + INSTRUMENT_NOTE_SPLITTER.STR
                     + "Note:" +
                     pretty_midi.note_number_to_name(note_obj.pitch),note_obj))

        # ---
        flat_instr_note_seq_len = len(flat_instr_note_seq)

        # File is to small for our neural networks to take; Raise flag;
        if flat_instr_note_seq_len <= MIDI_CONSTANTS.INPUT_SEQUENCE_LEN:
            return {"flat_instr_note_seq": flat_instr_note_seq,
                    "flat_instr_note_seq_len": flat_instr_note_seq_len,
                    "instruments": file_instruments,
                    "small_file_check": True,
                    "corrupted": False}

        # Sort notes in proper sequence based on their starting and end points
        flat_instr_note_seq.sort(key=lambda tup: (tup[1].start, tup[1].end))
        flat_instr_note_seq = [instr_note[0] for instr_note in
                              flat_instr_note_seq]

        # Return dict for more explict multi return type
        return {"flat_instr_note_seq": flat_instr_note_seq,
                "flat_instr_note_seq_len": flat_instr_note_seq_len,
                "instruments": file_instruments,
                "small_file_check": False,
                "corrupted": False}

    def __genre_dataset_init(self, genre_train_files):
        """
            Init full dataset attributes on MidiPreProcessor init
        """

        # Store meta data on file and genre specific data
        genre_instr_note_pairs = set()
        genre_instr_note_pairs_counter = Counter()
        genre_instruments = set()
        genre_file_meta_data = dict()
        genre_size = 0

        # Store invalid file paths
        corrupted_files = []
        small_files = []

        genre_name = genre_train_files[0].split("/")[-2].replace('_Midi', '')

        for _, file in enumerate(genre_train_files):

            # Update thread pool progress bar
            self.__pbar.update(1)
            self.__pbar.set_postfix_str(s=file.split("/", -1)[-1][:20],
                                        refresh=True)

            # Meta data on the file
            midi_file_attr = self.read_midi_file(file)

            # Check if flags were raised
            if midi_file_attr["corrupted"]:
                corrupted_files.append(file)
            elif midi_file_attr["small_file_check"]:
                small_files.append(file)

            # File passed requirements; store meta data on genre and file
            else:

                genre_instruments |= midi_file_attr["instruments"]
                genre_instr_note_pairs |= set(
                    midi_file_attr["flat_instr_note_seq"])
                genre_size += midi_file_attr["flat_instr_note_seq_len"]

                genre_file_meta_data[file] = {"flat_instr_note_seq":
                                                  midi_file_attr[
                                                      "flat_instr_note_seq"],
                                              "flat_instr_note_seq_len":
                                                  midi_file_attr[
                                                      "flat_instr_note_seq_len"],
                                              "instruments":
                                                  midi_file_attr[
                                                      "instruments"],}

                genre_instr_note_pairs_counter += Counter(midi_file_attr["flat_instr_note_seq"])

        return {"genre_name": genre_name,
                "genre_size": genre_size,
                "genre_instruments": genre_instruments,
                "genre_instr_note_pairs": genre_instr_note_pairs,
                "genre_instr_note_pairs_counter": genre_instr_note_pairs_counter,
                "genre_file_meta_data": {genre_name: genre_file_meta_data},
                "corrupted_files": corrupted_files,
                "small_files": small_files,}

    def __generate_validation_files(self):
        """
            Mark files for the validation set
        """

        self.__blacklisted_files_validation = set()

        # Find files for best fit the for the validation set per genre
        for genre_name, instr_note_counter in self.__genre_instr_note_counters.items():

            genre_note_count = sum(instr_note_counter.values())
            needed_validation_note_count = int(
                (genre_note_count / self.__total_intr_note_pair_size) \
                * genre_note_count)

            note_count_file_dict = {file_meta_data["flat_instr_note_seq_len"]: file_name
                                    for file_name, file_meta_data
                                    in self.__genre_file_dict[
                                        genre_name].items()}

            note_count_file_list = list(note_count_file_dict.keys())
            '''
            The validation count is decreasing per file note count;
            When it reaches this arbitrary threshold the validation
            set for this particular genre has been reached
            '''
            while True and needed_validation_note_count > 25:
                closest_file_note_count = find_nearest(
                    numbers=note_count_file_list,
                    target=needed_validation_note_count)
                needed_validation_note_count -= closest_file_note_count

                self.__blacklisted_files_validation.add(
                    note_count_file_dict[closest_file_note_count])

                note_count_file_list.remove(closest_file_note_count)

    def seq_train_test_split(self,
                         target_genre_name=False,
                         validation_set_required=True):
        """
            Returns back a training set and test set based on the input
            sizes of the models.
        """

        # Init train/test; input/output for models
        train_note_sequences = []
        train_output = []

        test_note_sequences = []
        test_output = []

        # Iterate through given genres and associated meta data on each file
        for genre_name, file_dict in self.__genre_file_dict.items():

            for file_path, instr_note_seq in file_dict.items():

                encoded_notes = [self.__master_note_encoder[instr_note_pair]
                                 for instr_note_pair in instr_note_seq]

                # Iterate by the input sequence length for each note sequnce;
                # Eliminate sequences that are under the input
                for i in range(0, len(encoded_notes) + 1,
                               MIDI_CONSTANTS.INPUT_SEQUENCE_LEN):
                    note_sequence = encoded_notes[i: i + MIDI_CONSTANTS.INPUT_SEQUENCE_LEN]

                    if len(note_sequence) == MIDI_CONSTANTS.INPUT_SEQUENCE_LEN:

                        try:

                            # Validation file found
                            if validation_set_required and file_path in self.__blacklisted_files_validation:

                                # Target is genre
                                if target_genre_name:
                                    test_output.append(self.__master_genre_encoder[genre_name])
                                # Target is a note
                                else:
                                    test_output.append(encoded_notes[i + MIDI_CONSTANTS.INPUT_SEQUENCE_LEN])

                                test_note_sequences.append(note_sequence)

                            # Must be a training file
                            else:

                                # Target is genre
                                if target_genre_name:
                                    train_output.append(self.__master_genre_encoder[genre_name])
                                # Target is a note
                                else:
                                    train_output.append(encoded_notes[i + MIDI_CONSTANTS.INPUT_SEQUENCE_LEN])

                                train_note_sequences.append(note_sequence)

                        # Input sequence was above but
                        except IndexError:
                            continue
                    else:
                        break

        return train_note_sequences, train_output, test_note_sequences, test_output

    # Delete the unused files from personal directory
    def delete_corrupted_files(self):
        for song in self.__corrupted_files_paths:
            os.remove(song)
        self.__corrupted_files_paths = []

    def delete_small_files(self):
        for song in self.__small_files_paths:
            os.remove(song)
        self.__small_files_paths = []

    # --- Getters
    def return_all_possible_instr_note_pairs(self):
        return copy.deepcopy(self.__all_possible_instr_note_pairs)

    def return_genre_instr_note_counters(self):
        return copy.deepcopy(self.__genre_instr_note_counters)

    def return_all_possible_instr_note_pairs_counter(self):
        return copy.deepcopy(self.__all_possible_instr_note_pairs_counter)

    # ----
    def return_all_instruments(self):
        return copy.deepcopy(self.__all_instruments)

    # ----
    def return_blacklisted_files_validation(self):
        return copy.deepcopy(self.__blacklisted_files_validation)

    def return_genre_file_dict(self):
        return copy.deepcopy(self.__genre_file_dict)

    # ----
    def return_corrupted_files_paths(self):
        return copy.deepcopy(self.__corrupted_files_paths)

    def return_small_files_paths(self):
        return copy.deepcopy(self.__small_files_paths)

    # ----
    def return_master_instr_note_encoder(self):
        return copy.deepcopy(self.__master_instr_note_encoder)

    def return_master_instr_note_decoder(self):
        return copy.deepcopy(self.__master_instr_note_decoder)

    # ----
    def return_master_instr_encoder(self):
        return copy.deepcopy(self.__master_instr_encoder)

    def return_master_instr_decoder(self):
        return copy.deepcopy(self.__master_instr_decoder)

    # ----
    def return_master_genre_encoder(self):
        return copy.deepcopy(self.__master_genre_encoder)

    def return_master_genre_decoder(self):
        return copy.deepcopy(self.__master_genre_decoder)

    # -------------------------------------------------
    # --- Setters
    def re_init_validation(self):
        pass
