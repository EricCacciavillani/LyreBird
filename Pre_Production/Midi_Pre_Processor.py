from music21 import converter, note, chord
from music21.midi import MidiException
import glob
import os
import copy
from collections import Counter
from multiprocessing.dummy import Pool as ThreadPool
from tqdm import tqdm

# Import shared files
import sys
sys.path.append('..')
from Shared_Files.Global_Util import *
from Shared_Files.Constants import *


'''
    Reads across multiple datasets stores meta data on each data set and files and
    stores them to be used for later.
'''
class MidiPreProcessor:

    def __init__(self, path_to_full_data_set,
                 validation_set=True,
                 genre_sub_sample_set=1000
                 ):

        # Progress-bar for threading-pool
        self.__pbar = None

        # Store all individual tones found
        self.__all_possible_notes = set()
        self.__all_possible_notes_counter = Counter()

        # Files to ignore for when splicing data into train/test
        self.__blacklisted_files_validation = set()

        # Stores all genres to another dict that stores
        # the corresponding file note size
        self.__all_genre_file_notes_dict = dict()

        # Stores the genre names with all possible note and associated counts
        self.__all_genre_note_counters = dict()

        # Stores all corrupted files found
        self.__found_corrupted_file_paths = []

        # Store files that are to small (Determined by the input sequence)
        self.__small_files_paths = []

        # Encoders for notes and genre names
        self.__master_note_encoder = dict()
        self.__master_note_decoder = dict()

        self.__master_genre_encoder = dict()
        self.__master_genre_decoder = dict()

        # ---
        self.__total_file_count = 0
        self.__total_note_count = 0

        # Thread pool out reading multiple files of each dataset
        thread_pool_results = self.__thread_pool_datasets_reader(
            self.__genre_dataset_init, path_to_full_data_set, genre_sub_sample_set)

        # Init data based on thread pool results
        for genre_count, genre_dataset_result in enumerate(thread_pool_results):

            self.__all_genre_file_notes_dict = {**self.__all_genre_file_notes_dict,
                                                **genre_dataset_result["genre_files_notes_dict"]}
            self.__found_corrupted_file_paths += genre_dataset_result["corrupted_files"]
            self.__small_files_paths += genre_dataset_result["small_files"]
            self.__all_genre_note_counters = {**self.__all_genre_note_counters,
                                              **genre_dataset_result[
                                                  "genre_note_counter"]}

            self.__all_possible_notes_counter += genre_dataset_result["genre_note_counter"][
                genre_dataset_result["genre_name"]]
            self.__total_note_count += genre_dataset_result["total_genre_notes"]

            self.__master_genre_encoder[genre_dataset_result["genre_name"]] = \
                genre_count + 1

        # Invert encoder dict
        self.__master_genre_decoder = {v: k for k, v
                                       in self.__master_genre_encoder.items()}
        self.__all_possible_notes = sorted(set(self.__all_possible_notes_counter.keys()))

        # Corrupted files were found.
        if self.__found_corrupted_file_paths:

            print("The Pre Processor found {0}".format(len(self.__found_corrupted_file_paths)))
            print("Displaying all corrupted songs:\n")
            for song in self.__found_corrupted_file_paths:
                print("\t", song.split("/", 6)[-1])
            print()

            display_options_menu(menu_intro="Corrupted files found!\n"
                                            "\tIt's fine if you don't delete"
                                            " them.Just know the pre-processor"
                                            " will not use them at all.",
                                 menu_options={1: "Delete all corrupted files",
                                               2: "Ignore"})
            user_input = input("\nInput: ")
            user_input = int(user_input)

            # Remove corrupted files
            if user_input == 1:
                self.delete_corrupted_files()
            else:
                pass

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
            user_input = int(user_input)

            # Remove small files
            if user_input == 1:
                self.delete_small_files()
            else:
                pass

        # Encode dict for nodes
        self.__master_note_encoder = dict()
        for i, tone in enumerate(self.__all_possible_notes):
            self.__master_note_encoder[tone] = i + 1

        # Decode dict for nodes
        self.__master_note_decoder = {v: k for k, v
                                      in self.__master_note_encoder.items()}

        # Mark files for validation set
        if validation_set:
            # Marks files to be selected for valadation
            self.__generate_validation_files()

    def return_core_atributes(self):

        return {"all_possible_notes": copy.deepcopy(
            self.__all_possible_notes),
            "all_possible_notes_counter": copy.deepcopy(
                self.__all_possible_notes_counter),
            "blacklisted_files_validation": copy.deepcopy(
                self.__blacklisted_files_validation),
            "all_genre_file_notes_dict": copy.deepcopy(
                self.__all_genre_file_notes_dict),
            "all_genre_note_counters": copy.deepcopy(
                self.__all_genre_note_counters),
            "found_corrupted_file_paths": copy.deepcopy(
                self.__found_corrupted_file_paths),
            "total_file_count": copy.deepcopy(self.__total_file_count),
            "total_note_count": copy.deepcopy(self.__total_note_count),
            "master_note_decoder": copy.deepcopy(self.__master_note_decoder),
            "master_note_encoder": copy.deepcopy(self.__master_note_encoder)}

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

    def read_midi_file_by_note(self, file):
        """
            Extract out the notes of the midi file.
        """

        song_notes = []

        # Attempt to parse midi file
        try:
            midi = converter.parse(file)
        except:
            # Midi file couldn't be opened
            return {"song_notes": [],
                    "note_count": [],
                    "small_file_check": False,
                    "corrupted": True}

        # Use the first track on the midi file (Multiple ones)
        midi = midi[MIDI_CONSTANTS.TRACK_INDEX]

        # Parse the midi file by the notes it contains
        notes_to_parse = midi.flat.notes

        # Convert and append notes to our list; store tone count
        for element in notes_to_parse:

            if isinstance(element, note.Note):
                tone = str(element.pitch)
                song_notes.append(tone)

            elif isinstance(element, chord.Chord):

                # Get the numerical representation
                tone = '.'.join(str(n) for n in element.normalOrder)
                song_notes.append(tone)

        song_notes_len = len(song_notes)
        if song_notes_len <= MIDI_CONSTANTS.INPUT_SEQUENCE_LEN:
            return {"song_notes": song_notes,
                    "note_count": song_notes_len,
                    "small_file_check": True,
                    "corrupted": False}

        return {"song_notes": song_notes,
                "note_count": song_notes_len,
                "small_file_check": False,
                "corrupted": False}

    def __genre_dataset_init(self, genre_train_files):
        """
            Init full dataset attributes on MidiPreProcessor init
        """

        all_genre_notes = []
        file_note_count_dict = dict()
        corrupted_files = []
        small_files = []

        genre_name = genre_train_files[0].split("/")[-2].replace('_Midi','')

        for _, file in enumerate(genre_train_files):

            self.__pbar.update(1)
            self.__pbar.set_postfix_str(s=file.split("/",-1)[-1][:20],
                                        refresh=True)
            midi_file_attr = self.read_midi_file_by_note(file)

            if midi_file_attr["corrupted"]:
                corrupted_files.append(file)
            elif midi_file_attr["small_file_check"]:
                small_files.append(file)
            else:
                file_note_count_dict[file] = midi_file_attr["song_notes"]
                all_genre_notes += midi_file_attr["song_notes"]

        return {"genre_name": genre_name,
                "genre_files_notes_dict": {genre_name: file_note_count_dict},
                "genre_note_counter": {genre_name: Counter(all_genre_notes)},
                "corrupted_files": corrupted_files,
                "small_files": small_files,
                "total_genre_notes": len(all_genre_notes)}

    # Delete the unused files from personal directory
    def delete_corrupted_files(self):
        for song in self.__found_corrupted_file_paths:
            os.remove(song)
        self.__found_corrupted_file_paths = []

    def delete_small_files(self):
        for song in self.__small_files_paths:
            os.remove(song)
        self.__small_files_paths = []

    def __generate_validation_files(self):
        """
            Mark files for the validation set
        """

        self.__blacklisted_files_validation = set()

        # Find files for best fit the for the validation set per genre
        for genre_name, note_counter in self.__all_genre_note_counters.items():

            genre_note_count = sum(note_counter.values())
            needed_validation_note_count = int(
                (genre_note_count / self.__total_note_count) \
                * genre_note_count)

            note_count_file_dict = {len(v): k for k, v
                                    in self.__all_genre_file_notes_dict[
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

    def train_test_split(self,
                         target_genre_name=False):
        """
            Returns back a training set and test set based on the input
            sizes of the models.
        """

        # Init train/test; input/output for models
        train_note_sequences = np.zeros(MIDI_CONSTANTS.INPUT_SEQUENCE_LEN)
        train_output = np.zeros(1)

        test_note_sequences = np.zeros(MIDI_CONSTANTS.INPUT_SEQUENCE_LEN)
        test_output = np.zeros(1)

        # Iterate through given genres and associated meta data on each file
        for genre_name, file_dict in self.__all_genre_file_notes_dict.items():

            for file_path, notes in file_dict.items():

                encoded_notes = [self.__master_note_encoder[tone] for tone
                                 in notes]

                # Iterate by the input sequence length for each note sequnce;
                # Eliminate sequences that are under the input
                for i in range(0, len(encoded_notes) + 1,
                               MIDI_CONSTANTS.INPUT_SEQUENCE_LEN):
                    note_sequence = np.array(encoded_notes[
                                             i: i + MIDI_CONSTANTS.INPUT_SEQUENCE_LEN])

                    if len(note_sequence) == MIDI_CONSTANTS.INPUT_SEQUENCE_LEN:

                        try:

                            # Validation file found
                            if file_path in self.__blacklisted_files_validation:

                                # Target is genre
                                if target_genre_name:
                                    test_output = np.append(test_output,
                                                            self.__master_genre_encoder[genre_name])
                                # Target is a note
                                else:
                                    test_output = np.append(
                                        test_output,
                                        encoded_notes[i + 50])

                                test_note_sequences = np.vstack(
                                    (test_note_sequences,
                                     note_sequence))

                            # Must be a training file
                            else:

                                # Target is genre
                                if target_genre_name:
                                    train_output = np.append(train_output,
                                                             self.__master_genre_encoder[
                                                                 genre_name])
                                # Target is a note
                                else:
                                    train_output = np.append(
                                        train_output,
                                        encoded_notes[i + 50])

                                train_note_sequences = np.vstack(
                                    (train_note_sequences,
                                     note_sequence))

                        # Input sequence was 50 but
                        except IndexError:
                            continue
                    else:
                        break

        # Get rid of dummy holder of zeros in beginning for init shape
        train_output = np.delete(train_output, 0, axis=0)
        train_note_sequences = np.delete(train_note_sequences, 0, axis=0)

        test_output = np.delete(test_output, 0, axis=0)
        test_note_sequences = np.delete(test_note_sequences, 0, axis=0)

        return train_note_sequences, train_output, test_note_sequences, test_output