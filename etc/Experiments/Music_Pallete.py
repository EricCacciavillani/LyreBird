import pretty_midi
import sys
import numpy as np
from tqdm import tqdm
from collections import Counter
sys.path.append('..')

from Pre_Production.Midi_Pre_Processor import *
from Shared_Files.Global_Util import *

class MusicPallete:

    def __init__(self, pre_processor_obj):
        self.__all_possible_instr_note_pairs = pre_processor_obj.return_all_possible_instr_note_pairs()

        self.__centroid_instr_note = None
        self.__instr_note_pair_attributes = dict()
        set_test_eval = set()

        full_matrix = []
        count = 0
        for instr_note_pair_str in (self.__all_possible_instr_note_pairs):

            if "True" in instr_note_pair_str:
                count += 1

            instr_note_pair_obj = convert_string_to_instr_note_pair(instr_note_pair_str)

            # Evaluate instr/note pair
            fluid_synth, synth, note_hz = extract_instr_note_pair_attributes(instr_note_pair_obj)
            # print(fluid_synth)
            # ---
            print(fluid_synth)

            if self.__centroid_instr_note is None:
                self.__centroid_instr_note = fluid_synth

            self.__instr_note_pair_attributes[
                instr_note_pair_str] = float(calculate_wave_mse(self.__centroid_instr_note, fluid_synth))

            set_test_eval.add(calculate_wave_mse(self.__centroid_instr_note, fluid_synth))

        print(len(set_test_eval))
        print(count)

        cnt = Counter([v for k,v in self.__instr_note_pair_attributes.items()])

        print(cnt.most_common()[:20])
        print()

        print([k for k,v in self.__instr_note_pair_attributes.items() if v in set([test[0] for test in cnt.most_common()[:20]])])