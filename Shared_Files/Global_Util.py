from twilio.rest import Client
import os
import numpy as np
import itertools
import pretty_midi
import math
import sys
sys.path.append('..')

from Shared_Files.Global_Util import *
from Shared_Files.Constants import *

# --------------- Numeric Handling ---------------
def find_nearest(numbers, target):
    """
        Find the closest fitting number to the target number
    """
    numbers = np.asarray(numbers)
    idx = (np.abs(numbers - target)).argmin()
    return numbers[idx]


def find_closest_sum(numbers, target):
    """
        Calculates all possible sums and find the closest sum.
        This is a inefficient way of doing this for large scale numbers.
    """
    numbers = numbers[:]

    if not numbers:
        return None
    combs = sum([list(itertools.combinations(numbers, r))
                 for r in range(1, len(numbers)+1)], [])
    sums = np.asarray(list(map(sum, combs)))

    return combs[np.argmin(np.abs(np.asarray(sums) - target))]
# ---------------------------------------

# --------------- Midi Handling ---------------
def create_pretty_midi_object(input_seq,
                              instr_decoder_obj=None,
                              note_start_end_list=[],
                              default_offset=DEFAULT_NOTE_CONSTANTS.OFFSET,
                              default_velocity=DEFAULT_NOTE_CONSTANTS.VELOCITY,
                              velocity_list=[]):
    """
        Takes a input_seq of instrument/note pairs and converts them into a
        pretty midi object.
    """

    # A 'note_start_end_list' was not provided; create one using default values
    if not note_start_end_list:
        note_start_end_list = [(i, float(i + default_offset))
                               for i in np.arange(0, math.ceil(default_offset
                                                               * len(input_seq)),
                                                  default_offset)]

    # The 'note_start_end_list' was not long enough for the input sequence;
    # fill in the rest with default values
    elif len(note_start_end_list) < len(input_seq):
        note_start_end_list += [(i, float(i + default_offset))
                                for i in np.arange(0,math.ceil(default_offset
                                                               * (len(input_seq) - len(velocity_list))), default_offset)]
    # ----

    # A 'velocity_list' was not provided; create one using default values
    if not velocity_list:
        velocity_list = [default_velocity
                         for _ in range(0, len(input_seq))]

    # The 'velocity_list' was not long enough for the input sequence;
    # fill in the rest with default values
    elif len(velocity_list) < len(input_seq):
        velocity_list += [default_velocity
                          for i in range(0,
                                          (len(input_seq) - len(velocity_list)))]

    # Encoder object passed; assume that 'input_seq' needs to be decoded
    if instr_decoder_obj:
        input_seq = [instr_decoder_obj[instr_note_pair]
                     for instr_note_pair in input_seq]

    # Stores all needed midi instruments
    instrument_dict = dict()

    # Iterate through the input sequence in proper order;
    # relate instruments with proper notes, velocities, and start/end points.
    for index, instr_note_pair in enumerate(input_seq):

        tokenized_str = instr_note_pair.split(INSTRUMENT_NOTE_SPLITTER.STR)

        # ----
        instrument = tokenized_str[0] + tokenized_str[1]

        # A new instrument was found
        if instrument not in instrument_dict.keys():

            # Convert string data to proper midi info
            program_number = int(tokenized_str[0].split(":")[1])
            is_drum = (tokenized_str[1].split(":")[1] == "True")

            # Relate to proper instrument
            instrument_dict[instrument] = pretty_midi.Instrument(
                program=program_number, is_drum=is_drum)

        pitch = pretty_midi.note_name_to_number(tokenized_str[2].split(":")[1])

        # Add note to instrument
        instrument_dict[instrument].notes.append(pretty_midi.Note(
            velocity=velocity_list[index], pitch=pitch,
            start=note_start_end_list[index][0], end=note_start_end_list[index][1]))

    # Generate midi object with instrument based data
    full_song = pretty_midi.PrettyMIDI()
    full_song.instruments = [instr for instr in instrument_dict.values()]
    #
    # for test_intr in full_song.instruments:
    #     print("program:", test_intr.program)
    #     print("isdrum:",test_intr.is_drum)
    #     print("piano:",test_intr.get_piano_roll())
    #     print("histogram:",(test_intr.get_pitch_class_histogram())[0])
    #     print("Notes:",len(test_intr.notes))
    #     print("histog:", test_intr.get_pitch_class_histogram().sum())
    #     print("get_chroma:",test_intr.get_chroma().sum())
    #     print("transition_matrix",test_intr.get_pitch_class_transition_matrix().sum())
    #     print("syn:",test_intr.synthesize())
    #     print(test_intr.get_end_time())
    #     print()

    return full_song



def get_instr_note_dict(instr_note_str):
    """
        Must have at the program number, is_drum, and the note name
        in the string.
    """

    return {comp_part.split(":")[0]: comp_part.split(":")[1] for comp_part in
                       instr_note_str.split(INSTRUMENT_NOTE_SPLITTER.STR)}



def convert_string_to_instr_note_pair(instr_note_str):

    instr_note_dict = get_instr_note_dict(instr_note_str)

    instr_obj = pretty_midi.Instrument(program=int(instr_note_dict["Program"]),
                                       is_drum=(instr_note_dict["Is_Drum"] == "True"))

    pitch_num = pretty_midi.note_name_to_number(instr_note_dict["Note"])

    note_obj = pretty_midi.Note(velocity=DEFAULT_NOTE_CONSTANTS.VELOCITY,
                                pitch=pitch_num,
                                start=0,
                                end=0+DEFAULT_NOTE_CONSTANTS.OFFSET)

    instr_obj.notes.append(note_obj)

    return instr_obj

def extract_instr_note_pair_attributes(instr_note_obj):
    return instr_note_obj.fluidsynth(), instr_note_obj.synthesize(), \
           instr_note_obj.notes[0].pitch


def calculate_wave_mse(wave_form_a,wave_form_b):
    return np.sqrt(np.mean((wave_form_a - wave_form_b) ** 2))
# --------------- Misc ---------------
def send_sms_to_me(message="You forget to add a message!"):
    """
        Send a sms text message to a phone using twilo.
    """
    account_sid = os.environ.get('TWILIO_ACCOUNT_SID')
    auth_token = os.environ.get('TWILIO_AUTH_TOKEN')


    client = Client(account_sid, auth_token)

    client.messages.create(from_=os.environ.get('TWILIO_PHONE_NUMBER'),
                           to=os.environ.get('MY_PHONE_NUMBER'),
                           body=message)


def display_options_menu(menu_intro="", menu_options={}):

    print(menu_intro)
    for input_option, text in menu_options.items():
        print("\t*{0} {1}".format(input_option, text))
# ---------------------------------------