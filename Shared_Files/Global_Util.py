from twilio.rest import Client
import os
import numpy as np
import itertools
import pretty_midi
import math
import sys
from tqdm import tqdm
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

def unique_rows(a):
    a = np.ascontiguousarray(a)
    unique_a = np.unique(a.view([('', a.dtype)]*a.shape[1]))
    return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))
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
            program_number = int(tokenized_str[0].split(PARAMETER_VAL_SPLITTER.STR)[1])
            is_drum = (tokenized_str[1].split(PARAMETER_VAL_SPLITTER.STR)[1] == "True")

            # Relate to proper instrument
            instrument_dict[instrument] = pretty_midi.Instrument(
                program=program_number, is_drum=is_drum)

        pitch = pretty_midi.note_name_to_number(tokenized_str[2].split(PARAMETER_VAL_SPLITTER.STR)[1])

        # Add note to instrument
        instrument_dict[instrument].notes.append(pretty_midi.Note(
            velocity=velocity_list[index], pitch=pitch,
            start=note_start_end_list[index][0], end=note_start_end_list[index][1]))

    # Generate midi object with instrument based data
    full_song = pretty_midi.PrettyMIDI()
    full_song.instruments = [instr for instr in instrument_dict.values()]

    return full_song

def calculate_wave_mse(wave_form_a,
                       wave_form_b):
    return np.sqrt(np.mean((wave_form_a - wave_form_b) ** 2))



def get_instr_note_dict(instr_note_str):
    """
        Must have at the program number, is_drum, and the note name
        in the string.
    """
    instr_note_dict = {comp_part.split(PARAMETER_VAL_SPLITTER.STR)[0]: comp_part.split(PARAMETER_VAL_SPLITTER.STR)[1]
                       for comp_part in instr_note_str.split(INSTRUMENT_NOTE_SPLITTER.STR)}

    instr_note_dict["Is_Drum"] = (instr_note_dict["Is_Drum"] == "True")

    return instr_note_dict

def convert_string_to_instr_note_pair(instr_note_str):

    instr_note_dict = get_instr_note_dict(instr_note_str)
    instr_obj = pretty_midi.Instrument(program=int(instr_note_dict["Program"]),
                                       is_drum=(instr_note_dict["Is_Drum"]))

    pitch_num = pretty_midi.note_name_to_number(instr_note_dict["Note"])

    note_obj = pretty_midi.Note(velocity=DEFAULT_NOTE_CONSTANTS.VELOCITY,
                                pitch=pitch_num,
                                start=0,
                                end=0+DEFAULT_NOTE_CONSTANTS.OFFSET)

    instr_obj.notes.append(note_obj)

    return instr_obj


def convert_string_to_instr_obj(instr_str):

    instr_note_dict = get_instr_note_dict(instr_str)
    instr_obj = pretty_midi.Instrument(program=int(instr_note_dict["Program"]),
                                       is_drum=(instr_note_dict["Is_Drum"] == "True"))

    return instr_obj

def get_instr_wave_forms(instrument_name_contains,
                         all_instruments,
                         instr_note_pairs_dict,
                         unique_matrix=False):
    instr_wave_forms = dict()

    pbar = tqdm(instrument_name_contains.items())
    for find_intstr_name, is_drum in pbar:
        instr_wave_forms[find_intstr_name] = list()

        pbar.set_postfix_str(s=find_intstr_name, refresh=True)
        for instr_str in all_instruments:

            instr_note_dict = get_instr_note_dict(instr_str)
            # Find only instruments that are drums
            if is_drum and instr_note_dict["Is_Drum"]:
                instr_wave_forms[find_intstr_name] += [convert_string_to_instr_note_pair(instr_note_pair).fluidsynth(
                    FLUID_SYNTH_CONSTANTS.SAMPLING_RATE)
                    for instr_note_pair in instr_note_pairs_dict[instr_str]]

            # Find only instruments that are NOT drums
            elif is_drum == False and instr_note_dict["Is_Drum"] == False and get_instr_name(instr_str).find(
                    find_intstr_name) != -1:
                instr_wave_forms[find_intstr_name] += [convert_string_to_instr_note_pair(instr_note_pair).fluidsynth(
                    FLUID_SYNTH_CONSTANTS.SAMPLING_RATE)
                    for instr_note_pair in instr_note_pairs_dict[instr_str]]

        instr_wave_forms[find_intstr_name] = np.array(instr_wave_forms[find_intstr_name])

    if unique_matrix:
        for instr, waves in instr_wave_forms.items():
            instr_wave_forms[instr] = unique_rows(waves)

    return instr_wave_forms

def get_instr_name(instr):
    return pretty_midi.program_to_instrument_class(convert_string_to_instr_obj(instr).program)

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