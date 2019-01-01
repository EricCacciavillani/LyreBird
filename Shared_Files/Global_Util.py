from twilio.rest import Client
import os
import numpy as np
import itertools
import pretty_midi
import math

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
                              velocity_list=[],
                              default_offset=.45,
                              default_velocity=100):
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

        tokenized_str = instr_note_pair.split("-:-")

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

    return full_song


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