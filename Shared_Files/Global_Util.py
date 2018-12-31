from twilio.rest import Client
import os
import numpy as np
import itertools
import pretty_midi

# --------------- Numeric Handling ---------------
def find_nearest(numbers, target):
    """
    """
    numbers = np.asarray(numbers)
    idx = (np.abs(numbers - target)).argmin()
    return numbers[idx]


def find_closest_sum(numbers, target):
    numbers = numbers[:]

    if not numbers:
        return None
    combs = sum([list(itertools.combinations(numbers, r))
                 for r in range(1, len(numbers)+1)], [])
    sums = np.asarray(list(map(sum, combs)))

    return combs[np.argmin(np.abs(np.asarray(sums) - target))]
# ---------------------------------------

# --------------- Midi Handling ---------------
def create_midi_object(input_seq,
                       instr_decoder_obj):

    if instr_decoder_obj:
        input_seq = [instr_decoder_obj[instr_note_pair]
                     for instr_note_pair in input_seq]

    instrument_dict = dict()
    counter = 0

    for instr_note_pair in input_seq:
        tokenized_str = instr_note_pair.split("-:-")

        instr_program_pair = tokenized_str[0] + tokenized_str[1]

        if instr_program_pair not in instrument_dict.keys():
            program_number = int(tokenized_str[0].split(":")[1])
            is_drum = (tokenized_str[1].split(":")[1] == "True")

            instrument_dict[instr_program_pair] = pretty_midi.Instrument(
                program=program_number, is_drum=is_drum)

        pitch = pretty_midi.note_name_to_number(tokenized_str[2].split(":")[1])
        instrument_dict[instr_program_pair].notes.append(pretty_midi.Note(
            velocity=100, pitch=pitch, start=counter, end=counter + .4))
        counter += .4

    full_song = pretty_midi.PrettyMIDI()
    full_song.instruments = [instr for instr in instrument_dict.values()]


# --------------- Misc ---------------
def send_sms_to_me(message="You forget to add a message!"):
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