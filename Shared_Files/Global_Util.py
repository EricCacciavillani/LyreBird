from twilio.rest import Client
import os
import numpy as np
import itertools


def send_sms_to_me(message="You forget to add a messag!"):
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


def find_nearest(numbers, target):
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