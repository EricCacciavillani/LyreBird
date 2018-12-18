from twilio.rest import Client
import os

def send_sms_to_me(message="You forget to add a messag!"):
    account_sid = os.environ.get('TWILIO_ACCOUNT_SID')
    auth_token = os.environ.get('TWILIO_AUTH_TOKEN')


    client = Client(account_sid, auth_token)

    client.messages.create(from_=os.environ.get('TWILIO_PHONE_NUMBER'),
                           to=os.environ.get('MY_PHONE_NUMBER'),
                           body=message)

def extract_genre_name_from_dir(dir_str):
    return (dir_str.split('/', -1)[-1]).replace('_Midi', '')