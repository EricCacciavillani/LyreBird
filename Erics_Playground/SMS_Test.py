import os
from twilio.rest import Client


account_sid = os.environ.get('TWILIO_ACCOUNT_SID')
auth_token = os.environ.get('TWILIO_AUTH_TOKEN')


client = Client(account_sid, auth_token)

client.messages.create(from_=os.environ.get('TWILIO_PHONE_NUMBER'),
                       to=os.environ.get('MY_PHONE_NUMBER'),
                       body='SMS Working')
