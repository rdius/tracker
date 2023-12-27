# coding: utf-8

from environs import Env

env = Env()
env.read_env()


# [ NOTIFICATION_RECIPIENT ]
NOTIFICATION_RECIPIENT_PHONE = env.str('NOTIFICATION_RECIPIENT_PHONE', 'recipient_phone_number')
NOTIFICATION_RECIPIENT_EMAIL = env.str('NOTIFICATION_RECIPIENT_EMAIL', 'recipient_email')

# [ TWILIO ]
TWILIO_PHONE_NUMBER = env.str('TWILIO_PHONE_NUMBER', 'your_twilio_virtul_phonr_number')
TWILIO_ACCOUNT_SID = env.str('TWILIO_ACCOUNT_SID', 'twilio_account_sid')
TWILIO_AUTH_TOKEN = env.str('TWILIO_AUTH_TOKEN', 'twilio_auth_token')

# [ EMAIL WITH SMTP SERVER]
# how to enable smt with gmail -> https://www.gmass.co/blog/gmail-smtp/
SMTP_SENDER_EMAIL = env.str('SMTP_SENDER_EMAIL', 'sender@mail.com')
SMTP_SENDER_PASSWORD = env.str('SMTP_SENDER_PASSWORD', 'your_password')
SMTP_SERVER_URL = env.str('SMTP_SERVER_URL', 'smtp.gmail.com')  # using gmail by default
SMTP_SERVER_PORT = env.int('SMTP_SERVER_PORT', 587)
