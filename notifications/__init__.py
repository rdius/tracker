# coding: utf-8

from . import settings
from .sms import TwilioAPI
from .email import SMTPClient

__all__ = [
    'twilio_api',
    'smtp_client',
]

twilio_api = TwilioAPI(
    phone_number=settings.TWILIO_PHONE_NUMBER,
    account_sid=settings.TWILIO_ACCOUNT_SID,
    auth_token=settings.TWILIO_AUTH_TOKEN,
)

smtp_client = SMTPClient(
    sender_email=settings.SMTP_SENDER_EMAIL,
    password=settings.SMTP_SENDER_PASSWORD,
    server_url=settings.SMTP_SERVER_URL,
    server_port=settings.SMTP_SERVER_PORT
)
