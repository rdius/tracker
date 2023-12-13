# coding: utf-8

import logging

from twilio.rest import Client


__all__ = [
    'TwilioAPI',
]

logger = logging.getLogger(__name__)


class TwilioAPI:
    _phone_number: str
    _account_sid: str
    _auth_token: str
    _client: Client

    def __init__(self, phone_number: str, account_sid: str, auth_token: str):
        self._phone_number = phone_number
        self._account_sid = account_sid
        self._auth_token = auth_token
        self._client = Client(account_sid, auth_token)
    
    def send(self, to: str, message: str) -> None:
        message = self._client.messages.create(
            to=to,
            from_=self._phone_number,
            body=message
        )
        logger.info(message)
