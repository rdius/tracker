# coding: utf-8

import logging
from typing import Any
import smtplib

logger = logging.getLogger(__name__)


class SMTPClient:
    _sender_email: str
    _password: str
    _server_url: str
    _server_port: int
    _client = Any

    def __init__(
            self,
            sender_email: str,
            password: str,
            server_url: str,
            server_port: int,
        ):

        self._sender_email = sender_email
        self._server_url = server_url
        self._server_port = server_port
        self._password = password

        self._client = smtplib.SMTP(self._server_url, 587)
        self._client.starttls()
        self._client.login(self._sender_email, self._password)

    def send(self, to: str, message: str) -> None:
        self._client.sendmail(self._sender_email, to, message)
        self._client.quit()
