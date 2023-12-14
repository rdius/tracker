# Description

This Python package includes tools for sending SMS and emails. It is designed to be easy to use and configure, and it offers a variety of features to meet the needs of developers of all levels.

# Supported platforms 

### SMS

- [x] Twilio
- [ ] Wirepick

### E-mail
- [x] Gmail
- [x] SMTP

# Settings

This file contains all configurations settings for the package. It is located in the ``./settings.py`` file. You can directly modify this file to set the credential values.

## Using .venv environment variables file

Create a file named **.env** in the project root folder and set environment variables like below:

```bash
NOTIFICATION_RECIPIENT_PHONE='+XXXXXXXXXXX'
NOTIFICATION_RECIPIENT_EMAIL='recipeint@email.test'

TWILIO_PHONE_NUMBER='+XXXXXXXXX'
TWILIO_ACCOUNT_SID='abcdefghijklmnopqrstuvwxyz'
TWILIO_AUTH_TOKEN='0123qbcdefghti'

SMTP_SENDER_EMAIL='user@email.test'
SMTP_SENDER_PASSWORD='password'
SMTP_SERVER_URL='smtp.gmail.com'
SMTP_SERVER_PORT=587
```

# Usage

```python
# from a python script in the projet


# import
from notifications import twilio_api
from notifications import smtp_client
from notifications.settings import NOTIFICATION_RECIPIENT_PHONE
from notifications.settings import NOTIFICATION_RECIPIENT_EMAIL


# send sms using twilio
twilio_api.send(
    to=NOTIFICATION_RECIPIENT_PHONE,
    message='your message'
)

# send email
smtp_client.send(
    to=NOTIFICATION_RECIPIENT_EMAIL,
    message='your message'
)

``````







