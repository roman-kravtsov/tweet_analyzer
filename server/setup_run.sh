#!/bin/bash

sudo apt install -y python3-venv

pip install Flask
python3 -m venv venv && source venv/bin/activate && \
pip install -r requirements.txt && \

export FLASK_ENV=development && \
export FLASK_APP=app.py && \
export FLASK_DEBUG=0 && \

flask run