cd %~dp0

pip3 install Flask
py -m venv venv

cd venv/Scripts && ^
activate && ^
cd ../.. && ^
pip3 install -r requirements.txt && ^
set FLASK_ENV=development && ^
set FLASK_APP=app.py && ^
set FLASK_DEBUG = 0 && ^
py -m flask run