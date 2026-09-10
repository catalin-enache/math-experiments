## RUN

python -m venv .venv
source .venv/bin/activate

python -m pip install requests flask

python -m pip freeze > requirements.txt

python -m pip install -r requirements.txt
python -m pip install --upgrade -r requirements.txt
