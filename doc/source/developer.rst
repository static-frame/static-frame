


python3 -m venv .env-sf

source .env-sf/bin/activate

pip install -r requirements-test.txt

invoke test

