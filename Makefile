
PYTHON=python3

all: data env

data:
	bash download_data.sh

env:
	${PYTHON} -m venv env
	./env/bin/pip install numpy
	./env/bin/pip install -r requirements.txt


train:
	./env/bin/python train.py

test:
	./env/bin/python test.py

run:
	./env/bin/python molecular_dynamics.py


clean:
	echo "not"


super-clean:
	rm -r data env


