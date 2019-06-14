
PYTHON=python3

all: data

data:
	bash download_data.sh

env:
	${PYTHON} -m venv env
	./env/bin/pip install numpy
	./env/bin/pip install -r requirements.txt
	
