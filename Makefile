
PYTHON=./env/bin/python
CONDA=conda

all: data env

data:
	bash download_data.sh

env:
	${CONDA} env create -f environment.yml -p env
	${PYTHON} -m pip install numpy
	${PYTHON} -m pip install -r requirements.txt --no-cache-dir

protocol-narupa:
	git clone --depth 1 https://gitlab.com/intangiblerealities/narupa-protocol.git narupa-protocol
	cd narupa-protocol; ./compile.sh --edit

activate: env
	source activate ./env

train:
	${PYTHON} train.py

test:
	${PYTHON} test.py

run_vr:
	${PYTHON} narupa-qml.py

run_opt:
	${PYTHON} optimize.py

run_md:
	${PYTHON} molecular_dynamics.py

clean:
	echo "not"

super-clean:
	rm -fr data env narupa-protocol __pycache__

