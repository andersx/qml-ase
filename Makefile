
PYTHON=./env/bin/python
CONDA=conda

all: data env

data:
	bash scripts/download_ethanol.sh
	bash scripts/download_aspirin.sh

setup: env narupa-protocol install-narupa

env:
	${CONDA} env create -f environment.yml -p env
	${PYTHON} -m pip install numpy
	${PYTHON} -m pip install -r requirements.txt --no-cache-dir


narupa-protocol:
	git clone https://gitlab.com/intangiblerealities/narupa-protocol.git narupa-protocol

install-narupa:
	( \
		source activate ./env;\
		cd narupa-protocol;\
		./compile.sh --edit;
	)

# production

train_folder:
	${PYTHON} train_csv.py csv_data

train:
	${PYTHON} train.py ethanol --correct

test:
	${PYTHON} test.py

run_vr_dft:
	${PYTHON} narupa-qml.py --model csv_data --mol examples/ethanol.xyz --temp 150

run_vr:
	${PYTHON} narupa-qml.py --temp 250

run_vr_aspirin:
	${PYTHON} narupa-qml.py --temp 250 --model aspirin --mol examples/aspirin.xyz

run_opt:
	${PYTHON} optimize.py

run_md:
	${PYTHON} molecular_dynamics.py

clean:
	echo "not"

super-clean:
	rm -fr data env narupa-protocol __pycache__

