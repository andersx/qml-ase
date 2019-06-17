#!/usr/bin/env bash

mkdir -p data
wget http://www.quantum-machine.org/gdml/data/npz/aspirin_ccsd.zip -O data/aspirin.zip
cd data
unzip aspirin.zip
rm -r __MACOSX/
cd ..
