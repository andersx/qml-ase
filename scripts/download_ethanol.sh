#!/usr/bin/env bash

mkdir -p data
wget http://www.quantum-machine.org/gdml/data/npz/ethanol_ccsd_t.zip -O data/ethanol.zip
cd data
unzip ethanol.zip
rm -r __MACOSX/
cd ..
