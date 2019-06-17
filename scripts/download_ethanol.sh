#!/usr/bin/env bash

mkdir -p data
wget http://www.quantum-machine.org/gdml/data/npz/ethanol_ccsd_t.zip -O data/ethanol.zip
cd data
unzip ethanol.zip
mv ethanol_ccsd_t-train.npz ethanol-train.npz
mv ethanol_ccsd_t-test.npz ethanol-test.npz
rm -r __MACOSX/
cd ..
