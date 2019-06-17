#!/usr/bin/env bash

mkdir -p data
wget http://www.quantum-machine.org/gdml/data/npz/aspirin_ccsd.zip -O data/aspirin.zip
cd data
unzip aspirin.zip
mv aspirin_ccsd-train.npz aspirin-train.npz
mv aspirin_ccsd-test.npz aspirin-test.npz
rm -r __MACOSX/
cd ..
