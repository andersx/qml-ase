#!/usr/bin/env python3

import cheminfo

import sys

import numpy as np
import qml
import scipy
from qml.kernels import (get_atomic_local_gradient_kernel,
                         get_atomic_local_kernel)
from qml.math import svd_solve
from qml.representations import generate_fchl_acsf
import rmsd

from train_test import csvdir_to_reps

np.random.seed(666)




def train(dataname, n_train=100):

    SIGMA = 10.0

    # Read training data from file
    # X,  dX,  Q,  E,  F  = get_data_from_file(filename_train, n=n_train)

    Xall,  dXall,  Qall,  Eall,  Fall =  csvdir_to_reps(dataname)
    
    if  len(Eall) < n_train:
        print("Not enough training data for", n_train)
        exit()

    idx = list(range(len(Eall)))
    np.random.shuffle(idx)
   
    train = idx[:n_train]
   
    print(len(train))

    X  = Xall[train]
    dX = dXall[train]
    Q  = [Qall[i] for i in train]
    E  = Eall[train]
    F  = [Fall[i] for i in train]

    offset = 0.0
    print("OFFSET: ", offset)

    F = np.concatenate(F)
    Y = np.concatenate((E, F.flatten()))

    print("Generating Kernels ...")
    Kte = get_atomic_local_kernel(X, X,  Q, Q,  SIGMA)
    Kt = get_atomic_local_gradient_kernel(X, X,  dX,  Q, Q,  SIGMA)

    C = np.concatenate((Kte, Kt))

    print("Alphas operator ...")
    alpha = svd_solve(C, Y, rcond=1e-11)

    np.save("data/"+dataname+"_offset.npy", offset)
    np.save("data/"+dataname+"_sigma.npy", SIGMA)
    np.save("data/"+dataname+"_alphas.npy", alpha)
    np.save("data/"+dataname+"_Q.npy", Q, allow_pickle=True)
    np.save("data/"+dataname+"_X.npy", X)

    return

if __name__ == "__main__":

    import argparse
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument('--correct', action='store_true', help='Add penalty and correction molecules to kernel')
    parser.add_argument('dataname', metavar='NAME', type=str)

    args = parser.parse_args()

    if len(sys.argv[1:]) < 1:
        print("choose some data")
        quit()

    dataname = args.dataname

    train(dataname, n_train=150)

    # # Get sample from test
    # data = np.load("data/" + dataname + "-test.npz")
    # coordinates = data["R"][0]
    # nuclear_charges = data["z"]
    # nuclear_charges = [cheminfo.convert(x) for x in nuclear_charges]

    # txt = rmsd.set_coordinates(nuclear_charges, coordinates)
    # f = open("examples/"+dataname+".xyz", 'w')
    # f.write(txt)
    # f.close()

