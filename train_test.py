#!/usr/bin/env python3

import sys
import os

import numpy as np
np.random.seed(666)

from copy import deepcopy
import scipy
import pandas
import ast

import qml

from qml.math import svd_solve

from qml.representations import generate_fchl_acsf

from qml.kernels import get_atomic_local_kernel
from qml.kernels import get_atomic_local_gradient_kernel


FILENAME_TEST = "data/ethanol_ccsd_t-test.npz"
FILENAME_TRAIN = "data/ethanol_ccsd_t-train.npz"

MAX_ATOMS = 25

def get_data_from_file(filename, n=100):

    data = np.load(filename)

    X  = []
    dX = []
    Q  = []

    E  = []
    F  = []

    max_n = len(data["E"])

    index = np.random.choice(max_n, size=n, replace=False)

    nuclear_charges = data["z"]
    # max_atoms = len(nuclear_charges)

    for i in index:

        coordinates = data["R"][i]
        
        (rep, drep) = generate_fchl_acsf(nuclear_charges, coordinates,
               gradients=True, pad=MAX_ATOMS, elements=[1,6,8])

        X.append(rep)
        dX.append(drep)
        Q.append(nuclear_charges)
        E.append(data["E"][i])
        F.append(data["F"][i])

        # print(coordinates)
        # print(data["E"][i])
        # print(data["F"][i])


    X  = np.array(X)
    dX = np.array(dX)
    E  = np.array(E).flatten()
    F  = np.array(F)
    
    return X, dX, Q, E, F


# def get_data_from_csv(dirname, n=100):

def csv_to_reps(csv_filename, n=32):

    # max_atoms = 12 # HARDCODED for ETHANOL

    df = pandas.read_csv(csv_filename, sep=";|")


    max_n = len(df["atomization_energy"])
    n = min(max_n, n)
    index = np.random.choice(max_n, size=n, replace=False)
    
    print(csv_filename, max_n)

    X  = []
    dX = []
    Q  = []

    E  = []
    F  = []


    for i in index:

        coordinates = np.array(ast.literal_eval(df["coordinates"][i]))
        nuclear_charges = np.array(ast.literal_eval(df["nuclear_charges"][i]), dtype=np.int32)
        atomtypes = ast.literal_eval(df["atomtypes"][i])

        force = np.array(ast.literal_eval(df["forces"][i]))
        energy = float(df["atomization_energy"][i])

        # HACK
        new_cut = 4.0

        cut_parameters = {
                "rcut": new_cut,
                "acut": new_cut,
                # "nRs2": int(24 * new_cut / 8.0),
                # "nRs3": int(20 * new_cut / 8.0),
        }

        (rep, drep) = generate_fchl_acsf(nuclear_charges, coordinates,
               gradients=True, pad=MAX_ATOMS, elements=[1,6,8], **cut_parameters)

        X.append(rep)
        dX.append(drep)
        Q.append(nuclear_charges)
        E.append(energy)
        F.append(force)

    X  = np.array(X)
    dX = np.array(dX)
    E  = np.array(E).flatten()
    #  = np.concatenate(F)

    return X, dX, Q, E, F


def csvdir_to_reps(dirname):

    csv_files = os.listdir(dirname) 

    X  = []
    dX = []
    Q  = []

    E  = []
    F  = []

    for f in csv_files:
    
        X1,  dX1,  Q1,  E1,  F1  = csv_to_reps(dirname + "/" + f)

        X.append(X1)
        dX.append(dX1)
        Q += Q1
        E.append(E1)
        F +=F1 

    X = np.concatenate(X)
    dX = np.concatenate(dX)
    E = np.concatenate(E)


    return X, dX, Q, E, F


def test_fchl_acsf_operator_ccsd():

    SIGMA = 10.0

    X,  dX,  Q,  E,  F  = get_data_from_file(FILENAME_TRAIN, n=40)
    Xs, dXs, Qs, Es, Fs = get_data_from_file(FILENAME_TEST, n=20)

    offset = E.mean()
    E  -= offset
    Es -= offset

    print("Representations ...")
    F = np.concatenate(F)
    Fs = np.concatenate(Fs)


    print("Kernels ...")
    Kte = get_atomic_local_kernel(X, X,  Q, Q,  SIGMA)
    Kse = get_atomic_local_kernel(X, Xs, Q, Qs, SIGMA)

    Kt = get_atomic_local_gradient_kernel(X, X,  dX,  Q, Q,  SIGMA)
    Ks = get_atomic_local_gradient_kernel(X, Xs, dXs, Q, Qs, SIGMA)

    C = np.concatenate((Kte, Kt))
    Y = np.concatenate((E, F.flatten()))

    print("Alphas operator ...")
    alpha = svd_solve(C, Y, rcond=1e-11)

    eYt = np.dot(Kte, alpha)
    eYs = np.dot(Kse, alpha)

    fYt = np.dot(Kt, alpha)
    fYs = np.dot(Ks, alpha)


    print("===============================================================================================")
    print("====  OPERATOR, FORCE + ENERGY  ===============================================================")
    print("===============================================================================================")

    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(E, eYt)
    print("TRAINING ENERGY   MAE = %10.4f  slope = %10.4f  intercept = %10.4f  r^2 = %9.6f" % \
            (np.mean(np.abs(E - eYt)), slope, intercept, r_value ))

    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(F.flatten(), fYt.flatten())
    print("TRAINING FORCE    MAE = %10.4f  slope = %10.4f  intercept = %10.4f  r^2 = %9.6f" % \
             (np.mean(np.abs(F.flatten() - fYt.flatten())), slope, intercept, r_value ))

    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(Es.flatten(), eYs.flatten())
    print("TEST     ENERGY   MAE = %10.4f  slope = %10.4f  intercept = %10.4f  r^2 = %9.6f" % \
            (np.mean(np.abs(Es - eYs)), slope, intercept, r_value ))

    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(Fs.flatten(), fYs.flatten())
    print("TEST     FORCE    MAE = %10.4f  slope = %10.4f  intercept = %10.4f  r^2 = %9.6f" % \
            (np.mean(np.abs(Fs.flatten() - fYs.flatten())), slope, intercept, r_value ))


def test_fchl_acsf_operator_dft():

    SIGMA = 10.0


    Xall,  dXall,  Qall,  Eall,  Fall  = csvdir_to_reps("csv_data")

    idx = list(range(len(Eall)))
    np.random.shuffle(idx)

    print(len(idx))
    train = idx[:100]
    test = idx[100:]
    print("train = ", len(train), "      test = ", len(test))
   
    X  = Xall[train]
    dX = dXall[train]
    Q  = [Qall[i] for i in train]
    E  = Eall[train]
    F  = [Fall[i] for i in train]

    Xs  = Xall[test]
    dXs = dXall[test]
    Qs  =  [Qall[i] for i in test]
    Es  = Eall[test]
    Fs  = [Fall[i] for i in test]


    print("Representations ...")
    F = np.concatenate(F)
    Fs = np.concatenate(Fs)

    print("Kernels ...")
    Kte = get_atomic_local_kernel(X, X,  Q, Q,  SIGMA)
    Kse = get_atomic_local_kernel(X, Xs, Q, Qs, SIGMA)

    Kt = get_atomic_local_gradient_kernel(X, X,  dX,  Q, Q,  SIGMA)
    Ks = get_atomic_local_gradient_kernel(X, Xs, dXs, Q, Qs, SIGMA)

    C = np.concatenate((Kte, Kt))
    Y = np.concatenate((E, F.flatten()))

    print("Alphas operator ...")
    alpha = svd_solve(C, Y, rcond=1e-11)

    eYt = np.dot(Kte, alpha)
    eYs = np.dot(Kse, alpha)

    fYt = np.dot(Kt, alpha)
    fYs = np.dot(Ks, alpha)


    print("===============================================================================================")
    print("====  OPERATOR, FORCE + ENERGY  ===============================================================")
    print("===============================================================================================")

    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(E, eYt)
    print("TRAINING ENERGY   MAE = %10.4f  slope = %10.4f  intercept = %10.4f  r^2 = %9.6f" % \
            (np.mean(np.abs(E - eYt)), slope, intercept, r_value ))

    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(F.flatten(), fYt.flatten())
    print("TRAINING FORCE    MAE = %10.4f  slope = %10.4f  intercept = %10.4f  r^2 = %9.6f" % \
             (np.mean(np.abs(F.flatten() - fYt.flatten())), slope, intercept, r_value ))

    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(Es.flatten(), eYs.flatten())
    print("TEST     ENERGY   MAE = %10.4f  slope = %10.4f  intercept = %10.4f  r^2 = %9.6f" % \
            (np.mean(np.abs(Es - eYs)), slope, intercept, r_value ))

    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(Fs.flatten(), fYs.flatten())
    print("TEST     FORCE    MAE = %10.4f  slope = %10.4f  intercept = %10.4f  r^2 = %9.6f" % \
            (np.mean(np.abs(Fs.flatten() - fYs.flatten())), slope, intercept, r_value ))


def train_only():

    SIGMA = 10.0

    # Read training data from file
    X,  dX,  Q,  E,  F  = get_data_from_file(FILENAME_TRAIN, n=40)

    offset = E.mean()
    E  -= offset
    print(offset)

    F = np.concatenate(F)
    Y = np.concatenate((E, F.flatten()))

    print("Kernels ...")
    Kte = get_atomic_local_kernel(X, X,  Q, Q,  SIGMA)
    Kt = get_atomic_local_gradient_kernel(X, X,  dX,  Q, Q,  SIGMA)

    C = np.concatenate((Kte, Kt))

    print("Alphas operator ...")
    alpha = svd_solve(C, Y, rcond=1e-11)

    np.save("data/training_alphas.npy", alpha)
    np.save("data/training_Q.npy", Q)
    np.save("data/training_X.npy", X)


def predict_only():


    # Initialize training data (only need to do this once)
    alpha = np.load("data/training_alphas.npy")
    X     = np.load("data/training_X.npy")
    Q     = np.load("data/training_Q.npy")

    # Define a molecule
    nuclear_charges = np.array([6, 6, 8, 1, 1, 1, 1, 1, 1])
    coordinates = np.array(
            [[ 0.07230959,  0.61441211, -0.03115568],
             [-1.26644639, -0.27012846, -0.00720771],
             [ 1.11516977, -0.30732869,  0.06414394],
             [ 0.10673943,  1.44346835, -0.79573006],
             [-0.02687486,  1.19350887,  0.98075343],
             [-2.06614011,  0.38757505,  0.39276693],
             [-1.68213881, -0.60620688, -0.97804526],
             [-1.18668224, -1.07395366,  0.67075071],
             [ 1.37492532, -0.56618891, -0.83172035]])

    # Generate representation
    max_atoms = X.shape[1]
    (rep, drep) = generate_fchl_acsf(nuclear_charges, coordinates,
                    gradients=True, pad=max_atoms)

    # Put data into arrays
    Qs = [nuclear_charges]
    Xs = np.array([rep])
    dXs = np.array([drep])
    
    SIGMA = 10.0

    # Get kernels
    Kse = get_atomic_local_kernel(X, Xs, Q, Qs, SIGMA)
    Ks = get_atomic_local_gradient_kernel(X, Xs, dXs, Q, Qs, SIGMA)

    # Offset from training
    offset = -97084.83100465109

    # Energy prediction
    energy_predicted = np.dot(Kse, alpha)[0] + offset

    energy_true = -97086.55524903

    print("True energy      %16.4f kcal/mol" % energy_true)
    print("Predicted energy %16.4f kcal/mol" % energy_predicted)

    # Force prediction
    forces_predicted= np.dot(Ks, alpha).reshape((len(nuclear_charges),3))

    forces_true = np.array(
            [[-66.66673100,   2.45752385,  49.92224945],
             [-17.98600137,  68.72856500, -28.82689294],
             [ 31.88432927,   8.98739402, -18.11946195],
             [  4.19798833, -31.31692744,   8.12825145],
             [ 16.78395377, -24.76072606, -38.99054658],
             [  6.03046276,  -7.24928076,  -3.88797517],
             [ 17.44954868,   0.21604968,   8.56118603],
             [ 11.73901551, -19.38200606,  13.26191987],
             [ -3.43256595,   2.31940789,   9.95126984]])

    print("True forces [kcal/mol]")
    print(forces_true)
    print("Predicted forces [kcal/mol]")
    print(forces_predicted)

if __name__ == "__main__":

    # test_fchl_acsf_operator_ccsd()
    test_fchl_acsf_operator_dft()
    # train_only()
    # predict_only()
