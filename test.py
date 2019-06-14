#!/usr/bin/env python3

import time

import numpy as np
import qml
import scipy
from qml.kernels import (get_atomic_local_gradient_kernel,
                         get_atomic_local_kernel)
from qml.math import svd_solve
from qml.representations import generate_fchl_acsf

np.random.seed(666)

# CONSTANTS

FILENAME_ALPHAS = "data/training_alphas.npy"
FILENAME_REPRESENTATIONS = "data/training_X.npy"
FILENAME_CHARGES = "data/training_Q.npy"

SIGMA = 10.0


def predict(nuclear_charges, coordinates):
    """

    Given a query molecule (charges and coordinates) predict energy and forces

    """

    # Initialize training data (only need to do this once)
    alpha = np.load(FILENAME_ALPHAS)
    X = np.load(FILENAME_REPRESENTATIONS)
    Q = np.load(FILENAME_CHARGES)


    # Generate representation
    max_atoms = X.shape[1]
    (rep, drep) = generate_fchl_acsf(nuclear_charges, coordinates,
                    gradients=True, pad=max_atoms)

    # Put data into arrays
    Qs = [nuclear_charges]
    Xs = np.array([rep])
    dXs = np.array([drep])

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

    return

if __name__ == "__main__":

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

    predict(nuclear_charges, coordinates)
