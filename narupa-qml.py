import time
import cheminfo
import ase
import numpy as np
import rmsd
from ase import units
from ase.io.trajectory import Trajectory
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase.md.nvtberendsen import NVTBerendsen
from ase.optimize import BFGS
from narupa.ase import ASEImdServer
from calculators import QMLCalculator, get_calculator

FILENAME_ALPHAS = "data/training_alphas.npy"
FILENAME_REPRESENTATIONS = "data/training_X.npy"
FILENAME_CHARGES = "data/training_Q.npy"

tmpdir = "_tmp_/"


def dump_xyz(atoms, filename):
    coordinates = atoms.get_positions()
    nuclear_charges = atoms.get_atomic_numbers()
    nuclear_charges = [cheminfo.convert(x) for x in nuclear_charges]

    x = rmsd.set_coordinates(nuclear_charges, coordinates)

    f = open(filename, 'w')
    f.write(x)
    f.close()

    return


def serve_md(nuclear_charges, coordinates, calculator=None, temp=None):
    """
    """

    if calculator is None:
        parameters = {}
        parameters["offset"] = -97084.83100465109
        parameters["sigma"] = 10.0
        alphas = np.load(FILENAME_ALPHAS)
        X = np.load(FILENAME_REPRESENTATIONS)
        Q = np.load(FILENAME_CHARGES)
        alphas = np.array(alphas, order="F")
        X = np.array(X, order="F")
        calculator = QMLCalculator(parameters, X, Q, alphas)


    # SET MOLECULE
    molecule = ase.Atoms(nuclear_charges, coordinates)
    molecule.set_calculator(calculator)

    # SET ASE MD
    # Set the momenta corresponding to T=300K
    MaxwellBoltzmannDistribution(molecule, 200 * units.kB)

    if temp is None:
        # We want to run MD with constant energy using the VelocityVerlet algorithm.
        dyn = VelocityVerlet(molecule, 1 * units.fs)  # 1 fs time step.

    else:
        dyn = NVTBerendsen(molecule, 1*units.fs, temp, 1*units.fs)

    # SET AND SERVE NARUPA MD
    imd = ASEImdServer(dyn)

    while True:
        imd.run(10)
        dump_xyz(molecule, tmpdir + "snapshot.xyz")

    return


def main():

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mol', action='store', default=None, help='Load molecule for live simulation', metavar="FILE")
    parser.add_argument('--model', action='store', default="ethanol", help='')
    parser.add_argument('--temp', action='store', default=None, help='')
    args = parser.parse_args()

    # input

    if args.mol is None:
        nuclear_charges = np.array([6, 6, 8, 1, 1, 1, 1, 1, 1])
        coordinates = np.array(
            [[0.07230959, 0.61441211, -0.03115568],
            [-1.26644639, -0.27012846, -0.00720771],
            [1.11516977, -0.30732869, 0.06414394],
            [0.10673943, 1.44346835, -0.79573006],
            [-0.02687486, 1.19350887, 0.98075343],
            [-2.06614011, 0.38757505, 0.39276693],
            [-1.68213881, -0.60620688, -0.97804526],
            [-1.18668224, -1.07395366, 0.67075071],
            [1.37492532, -0.56618891, -0.83172035]])
    else:

        nuclear_charges, coordinates = rmsd.get_coordinates_xyz(args.mol)

    # model and simulation

    if args.temp is not None:
        args.temp = float(args.temp)

    calculator = get_calculator(args.model)

    serve_md(nuclear_charges, coordinates, calculator=calculator, temp=args.temp)

    return


if __name__ == "__main__":
    main()
