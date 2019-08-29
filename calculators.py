import numpy as np
import time

from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from qml.kernels import (get_atomic_local_gradient_kernel,
                         get_atomic_local_kernel)
from qml.representations import generate_fchl_acsf

from rdkit import Chem
from rdkit.Chem import AllChem, ChemicalForceFields
from rdkit.Chem import rdmolfiles


def get_forcefield(molobj):

    ffprop = ChemicalForceFields.MMFFGetMoleculeProperties(molobj)
    forcefield = ChemicalForceFields.MMFFGetMoleculeForceField(molobj, ffprop) # 0.01 overhead

    return ffprop, forcefield


def run_forcefield(ff, steps, energy=1e-2, force=1e-3):
    """
    """

    try:
        status = ff.Minimize(maxIts=steps, energyTol=energy, forceTol=force)
    except RuntimeError:
        return 1

    return status


class QMLCalculator(Calculator):
    name = 'QMLCalculator'
    implemented_properties = ['energy', 'forces']

    def __init__(self, parameters, representations, charges, alphas, **kwargs):
        super().__init__(**kwargs)
        # unpack parameters
        offset = parameters["offset"]
        sigma = parameters["sigma"]

        self.set_model(alphas, representations, charges, offset, sigma)

        self.energy = 0.0

    def calculate(self, atoms: Atoms = None, properties=('energy', 'forces'),
                  system_changes=all_changes):

        if atoms is None:
            atoms = self.atoms
        if atoms is None:
            raise ValueError(
                'No ASE atoms supplied to calculation, and no ASE atoms supplied with initialisation.')

        self.query(atoms)

        if 'energy' in properties:
            self.results['energy'] = self.energy

        if 'forces' in properties:
            self.results['forces'] = self.forces

        return

    def set_model(self, alphas, representations, charges, offset, sigma):

        self.alphas = alphas
        self.repr = representations
        self.charges = charges

        # Offset from training
        self.offset = offset

        # Hyper-parameters
        self.sigma = sigma
        self.max_atoms = self.repr.shape[1]

        self.n_atoms = len(charges[0])

        return

    def query(self, atoms=None, print_time=True):

        if print_time:
            start = time.time()

        # kcal/mol til ev
        # kcal/mol/aangstrom til ev / aangstorm
        conv_energy = 0.0433635093659
        conv_force = 0.0433635093659

        coordinates = atoms.get_positions()
        nuclear_charges = atoms.get_atomic_numbers()
        n_atoms = coordinates.shape[0]


        new_cut = 4.0

        cut_parameters = {
                "rcut": new_cut,
                "acut": new_cut,
                # "nRs2": int(24 * new_cut / 8.0),
                # "nRs3": int(20 * new_cut / 8.0),
        }

        rep, drep = generate_fchl_acsf(
            nuclear_charges,
            coordinates,
            gradients=True,
            elements=[1,6,8],
            pad=self.max_atoms,
            **cut_parameters)

        # Put data into arrays
        Qs = [nuclear_charges]
        Xs = np.array([rep], order="F")
        dXs = np.array([drep], order="F")

        # Get kernels
        Kse = get_atomic_local_kernel(self.repr, Xs, self.charges, Qs, self.sigma)
        Ks = get_atomic_local_gradient_kernel(self.repr, Xs, dXs, self.charges, Qs, self.sigma)

        # Energy prediction
        energy_predicted = np.dot(Kse, self.alphas)[0] + self.offset
        self.energy = energy_predicted * conv_energy

        # Force prediction
        forces_predicted = np.dot(Ks, self.alphas).reshape((n_atoms, 3))
        self.forces = forces_predicted * conv_force

        if print_time:
            end = time.time()
            print("qml query {:7.3f}s {:10.3f} ".format(end-start, energy_predicted))

        return

    def get_potential_energy(self, atoms=None, force_consistent=False):

        energy = self.energy

        return energy

    def get_forces(self, atoms=None):

        self.query(atoms=atoms)
        forces = self.forces

        return forces


def get_calculator(dataname):

    datadir = "data/"
    ext = ".npy"

    filename_sigma = datadir + dataname + "_sigma" + ext
    filename_offset = datadir + dataname + "_offset" + ext
    filename_alphas = datadir + dataname + "_alphas" + ext
    filename_representations = datadir + dataname + "_X" + ext
    filename_charges = datadir + dataname + "_Q" + ext

    offset = np.load(filename_offset)
    sigma = np.load(filename_sigma)

    # LOAD AND SET MODEL
    parameters = {}
    parameters["offset"] = offset
    parameters["sigma"] = sigma

    alphas = np.load(filename_alphas)
    X = np.load(filename_representations)
    Q = np.load(filename_charges, allow_pickle=True)

    alphas = np.array(alphas, order="F")
    X = np.array(X, order="F")

    # SET CALCULATE CLASS
    calculator = QMLCalculator(parameters, X, Q, alphas)

    return calculator



class RdkitCalculator(Calculator):
    name = 'RdkitCalculator'
    implemented_properties = ['energy', 'forces']

    def __init__(self, molobj, **kwargs):
        super().__init__(**kwargs)

        self.set_model(molobj)
        self.energy = 0.0


    def calculate(self, atoms: Atoms = None, properties=('energy', 'forces'),
                  system_changes=all_changes):

        if atoms is None:
            atoms = self.atoms
        if atoms is None:
            raise ValueError(
                'No ASE atoms supplied to calculation, and no ASE atoms supplied with initialisation.')

        self.query(atoms)

        if 'energy' in properties:
            self.results['energy'] = self.energy

        if 'forces' in properties:
            self.results['forces'] = self.forces

        return

    def set_model(self, molobj):

        ffprop, ff = get_forcefield(molobj)
        conformer = molobj.GetConformer()

        self.conformer = conformer
        self.forcefield = ff

        coordinates = conformer.GetPositions()
        coordinates = np.asarray(coordinates)

        atoms = molobj.GetAtoms()
        atoms = [atom.GetAtomicNum() for atom in atoms]
        atoms = np.array(atoms)

        # start
        self.coordinates = coordinates
        self.nuclear_charges = atoms

        return


    def set_coordinates(self, coordinates):

        for i, pos in enumerate(coordinates):
            self.conformer.SetAtomPosition(i, pos)

        return


    def get_energy(self):

        energy = self.forcefield.CalcEnergy()

        return energy


    def query(self, atoms=None, print_time=True):

        if print_time:
            start = time.time()

        # kcal/mol til ev
        # kcal/mol/aangstrom til ev / aangstorm
        conv_energy = 0.0433635093659
        conv_force = 0.0433635093659

        coordinates = atoms.get_positions()
        nuclear_charges = atoms.get_atomic_numbers()
        n_atoms = coordinates.shape[0]

        self.set_coordinates(coordinates)

        # Energy prediction
        energy_predicted = self.get_energy()
        self.energy = energy_predicted * conv_energy

        # Force prediction
        forces_predicted = self.get_forces()
        self.forces = forces_predicted * conv_force

        if print_time:
            end = time.time()
            print("rdkit query {:7.4f}".format(end-start))

        return

    def get_potential_energy(self, atoms=None, force_consistent=False):

        energy = self.get_energy()

        return energy

    def get_forces(self, atoms=None):

        print(atoms)

        forces = self.forcefield.CalcGrad()
        forces = np.array(forces)
        # forces = forces.reshape(())

        return -forces



def read_sdffile(filename, remove_hs=False, sanitize=True):
    """
    """

    ext = filename.split(".")[-1]

    if ext == "sdf":

        suppl = Chem.SDMolSupplier(filename,
            removeHs=remove_hs,
            sanitize=sanitize)

    elif ext == "gz":

        fobj = gzip.open(filename)
        suppl = Chem.ForwardSDMolSupplier(fobj,
            removeHs=remove_hs,
            sanitize=sanitize)

    else:
        print("could not read file")
        quit()

    return suppl


def get_calculator_rdkit(filename):

    molobjs = read_sdffile(filename)
    molobjs = [molobj for molobj in molobjs]
    molobj = molobjs[0]

    calculator = RdkitCalculator(molobj)
    coordinates = calculator.coordinates
    nuclear_charges = calculator.nuclear_charges

    return nuclear_charges, coordinates, calculator


