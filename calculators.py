import numpy as np
import time

from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from qml.kernels import (get_atomic_local_gradient_kernel,
                         get_atomic_local_kernel)
from qml.representations import generate_fchl_acsf


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

    def query(self, atoms=None):

        # kcal/mol til ev
        # kcal/mol/aangstrom til ev / aangstorm
        conv_energy = 0.0433635093659
        conv_force = 0.0433635093659

        coordinates = atoms.get_positions()
        nuclear_charges = atoms.get_atomic_numbers()

        rep, drep = generate_fchl_acsf(
            nuclear_charges,
            coordinates,
            gradients=True,
            pad=self.max_atoms)

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
        forces_predicted = np.dot(Ks, self.alphas).reshape((self.n_atoms, 3))
        self.forces = forces_predicted * conv_force

        return

    def get_potential_energy(self, atoms=None, force_consistent=False):

        # if self.energy == 0.0:
        #     self.query(atoms=atoms)

        energy = self.energy

        return energy

    def get_forces(self, atoms=None):

        self.query(atoms=atoms)
        forces = self.forces

        return forces
