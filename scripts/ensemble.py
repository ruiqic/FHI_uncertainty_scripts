import random
import numpy as np
from ase.calculators.singlepoint import SinglePointCalculator as sp

# some functions modified from https://github.com/uncertainty-toolbox/uncertainty-toolbox
# and https://github.com/ulissigroup/uncertainty_benchmarking

def bootstrap(subset, n_ensembles):
    """
    subset: list
        training set
    n_ensembles: int
        the number of models
    
    return: list list
    """
    
    bootstrapped_data = []
    subset_len = len(subset)
    for i in range(n_ensembles):
        bootstrapped_data.append(random.choices(subset, k=subset_len))
    return bootstrapped_data

def subsample(subset, n_subsampled, n_ensembles):
    """
    subset: list
        training set
    n_subsampled: int
        the number of trainig data per subsample
    n_ensembles: int
        the number of models
    
    return: list list
    """
    
    subsampled_data = []
    for i in range(n_ensembles):
        subsampled_data.append(random.sample(subset, k=n_subsampled))
    return subsampled_data

def collate_atoms(atoms_list):
    """
    Combines a list of ase.Atoms into a single ase.Atoms with mean and stdev
    information about energies and forces.
    
    atoms_list: list
        list of ase.Atoms with the same order of structures
        expects fields:
            .info["energy"]
            .arrays["forces"]
    
    return: ase.Atoms
        with attached singlepoint calculator of mean forces and energies.
        additionally fields populated:
            .info["energy"]
            .info["std_energy"]
            .arrays["forces"]
            .arrays["std_forces"]
    """
    
    atoms = []
    n = len(atoms_list)
    for i in range(len(atoms_list[0])):
        atoms.append(atoms_list[0][i].copy())
        energies = []
        forces = []
        for j in range(n):
            energies.append(atoms_list[j][i].info["energy"])
            forces.append(atoms_list[j][i].arrays["forces"])
        energy = np.mean(energies)
        force = np.mean(forces, axis=0)
        std_energy = np.std(energies)
        std_force = np.std(forces, axis=0)
        
        sp_calc = sp(atoms=atoms[i], energy=energy, forces=force)
        atoms[i].calc = sp_calc
        atoms[i].info["energy"] = energy
        atoms[i].info["std_energy"] = std_energy
        atoms[i].arrays["forces"] = force
        atoms[i].arrays["std_forces"] = std_force
        if "stress" in atoms[i].info:
            del atoms[i].info["stress"]
        
    return atoms
