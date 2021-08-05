import pandas as pd
import numpy as np
"""
These functions organize ase.Atoms objects into useful dataframes.
Dataframes are particularly useful for visualization and analysis.
"""

def make_energy_df(train_in, train_out, valid_in, valid_out, natoms_cutoff=500):
    """
    make a energy dataframe with columns: "predicted value", "dft value", "structure", "set"
    given the atoms objects.
    
    train_in: ase.Atoms
        training data with true labels
    train_out: ase.Atoms
        training data with predicted labels
    valid_in: ase.Atoms
        validation data with true labels
    valid_out: ase.Atoms
        validation data with predicted labels
    natoms_cutoff: int
        number of atoms cutoff between "large" and "small" structure
    
    returns: pandas.DataFrame
    """
    
    columns = ["predicted value", "dft value", "structure", "set"]
    l = []
    def list_addrow(in_atoms, out_atoms, subset, l):
        for ia, oa in zip(in_atoms, out_atoms):
            symbols = ia.get_chemical_symbols()
            n_atoms = len(symbols)
            structure = "small" if n_atoms < natoms_cutoff else "large"
            pred_energy = ia.info["energy"] / n_atoms
            dft_energy = oa.info["energy"] / n_atoms
            l.append([pred_energy, dft_energy, structure, subset])
    list_addrow(train_in, train_out, "train", l)
    list_addrow(valid_in, valid_out, "valid", l)
    df = pd.DataFrame(l, columns=columns)
    return df
    
    
def make_forces_df(train_in, train_out, valid_in, valid_out, natoms_cutoff=500):
    """
    make a forces dataframe with columns: "predicted value", "dft value", "structure", "set", "species"
    given the atoms objects.
    
    train_in: ase.Atoms
        training data with true labels
    train_out: ase.Atoms
        training data with predicted labels
    valid_in: ase.Atoms
        validation data with true labels
    valid_out: ase.Atoms
        validation data with predicted labels
    natoms_cutoff: int
        number of atoms cutoff between "large" and "small" structure
    
    returns: pandas.DataFrame
    """
    
    columns = ["predicted value", "dft value", "structure", "set", "species"]
    l = []
    def list_addrow(in_atoms, out_atoms, subset, l):
        for ia, oa in zip(in_atoms, out_atoms):
            symbols = ia.get_chemical_symbols()
            forcesi = ia.arrays["forces"]
            forceso = oa.arrays["forces"]
            structure = "small" if len(symbols) < natoms_cutoff else "large"
            for i, spec in enumerate(symbols):
                for j in range(3):
                    fi = forcesi[i][j]
                    fo = forceso[i][j]
                    l.append([fo, fi, structure, subset, spec])
    list_addrow(train_in, train_out, "train", l)
    list_addrow(valid_in, valid_out, "valid", l)
    df = pd.DataFrame(l, columns=columns)
    return df

def make_energy_res_df(train_in, train_out, valid_in, valid_out, natoms_cutoff=500):
    """
    make a energy dataframe with columns: "residual", "uncertainty", "structure", "set"
    given the atoms objects.
    
    train_in: ase.Atoms
        training data with true labels
    train_out: ase.Atoms
        training data with predicted labels
    valid_in: ase.Atoms
        validation data with true labels
    valid_out: ase.Atoms
        validation data with predicted labels
    natoms_cutoff: int
        number of atoms cutoff between "large" and "small" structure
    
    returns: pandas.DataFrame
    """
    
    columns = ["residual", "uncertainty", "structure", "set"]
    l = []
    def list_addrow(in_atoms, out_atoms, subset, l):
        for ia, oa in zip(in_atoms, out_atoms):
            symbols = ia.get_chemical_symbols()
            n_atoms = len(symbols)
            structure = "small" if n_atoms < natoms_cutoff else "large"
            pred_energy = ia.info["energy"] / n_atoms
            dft_energy = oa.info["energy"] / n_atoms
            error = pred_energy - dft_energy
            unc = np.sqrt(oa.info["std_energy"])
            l.append([error, unc, structure, subset])
    list_addrow(train_in, train_out, "train", l)
    list_addrow(valid_in, valid_out, "valid", l)
    df = pd.DataFrame(l, columns=columns)
    return df
    
def make_forces_res_df(train_in, train_out, valid_in, valid_out, natoms_cutoff=500):
    """
    make a forces dataframe with columns: "residual", "uncertainty", "structure", "set", "species"
    given the atoms objects.
    
    train_in: ase.Atoms
        training data with true labels
    train_out: ase.Atoms
        training data with predicted labels
    valid_in: ase.Atoms
        validation data with true labels
    valid_out: ase.Atoms
        validation data with predicted labels
    natoms_cutoff: int
        number of atoms cutoff between "large" and "small" structure
    
    returns: pandas.DataFrame
    """
    
    columns = ["residual", "uncertainty", "structure", "set", "species"]
    l = []
    def list_addrow(in_atoms, out_atoms, subset, l):
        for ia, oa in zip(in_atoms, out_atoms):
            symbols = ia.get_chemical_symbols()
            dft_forces = ia.arrays["forces"]
            pred_forces = oa.arrays["forces"]
            ae = pred_forces - dft_forces
            uncs = np.sqrt(np.abs(oa.arrays["std_forces"]))
            structure = "small" if len(symbols) < natoms_cutoff else "large"
            for i, spec in enumerate(symbols):
                for j in range(3):
                    error = ae[i][j]
                    unc = uncs[i][j]
                    l.append([error, unc, structure, subset, spec])
    list_addrow(train_in, train_out, "train", l)
    list_addrow(valid_in, valid_out, "valid", l)
    df = pd.DataFrame(l, columns=columns)
    return df


def make_forces_res_mean_df(train_in, train_out, valid_in, valid_out, natoms_cutoff=500):
    """
    make a forces dataframe with columns: "mean absolute error", "uncertainty", "structure", "set"
    given the atoms objects.
    
    train_in: ase.Atoms
        training data with true labels
    train_out: ase.Atoms
        training data with predicted labels
    valid_in: ase.Atoms
        validation data with true labels
    valid_out: ase.Atoms
        validation data with predicted labels
    natoms_cutoff: int
        number of atoms cutoff between "large" and "small" structure
    
    returns: pandas.DataFrame
    """
    
    columns = ["mean absolute error", "uncertainty", "structure", "set"]
    l = []
    def list_addrow(in_atoms, out_atoms, subset, l):
        for ia, oa in zip(in_atoms, out_atoms):
            symbols = ia.get_chemical_symbols()
            n_atoms = len(symbols)
            structure = "small" if n_atoms < natoms_cutoff else "large"
            pred_forces = oa.arrays["force"]
            dft_forces = ia.arrays["forces"]
            mae = np.mean(np.abs(pred_forces - dft_forces))
            uncs = np.sqrt(np.abs(oa.arrays["gap_variance_gradient"]))
            unc = np.mean(uncs)
            l.append([mae, unc, structure, subset])
    list_addrow(train_in, train_out, "train", l)
    list_addrow(valid_in, valid_out, "valid", l)
    df = pd.DataFrame(l, columns=columns)
    return df
