import numpy as np
import ase
import ase.io
import re
from ase.atoms import Atoms
from ase.calculators.calculator import FileIOCalculator, all_changes
from ase.units import Rydberg, Bohr, GPa

# cfg functions from Hao

def dump_cfg(frames, filename, symbol_to_type, mode='w'):
    """
    writes .cfg file given ase.Atoms object
    
    frames: ase.Atoms
        atoms to be written
    filename: string
        filename ending with .cfg
    symbol_to_type: dict
        mapping of element to index. e.g. {"Cu" : 0, "C" : 1}
    mode: string
        write or append
    
    return: None
    
    """
    with open(filename, mode) as f:
        for atoms in frames:
            ret = ''
            ret += 'BEGIN_CFG\n'
            ret += 'Size\n{}\n'.format(len(atoms))
            pbc = atoms.get_pbc()
            try:
                cell = atoms.get_cell()[:]
                ret += 'Supercell\n'
                for axis in range(3):
                    # Consider PBC. Assume pbc is [1,0,0] [1,1,0] or [1,1,1]
                    if pbc[axis]:
                        ret += '{} {} {}\n'.format(*cell[axis])
                #ret += 'Supercell\n{} {} {}\n{} {} {}\n{} {} {}\n'.format(*cell[0], *cell[1], *cell[2])
            except:
                pass
            cartes = atoms.positions
            atoms.info = dict()
            try:
                atoms.info['forces'] = atoms.get_forces()
            except:
                pass
            if 'forces' in atoms.info:
                forces = atoms.info['forces']
                ret += 'AtomData: id type cartes_x cartes_y cartes_z fx fy fz\n'
                for i, atom in enumerate(atoms):
                    ret += '{} {} {} {} {} {} {} {}\n'.format(i + 1, symbol_to_type[atom.symbol], *cartes[i], *forces[i])
            else:
                ret += 'AtomData: id type cartes_x cartes_y cartes_z\n'
                for i, atom in enumerate(atoms):
                    ret += '{} {} {} {} {}\n'.format(i + 1, symbol_to_type[atom.symbol], *cartes[i])
            try:
                atoms.info['energy'] = atoms.get_potential_energy()
            except:
                pass
            if 'energy' in atoms.info:
                ret += 'Energy\n{}\n'.format(atoms.info['energy'])
            try:
                atoms.info['stress'] = atoms.get_stress()
            except:
                pass
            if 'stress' in atoms.info:
                stress = atoms.info['stress'] * atoms.get_volume() * -1.
                ret += 'PlusStress: xx yy zz yz xz xy\n{} {} {} {} {} {}\n'.format(*stress)
            if 'identification' in atoms.info:
                ret += 'Feature identification {}\n'.format(atoms.info['identification'])
            ret += 'END_CFG\n'
            f.write(ret)
            
            
def replace_virial(m):
    whole = m.group(0)
    lattice = m.group(1)
    stress = m.group(3)
    lattice_matrix = np.array(list(map(float,lattice.split(" ")))).reshape((3,3))
    volume = np.linalg.det(lattice_matrix)
    virial = np.array(list(map(float,stress.split(" ")))) * volume
    virial_str = " ".join(map(str, virial))
    new_str = re.sub('(?<=stress=")(-*\d+?\.\d+ ){8}-*\d+?\.\d+', virial_str, whole)
    new_str = re.sub('stress', "virial", new_str)
    return new_str

def clear_data(atoms, momenta=True, stress=True):
    for atoms in atoms:
        if momenta and "momenta" in atoms.arrays:
            del atoms.arrays["momenta"]
        if stress and "stress" in atoms.info:
            del atoms.info["stress"]
        if stress and "stress" in atoms.calc.results:
            del atoms.calc.results["stress"]

def dump_xyz(atoms, fname, virial=True):
    """
    writes .xyz file given ase.Atoms object.
    
    atoms: ase.Atoms
        atoms to be written
    filename: string
        filename ending without filename extension
    virial: bool
        True will write stress, False will not
    
    return: None
    
    """
    
    filename = '%s.xyz'%fname
    clear_data(atoms, stress=not virial)
    ase.io.write(filename, atoms)
    if virial:
        file = open(filename, "rt")
        text = file.read()
        file.close()
        new_text = re.sub('Lattice\="((\d+?\.\d+ ){8}\d+?\.\d+)" .* stress="((-*\d+?\.\d+ ){8}-*\d+?\.\d+)"',
               replace_virial, text)
        file = open(filename, "wt")
        file.write(new_text)
        file.close()
        
def reading_state(line, state):
    # change the reading state based on current state and line
    if 'BEGIN_CFG' in line:
        state = 'begin'
    elif 'Size' in line:
        state = 'size'
    elif 'Supercell' in line: 
        state = 'cell'
    elif 'AtomData' in line: 
        state = 'atom'
    elif 'Energy' in line: 
        state = 'en'
    elif 'PlusStress' in line: 
        state = 'stress'
    elif "Feature" in line:
        state = "grade"
    elif 'END_CFG' in line: 
        state = 'end'

    return state    

def load_cfg(filename, type_to_symbol):
    """
    reads .cfg file to ase.Atoms object
    
    filename: string
        filename ending with .cfg
    type_to_symbol: dict
        mapping of index to element. e.g. {0 : "Cu", 1 : "C"}
    
    return: ase.Atoms
    
    """
    frames = []
    state = 'no'
    with open(filename) as f:
        line = 'chongchongchong!'
        while line:
            state = reading_state(line, state)
            #print(line)
            #print(state)
            #print(' ')

            if state == 'no':
                pass

            if state == 'begin':
                cell = np.zeros((3, 3))
                positions = []
                symbols = []
                celldim = 0

            if state == 'size':
                line = f.readline()
                natoms = int(line.split()[0])
            
            if state == 'cell':
                #for i in range(3):
                #    line = f.readline()
                # 0D systems have no Supercell field
                if 'Supercell' not in line:
                    for j in range(3):
                        cell[celldim, j] = float(line.split()[j])
                    celldim += 1
            
            if state == 'atom':
                has_force = False
                if 'fx' in line:
                    has_force = True
                    forces = []
                
                for _ in range(natoms):
                    line = f.readline()
                    fields = line.split()
                    symbols.append(type_to_symbol[int(fields[1])])
                    positions.append(list(map(float, fields[2: 5])))
                    if has_force:
                        forces.append(list(map(float, fields[5: 8])))
                if celldim == 1:
                    pbc = [1,0,0]
                    #cell[1,1] = 30
                    #cell[2,2] = 30
                elif celldim == 2:
                    pbc = [1,1,0]
                    #cell[2,2] = 30
                elif celldim == 3:
                    pbc = [1,1,1]
                else:
                    pbc = False
                atoms = Atoms(symbols=symbols, cell=cell, positions=positions, pbc=pbc)
                if has_force:
                    atoms.arrays['forces'] = np.array(forces)

            if state == 'en':
                line = f.readline()
                atoms.info['energy'] = float(line.split()[0])

            if state == 'stress':
                line = f.readline()
                plus_stress = np.array(list(map(float, line.split())))
                # It is possible the cell is not pbc along 3 directions
                if atoms.get_pbc().all():
                    atoms.info['stress'] = -plus_stress / atoms.get_volume()
                    atoms.info['pstress'] = atoms.info['stress'] / GPa
                    
            if state == "grade":
                grade = line.split()[-1]
                atoms.info["grade"] = float(grade)
                
            if state == 'end':
                frames.append(atoms)
                state = 'no'

            line = f.readline()
            #if 'identification' in line:
            #    atoms.info['identification'] = int(line.split()[2])
    return frames
