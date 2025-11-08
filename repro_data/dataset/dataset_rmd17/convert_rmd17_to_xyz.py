#!/usr/bin/env python3

import numpy as np
import argparse
from pathlib import Path
from ase import Atoms
from ase.io import write
import pandas as pd

def convert_rmd17_to_xyz(args):
    rmd17_data_dir = Path(args.rmd17_dir)
    npz_data_dir = rmd17_data_dir / 'npz_data'
    splits_dir = rmd17_data_dir / 'splits'
    output_dir = Path(args.output_dir) / args.molecule
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.molecule.startswith('revised_'):
        molecule_name = args.molecule.replace('revised_', '')
        npz_file = npz_data_dir / f'rmd17_{molecule_name}.npz'
    else:
        npz_file = npz_data_dir / f'rmd17_{args.molecule}.npz'
    
    print(f"Loading data file: {npz_file}")
    raw_data = np.load(npz_file)
    
    z = raw_data["nuclear_charges"]
    pos = raw_data["coords"]
    energy = raw_data["energies"]
    force = raw_data["forces"]
    
    print(f"Dataset size: {len(energy)} configurations")
    print(f"Number of atoms: {len(z)}")
    
    KCAL_TO_EV = 0.0433641
    energy_ev = energy * KCAL_TO_EV
    force_ev = force * KCAL_TO_EV
    
    print(f"Energy range: {energy_ev.min():.6f} ~ {energy_ev.max():.6f} eV")
    print(f"Force range: {force_ev.min():.6f} ~ {force_ev.max():.6f} eV/Å")
    
    print(f"Custom split: train {args.n_train}, valid {args.n_valid}, test {args.n_test}")
    
    np.random.seed(123)
    
    total_needed = args.n_train + args.n_valid + args.n_test
    if len(energy) < total_needed:
        print(f"Warning: dataset size ({len(energy)}) is less than required samples ({total_needed})")
        print(f"All available data will be used for splitting")
    
    all_indices = np.random.permutation(len(energy))
    
    train_end = min(args.n_train, len(energy))
    valid_end = min(train_end + args.n_valid, len(energy))
    test_end = min(valid_end + args.n_test, len(energy))
    
    train_indices = all_indices[:train_end]
    valid_indices = all_indices[train_end:valid_end]
    test_indices = all_indices[valid_end:test_end]
    
    print(f"Actual split: train {len(train_indices)}, valid {len(valid_indices)}, test {len(test_indices)}")
    
    datasets = {
        'train': train_indices,
        'valid': valid_indices, 
        'test': test_indices
    }
    
    for split_name, indices in datasets.items():
        print(f"Converting {split_name} dataset...")
        
        atoms_list = []
        for idx in indices:
            atoms = Atoms(
                numbers=z,
                positions=pos[idx],
                cell=[20, 20, 20],
                pbc=True
            )
            
            atoms.info['REF_energy'] = float(energy_ev[idx])
            atoms.arrays['REF_forces'] = force_ev[idx]
            atoms.info['config_type'] = 'Default'
            
            atoms_list.append(atoms)
        
        output_file = output_dir / f'{split_name}.xyz'
        write(output_file, atoms_list)
        print(f"Saved {split_name} dataset to: {output_file}")
    
    print("Data conversion finished!")

def main():
    parser = argparse.ArgumentParser(description='Convert rMD17 dataset to official MACE-supported XYZ format')
    
    parser.add_argument('--molecule', type=str, default='all',
                      choices=['revised_benzene', 'revised_uracil', 'revised_aspirin', 
                              'revised_ethanol', 'revised_malonaldehyde', 'revised_naphthalene',
                              'revised_paracetamol', 'revised_salicylic', 'revised_toluene',
                              'revised_azobenzene', 'all'],
                      help='Molecule to convert')
    
    parser.add_argument('--split-id', type=str, default='01',
                      choices=['01', '02', '03', '04', '05'],
                      help='Split ID')
    
    parser.add_argument('--rmd17-dir', type=str, 
                      help='rMD17 dataset directory')
    
    parser.add_argument('--output-dir', type=str, 
                      help='Output directory')
    
    parser.add_argument('--n-train', type=int, default=100,
                      help='Number of training samples')
    
    parser.add_argument('--n-valid', type=int, default=900,
                      help='Number of validation samples')
    
    parser.add_argument('--n-test', type=int, default=1000,
                      help='Number of test samples')
    
    args = parser.parse_args()
    if args.molecule == 'all':
        for molecule in ['benzene', 'uracil', 'aspirin', 'ethanol', 'malonaldehyde', 'naphthalene', 'paracetamol', 'salicylic', 'toluene', 'azobenzene']:
            args.molecule = f'revised_{molecule}'
            convert_rmd17_to_xyz(args)
    else:
        convert_rmd17_to_xyz(args)

if __name__ == "__main__":
    main() 