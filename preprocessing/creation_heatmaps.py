import argparse
import fnmatch
import numpy as np
from tqdm import tqdm
import multiprocessing
import os
from matplotlib import pyplot as plt
from Bio.PDB.PDBParser import PDBParser

def read_structure(path):
    """This function read the pdb file and extract the Bio.PDB structure

    Args:
        path (str): The path to the pdb file

    Returns:
        Bio.PDB.structure: Structure file 
    """
    parser = PDBParser(PERMISSIVE=1)
    base = os.path.basename(path)
    name_f = os.path.splitext(base)[0]
    name_f = "_".join(name_f.split("_")[:-5])
    structure = parser.get_structure(name_f, path)
    return name_f, structure


def filter_residues(structure):
    """Filter the amino acids that are not valid

    Args:
        structure (Bio.PDB.structure): stucture

    Returns:
        Bio.PDB.chain: returns the chain of the protein in terms of residues
    """
    residues = tuple(structure.get_residues())
    residues = tuple(filter(lambda x: x.id[0] == " ", residues))
    return residues


def compute_distance_minimum(residue1, residue2):
    """This function computes the minimum distance between two residues. 
    This distance is equal to the minimum of all the distances between the 
    atoms from residue1 and the atoms from residue2

    Args:
        residue1 (Bio.PDB.Residue): Residue n°1
        residue2 (Bio.PDB.Residue): Residue n°2

    Returns:
        int: minimum distance between two residues
    """
    distances = []
    for atom1 in residue1:
        for atom2 in residue2:
            distances.append(atom1-atom2)
    return np.min(distances)


def compute_distance_residue(residue1, residue2):
    """This function computes the distance between the C-alpha atoms of the residuesz

    Args:
        residue1 (Bio.PDB.Residue): Residue n°1
        residue2 (Bio.PDB.Residue): Residue n°2

    Returns:
        int: distance
    """
    return residue1["CA"] - residue2["CA"]

def compute_distance(chain, method=None):
    """This function computes the distances between all residues of the same sequence and returns a distance matrix

    Args:
        chain (Bio.PDB.chain): chain of the protein
        method (str, optional): the method to use to compûte the distance between each two residues. Can be minimum or distance between C-alpha atoms. Defaults to None.

    Returns:
        numpy.array: a numpy array representing the distances between the residues of the chain.
    """
    size = len(chain)
    distance_matrix = np.zeros((size, size), np.float)
    for row, residue1 in enumerate(chain):
        for col, residue2 in enumerate(chain):
            if method == 'min':
                distance_matrix[row, col] = compute_distance_minimum(residue1, residue2)
            else:
                distance_matrix[row, col] = compute_distance_residue(residue1, residue2)
    return distance_matrix

def plot_heatmap(matrix, full_path, name_file, title, option):
    """This function serves to plot the heatmap of the distanc ematrix

    Args:
        matrix (numpy.array): the distance matrix
        name_file (str): name of the protein
        title (str): title of the figure
        option (str): the method of computing the distance

    Returns:
        str: path to the figure
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    fig.suptitle(title)
    heatmap = plt.pcolormesh(matrix)
    legend = plt.colorbar(heatmap)
    legend.set_label("distance")
    ax.imshow(matrix, interpolation='none')
    name_out = os.path.join(full_path, 'figures/heatmap_{}_{}.png'.format(name_file, option))
    plt.savefig(name_out, format="png")
    plt.close()
    return name_out

def generate_save_matrice(path):
    """Generate the distance matrices from the pdb files and save them along with the heatmap

    Args:
        path (str): path to the pdbfile
        method (str): Method to compute the distance between each two residues
    """
    name_f, structure = read_structure(path)
    residues = filter_residues(structure)
    if os.path.exists(os.path.join(PATH_PDB, f'matrix_data/{name_f}.npy')):
        return None
    distance_matrix = compute_distance(residues, method = METHOD)

    np.save(os.path.join(PATH_SAVE, f'matrix_data/{name_f}.npy'),  distance_matrix)
    plot_heatmap(distance_matrix,PATH_SAVE, name_f, 'Distance Matrix ', METHOD)

def main(args):
    pool = multiprocessing.Pool(16)
    global METHOD
    global PATH_PDB
    global PATH_SAVE
    METHOD = args.method
    PATH_PDB = args.path_pdb
    PATH_SAVE = args.path_save

    full_path = os.path.join(os.path.join(args.path, 'PDBFiles'), args.arg)
    files = fnmatch.filter(os.listdir(full_path), "*.pdb")
    for result in tqdm(pool.imap_unordered(generate_save_matrice, [os.path.join(full_path, file) for file in files]), total=len(files)):
        continue
    # for file in tqdm(files):
    #     generate_save_matrice(args.path, os.path.join(full_path, file), args.method, args.arg)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_pdb", type=str, default="../../", 
                        help="Folder containing the PDB Files")
    parser.add_argument("--path_save", type=str, default="../../", 
                        help="Folder to save the reuslting matrices")
    parser.add_argument("-m", "--method", type=str, default = 'residue',
                        help = "Method to compute the distance between each two residues")
    
    main(parser.parse_args())