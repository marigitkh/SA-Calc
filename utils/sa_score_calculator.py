import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem


def calculate_contribution_scores(fragment_counts):
    """
    Calculate contribution scores for fragments.

    Parameters:
        fragment_counts (Counter): Fragment counts from the dataset.

    Returns:
        dict: Fragment contribution scores {fragment: score}.
    """
    total_fragments = sum(fragment_counts.values())
    threshold_count = total_fragments * 0.8

    # Sort fragments by count in descending order
    sorted_fragments = sorted(fragment_counts.items(), key=lambda x: -x[1])
    
    # Identify fragment types contributing to 80% of total occurrences
    cumulative_count = 0
    frequent_fragment_types = set()
    for fragment, count in sorted_fragments:
        cumulative_count += count
        frequent_fragment_types.add(fragment)
        if cumulative_count >= threshold_count:
            break

    # Number of fragment types forming 80% of the total database
    num_frequent_fragment_types = len(frequent_fragment_types)
    print(num_frequent_fragment_types)
    # Calculate scores
    contribution_scores = {}
    for fragment, count in fragment_counts.items():
        if num_frequent_fragment_types > 0:
            contribution_scores[fragment] = np.log(count / num_frequent_fragment_types)
        else:
            contribution_scores[fragment] = float('-inf')  # Handle edge case where no fragment types are identified
    
    return contribution_scores


def score_molecules(smiles_list, contribution_scores, radius=2):
    """
    Calculate the sum of fragment contribution scores for each molecule.

    Parameters:
        smiles_list (list of str): List of SMILES strings for molecules.
        contribution_scores (dict): Dictionary of fragment contribution scores.
        radius (int): Radius for Morgan fingerprint.

    Returns:
        dict: A dictionary with SMILES as keys and molecule scores as values.
    """
    molecule_scores = {}

    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"Invalid SMILES: {smiles}")
            continue
        
        # Generate ECFP and calculate score
        fp = AllChem.GetMorganFingerprint(mol, radius)
        score = 0
        for fragment, count in fp.GetNonzeroElements().items():
            score += contribution_scores.get(fragment, 0) * count
        molecule_scores[smiles] = score

    return molecule_scores
