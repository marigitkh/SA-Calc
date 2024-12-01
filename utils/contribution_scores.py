import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from collections import Counter

def preprocess_smiles(df: pd.DataFrame, column_name: str = 'Smiles'):
    """
    Preprocess a DataFrame to remove rows with null or invalid SMILES strings
    and convert valid SMILES strings to molecular objects.

    Parameters:
        df (pd.DataFrame): Input DataFrame containing a column with SMILES strings.
        column_name (str): Name of the column containing SMILES strings.

    Returns:
        list: A list of RDKit molecule objects.
    """
    # Filter out invalid SMILES and convert to molecular objects
    molecules = [mol for smiles in df[column_name].dropna() if (mol := Chem.MolFromSmiles(smiles))]
    
    return molecules


def calculate_fragment_counts(molecules_list: list):
    """
    Calculate fragment counts across a dataset using ECFC_4#.

    Parameters:
        molecules_list (list of RDKit Mol objects): List of molecule objects.
        radius (int): Radius for Morgan fingerprint.

    Returns:
        Counter: A Counter object with fragment (hashed) as keys and counts as values.
    """
    def get_fragment_counts(mol):
        # Generate fragment counts for a single molecule
        return Counter(AllChem.GetMorganFingerprint(mol, 2).GetNonzeroElements())
    
    # Compute fragment counts sequentially for each molecule
    counters = [get_fragment_counts(mol) for mol in molecules_list]
    
    # Aggregate counts from all molecules
    return sum(counters, Counter())


def calculate_contribution_scores(fragment_counts: Counter):
    """
    Calculate contribution scores for fragments.

    Parameters:
        fragment_counts (Counter): Fragment counts from the dataset.

    Returns:
        dict: Fragment contribution scores {fragment: score}.
    """
    total_fragments = sum(fragment_counts.values())

    # Sort fragments by count in descending order
    sorted_fragments = sorted(fragment_counts.items(), key=lambda x: -x[1])
    
    # Identify fragment types contributing to 80% of total occurrences
    cumulative_count = 0
    frequent_fragment_types = set()
    for fragment, count in sorted_fragments:
        cumulative_count += count
        frequent_fragment_types.add(fragment)
        if cumulative_count >= total_fragments * 0.8:
            break

    # Number of fragment types forming 80% of the total database
    num_frequent_fragment_types = len(frequent_fragment_types)
    
    # Calculate scores
    contribution_scores = {}
    for fragment, count in fragment_counts.items():
        if num_frequent_fragment_types > 0:
            contribution_scores[fragment] = np.log(count / num_frequent_fragment_types)
        else:
            contribution_scores[fragment] = float('-inf')  # Handle case when no fragment types are identified
    
    return contribution_scores


