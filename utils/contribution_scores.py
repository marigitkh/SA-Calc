import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from collections import Counter

def compute_contribution_scores(df: pd.DataFrame, smiles_column: str = 'Smiles'):
    """
    Computes contribution scores for molecular fragments from a DataFrame of SMILES strings.
    
    Parameters:
        df (pd.DataFrame): DataFrame containing SMILES strings.
        smiles_column (str): Name of the column with SMILES strings.

    Returns:
        dict: Contribution scores as {fragment_hash: score}.
    """
    # Clean SMILES column to ensure valid inputs
    df[smiles_column] = df[smiles_column].fillna('').astype(str)
    
    # Add a column for RDKit molecule objects
    PandasTools.AddMoleculeColumnToFrame(df, smilesCol='Mol', molCol=mol_column)
    df = df[df[mol_column].notnull()].reset_index(drop=True)
    
    # Calculate fragment counts
    def get_fragment_counts(mol):
        return Counter(AllChem.GetMorganFingerprint(mol, 2).GetNonzeroElements())
    
    total_fragment_counts = sum(
        (get_fragment_counts(mol) for mol in df[mol_column]), Counter()
    )
    
    # Calculate total number of fragments
    total_fragments = sum(total_fragment_counts.values())
    if total_fragments == 0:
        return {}
    
    # Sort fragments by descending count
    sorted_fragments = sorted(total_fragment_counts.items(), key=lambda x: -x[1])

    # Identify fragments contributing to 80% of occurrences
    cumulative_count = 0
    frequent_fragment_types = set()
    for fragment, count in sorted_fragments:
        cumulative_count += count
        frequent_fragment_types.add(fragment)
        if cumulative_count >= total_fragments * 0.8:
            break
            
    # Compute contribution scores
    num_frequent_types = len(frequent_fragment_types)
    contribution_scores = {
        fragment: np.log(count / num_frequent_types) if num_frequent_types > 0 else float('-inf')
        for fragment, count in total_fragment_counts.items()
    }
    
    return contribution_scores
