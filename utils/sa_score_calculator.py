import math
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from collections import Counter


def calculate_fragment_score(molecule, contribution_scores):
    """
    Compute the fragment-based contribution score for a molecule.

    Parameters:
        molecule (Chem.Mol): RDKit molecule object.
        contribution_scores (dict): Dictionary of fragment contribution scores.

    Returns:
        float: The total score derived from the contribution of molecular fragments.
    """
    # Extract ECFC_4# fragment counts from the molecule
    fragment_counts = Counter(
        AllChem.GetMorganFingerprint(molecule, 2).GetNonzeroElements()
    )

    # Calculate the fragment contribution score
    fragment_score = sum(
        contribution_scores.get(fragment, 0) * count
        for fragment, count in fragment_counts.items()
    )

    return fragment_score


def calculate_complexity_score(molecule):
    """
    Compute the complexity score for a molecule based on structural features.

    Parameters:
        molecule (Chem.Mol): RDKit molecule object.

    Returns:
        float: A score quantifying the structural complexity of the molecule.
    """
    # Ring complexity: accounts for bridgehead and spiro atoms
    ring_info = molecule.GetRingInfo()
    n_ring_bridge_atoms = AllChem.CalcNumBridgeheadAtoms(molecule)
    n_spiro_atoms = AllChem.CalcNumSpiroAtoms(molecule)
    ring_complexity_score = math.log(n_ring_bridge_atoms + 1) + math.log(n_spiro_atoms + 1)

    # Stereo complexity: based on chiral centers
    n_stereo_centers = len(
        Chem.FindMolChiralCenters(molecule, includeUnassigned=True)
    )
    stereo_complexity_score = math.log(n_stereo_centers + 1)

    # Macrocycle penalty: penalizes large rings (size > 8)
    n_macrocycles = sum(1 for ring in ring_info.AtomRings() if len(ring) > 8)
    macrocycle_penalty = math.log(n_macrocycles + 1)

    # Size penalty: based on the number of heavy atoms
    n_atoms = molecule.GetNumHeavyAtoms()
    size_penalty = n_atoms**1.005 - n_atoms

    # Total complexity score
    complexity_score = (
        ring_complexity_score
        + stereo_complexity_score
        + macrocycle_penalty
        + size_penalty
    )

    return complexity_score


def calculate_sa(molecule, contribution_scores):
    """
    Calculate the Synthetic Accessibility (SA) score for a molecule.

    Parameters:
        molecule (Chem.Mol): RDKit molecule object.
        contribution_scores (dict): Dictionary of fragment contribution scores.

    Returns:
        float: SA score scaled between 1 and 10.
    """
    # Compute the fragment and complexity scores
    fragment_score = calculate_fragment_score(molecule, contribution_scores)
    complexity_score = calculate_complexity_score(molecule)

    # Compute the raw SA score
    raw_score = fragment_score - complexity_score

    # Scale the raw SA score to the range [1, 10]
    scaled_score = -1 * raw_score
    sa_score = 1 + (9 / (1 + np.exp(-scaled_score)))

    return sa_score
