import csv
import numpy as np
from pathlib import Path


def load_lip_coordinates(csv_file: Path):
    """
    Load lip coordinate CSV.
    Returns a NumPy array of shape (40, 2).
    """

    coordinates = []

    with open(csv_file, "r") as file:

        reader = csv.reader(file)

        next(reader)  # Skip header

        for row in reader:

            x = int(row[1])
            y = int(row[2])

            coordinates.append([x, y])

    return np.array(coordinates)


def load_mfcc(mfcc_file: Path):
    """
    Load MFCC NumPy file.
    """

    return np.load(mfcc_file)


def save_feature_vector(feature_vector, output_file: Path):
    """
    Save synchronized feature vector.
    """

    output_file.parent.mkdir(parents=True, exist_ok=True)

    np.save(output_file, feature_vector)