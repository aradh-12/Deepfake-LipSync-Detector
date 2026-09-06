import csv

import numpy as np

from feature_fusion.utils import (
    load_lip_coordinates,
    load_mfcc,
    save_feature_vector,
)


def test_load_lip_coordinates(tmp_path):

    csv_file = tmp_path / "lip_coordinates.csv"

    with open(
        csv_file,
        "w",
        newline=""
    ) as file:

        writer = csv.writer(file)

        writer.writerow(
            ["landmark", "x", "y"]
        )

        writer.writerow([0, 10, 20])
        writer.writerow([1, 30, 40])
        writer.writerow([2, 50, 60])


    coordinates = load_lip_coordinates(
        csv_file
    )


    assert isinstance(
        coordinates,
        np.ndarray
    )

    assert coordinates.shape == (3, 2)

    np.testing.assert_array_equal(

        coordinates,

        np.array(

            [
                [10, 20],
                [30, 40],
                [50, 60],
            ]

        )

    )


def test_load_mfcc(tmp_path):

    mfcc_file = tmp_path / "mfcc.npy"

    expected = np.array(

        [
            [1.0, 2.0],
            [3.0, 4.0],
        ]

    )


    np.save(
        mfcc_file,
        expected
    )


    result = load_mfcc(
        mfcc_file
    )


    np.testing.assert_array_equal(

        result,

        expected

    )


def test_save_feature_vector(tmp_path):

    output_file = (

        tmp_path
        / "nested"
        / "features.npy"

    )


    features = np.array(

        [
            1.0,
            2.0,
            3.0,
        ]

    )


    save_feature_vector(

        features,

        output_file

    )


    assert output_file.exists()


    loaded = np.load(
        output_file
    )


    np.testing.assert_array_equal(

        loaded,

        features

    )
