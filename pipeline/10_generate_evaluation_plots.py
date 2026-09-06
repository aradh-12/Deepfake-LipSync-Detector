from pathlib import Path
import csv

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parent.parent


FINAL_PIPELINE_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "final_pipeline"
)


EVALUATION_ROOT = (
    FINAL_PIPELINE_ROOT
    / "evaluation"
)


PREDICTIONS_PATH = (
    EVALUATION_ROOT
    / "test_predictions.csv"
)


PLOTS_ROOT = (
    EVALUATION_ROOT
    / "plots"
)


# ============================================================
# CONFIGURATION
# ============================================================

CLASS_NAMES = [
    "REAL",
    "FAKE"
]


# ============================================================
# CHECK REQUIRED FILES
# ============================================================

def check_required_files():

    print()

    print("=" * 70)

    print(
        "GENERATING EVALUATION PLOTS"
    )

    print("=" * 70)

    print()

    print(
        "Predictions file:"
    )

    print(
        PREDICTIONS_PATH
    )

    print()

    if not PREDICTIONS_PATH.exists():

        raise FileNotFoundError(

            "Prediction file not found:\n"
            f"{PREDICTIONS_PATH}"
        )

    print(
        "All required files found."
    )

    print()


# ============================================================
# LOAD PREDICTIONS
# ============================================================

def load_predictions():

    print("=" * 70)

    print(
        "LOADING TEST PREDICTIONS"
    )

    print("=" * 70)

    print()

    actual_labels = []

    predicted_labels = []

    fake_probabilities = []

    with open(

        PREDICTIONS_PATH,

        "r",

        encoding="utf-8"

    ) as file:

        reader = csv.DictReader(

            file
        )

        for row in reader:

            actual_labels.append(

                int(
                    row["actual_label"]
                )
            )

            predicted_labels.append(

                int(
                    row["predicted_label"]
                )
            )

            fake_probabilities.append(

                float(
                    row["fake_probability"]
                )
            )

    y_true = np.array(

        actual_labels,

        dtype=np.int32
    )

    y_pred = np.array(

        predicted_labels,

        dtype=np.int32
    )

    probabilities = np.array(

        fake_probabilities,

        dtype=np.float32
    )

    print(
        "Samples loaded:",
        len(y_true)
    )

    print(
        "Real samples:",
        int(
            np.sum(y_true == 0)
        )
    )

    print(
        "Fake samples:",
        int(
            np.sum(y_true == 1)
        )
    )

    print()

    return (

        y_true,

        y_pred,

        probabilities
    )


# ============================================================
# CONFUSION MATRIX PLOT
# ============================================================

def generate_confusion_matrix_plot(

    y_true,

    y_pred

):

    print("=" * 70)

    print(
        "GENERATING CONFUSION MATRIX"
    )

    print("=" * 70)

    print()

    matrix = confusion_matrix(

        y_true,

        y_pred,

        labels=[0, 1]
    )

    figure, axis = plt.subplots(

        figsize=(7, 6)
    )

    image = axis.imshow(

        matrix
    )

    axis.set_xticks(

        [0, 1]
    )

    axis.set_yticks(

        [0, 1]
    )

    axis.set_xticklabels(

        CLASS_NAMES
    )

    axis.set_yticklabels(

        CLASS_NAMES
    )

    axis.set_xlabel(

        "Predicted Class"
    )

    axis.set_ylabel(

        "Actual Class"
    )

    axis.set_title(

        "Confusion Matrix"
    )

    for row in range(

        matrix.shape[0]
    ):

        for column in range(

            matrix.shape[1]
        ):

            axis.text(

                column,

                row,

                str(
                    matrix[row, column]
                ),

                ha="center",

                va="center",

                fontsize=16
            )

    figure.colorbar(

        image,

        ax=axis
    )

    figure.tight_layout()

    output_path = (

        PLOTS_ROOT
        / "confusion_matrix.png"
    )

    figure.savefig(

        output_path,

        dpi=300,

        bbox_inches="tight"
    )

    plt.close(

        figure
    )

    print(
        "Confusion Matrix saved:"
    )

    print(
        output_path
    )

    print()


# ============================================================
# ROC CURVE
# ============================================================

def generate_roc_curve(

    y_true,

    probabilities

):

    print("=" * 70)

    print(
        "GENERATING ROC CURVE"
    )

    print("=" * 70)

    print()

    false_positive_rate, true_positive_rate, _ = (

        roc_curve(

            y_true,

            probabilities
        )
    )

    roc_auc = auc(

        false_positive_rate,

        true_positive_rate
    )

    figure, axis = plt.subplots(

        figsize=(7, 6)
    )

    axis.plot(

        false_positive_rate,

        true_positive_rate,

        linewidth=2,

        label=(
            f"ROC Curve "
            f"(AUC = {roc_auc:.4f})"
        )
    )

    axis.plot(

        [0, 1],

        [0, 1],

        linestyle="--"
    )

    axis.set_xlabel(

        "False Positive Rate"
    )

    axis.set_ylabel(

        "True Positive Rate"
    )

    axis.set_title(

        "ROC Curve"
    )

    axis.legend(

        loc="lower right"
    )

    axis.grid(

        True
    )

    figure.tight_layout()

    output_path = (

        PLOTS_ROOT
        / "roc_curve.png"
    )

    figure.savefig(

        output_path,

        dpi=300,

        bbox_inches="tight"
    )

    plt.close(

        figure
    )

    print(
        "ROC Curve saved:"
    )

    print(
        output_path
    )

    print()

    return roc_auc


# ============================================================
# PROBABILITY DISTRIBUTION
# ============================================================

def generate_probability_distribution(

    y_true,

    probabilities

):

    print("=" * 70)

    print(
        "GENERATING PROBABILITY DISTRIBUTION"
    )

    print("=" * 70)

    print()

    real_probabilities = probabilities[
        y_true == 0
    ]

    fake_probabilities = probabilities[
        y_true == 1
    ]

    figure, axis = plt.subplots(

        figsize=(9, 6)
    )

    axis.hist(

        real_probabilities,

        bins=20,

        alpha=0.7,

        label="Actual REAL"
    )

    axis.hist(

        fake_probabilities,

        bins=20,

        alpha=0.7,

        label="Actual FAKE"
    )

    axis.axvline(

        0.5,

        linestyle="--",

        linewidth=2,

        label="Decision Threshold"
    )

    axis.set_xlabel(

        "Predicted Fake Probability"
    )

    axis.set_ylabel(

        "Number of Videos"
    )

    axis.set_title(

        "Prediction Probability Distribution"
    )

    axis.legend()

    axis.grid(

        True
    )

    figure.tight_layout()

    output_path = (

        PLOTS_ROOT
        / "probability_distribution.png"
    )

    figure.savefig(

        output_path,

        dpi=300,

        bbox_inches="tight"
    )

    plt.close(

        figure
    )

    print(
        "Probability Distribution saved:"
    )

    print(
        output_path
    )

    print()


# ============================================================
# MAIN
# ============================================================

def main():

    check_required_files()

    PLOTS_ROOT.mkdir(

        parents=True,

        exist_ok=True
    )

    y_true, y_pred, probabilities = (

        load_predictions()
    )

    generate_confusion_matrix_plot(

        y_true,

        y_pred
    )

    roc_auc = generate_roc_curve(

        y_true,

        probabilities
    )

    generate_probability_distribution(

        y_true,

        probabilities
    )

    print("=" * 70)

    print(
        "EVALUATION PLOTS COMPLETE"
    )

    print("=" * 70)

    print()

    print(
        f"ROC-AUC: {roc_auc:.4f}"
    )

    print()

    print(
        "Plots directory:"
    )

    print(
        PLOTS_ROOT
    )

    print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()