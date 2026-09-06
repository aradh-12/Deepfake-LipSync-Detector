from pathlib import Path
import csv
import json

import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score
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


MODEL_ROOT = (
    FINAL_PIPELINE_ROOT
    / "models"
)


SUMMARY_PATH = (
    EVALUATION_ROOT
    / "final_results_summary.txt"
)


JSON_SUMMARY_PATH = (
    EVALUATION_ROOT
    / "final_results_summary.json"
)


# ============================================================
# PROJECT INFORMATION
# ============================================================

PROJECT_TITLE = (
    "Multimodal Deepfake Lip-Sync Anomaly Detector"
)


MODEL_NAME = (
    "Multimodal LSTM"
)


FEATURE_INFORMATION = (
    "Lip Features (80) + MFCC Features (39)"
)


TOTAL_FEATURES = 119


MAX_SEQUENCE_LENGTH = 150


CLASS_NAMES = {
    0: "REAL",
    1: "FAKE"
}


# ============================================================
# CHECK REQUIRED FILES
# ============================================================

def check_required_files():

    print()

    print("=" * 70)

    print(
        "GENERATING FINAL RESULTS SUMMARY"
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

            "Predictions file not found:\n"
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

    video_ids = []

    y_true = []

    y_pred = []

    probabilities = []

    with open(

        PREDICTIONS_PATH,

        "r",

        encoding="utf-8"

    ) as file:

        reader = csv.DictReader(
            file
        )

        for row in reader:

            video_ids.append(

                row["video_id"]
            )

            y_true.append(

                int(
                    row["actual_label"]
                )
            )

            y_pred.append(

                int(
                    row["predicted_label"]
                )
            )

            probabilities.append(

                float(
                    row["fake_probability"]
                )
            )

    y_true = np.array(

        y_true,

        dtype=np.int32
    )

    y_pred = np.array(

        y_pred,

        dtype=np.int32
    )

    probabilities = np.array(

        probabilities,

        dtype=np.float32
    )

    print(
        "Total test samples:",
        len(y_true)
    )

    print(
        "REAL samples:",
        int(np.sum(y_true == 0))
    )

    print(
        "FAKE samples:",
        int(np.sum(y_true == 1))
    )

    print()

    return (

        video_ids,

        y_true,

        y_pred,

        probabilities
    )


# ============================================================
# CALCULATE RESULTS
# ============================================================

def calculate_results(

    y_true,

    y_pred,

    probabilities

):

    print("=" * 70)

    print(
        "CALCULATING FINAL RESULTS"
    )

    print("=" * 70)

    print()

    accuracy = accuracy_score(

        y_true,

        y_pred
    )

    precision = precision_score(

        y_true,

        y_pred,

        zero_division=0
    )

    recall = recall_score(

        y_true,

        y_pred,

        zero_division=0
    )

    f1 = f1_score(

        y_true,

        y_pred,

        zero_division=0
    )

    try:

        roc_auc = roc_auc_score(

            y_true,

            probabilities
        )

    except ValueError:

        roc_auc = None


    matrix = confusion_matrix(

        y_true,

        y_pred,

        labels=[0, 1]
    )


    tn = int(matrix[0, 0])

    fp = int(matrix[0, 1])

    fn = int(matrix[1, 0])

    tp = int(matrix[1, 1])


    results = {

        "accuracy": float(accuracy),

        "precision": float(precision),

        "recall": float(recall),

        "f1_score": float(f1),

        "roc_auc":

            float(roc_auc)

            if roc_auc is not None

            else None,


        "total_test_samples":

            int(len(y_true)),


        "real_samples":

            int(np.sum(y_true == 0)),


        "fake_samples":

            int(np.sum(y_true == 1)),


        "true_negatives":

            tn,


        "false_positives":

            fp,


        "false_negatives":

            fn,


        "true_positives":

            tp

    }


    print(

        f"Accuracy : "
        f"{accuracy:.4f}"
    )

    print(

        f"Precision: "
        f"{precision:.4f}"
    )

    print(

        f"Recall   : "
        f"{recall:.4f}"
    )

    print(

        f"F1 Score : "
        f"{f1:.4f}"
    )

    if roc_auc is not None:

        print(

            f"ROC-AUC  : "
            f"{roc_auc:.4f}"
        )

    print()

    return results


# ============================================================
# SAVE TEXT SUMMARY
# ============================================================

def save_text_summary(

    results

):

    EVALUATION_ROOT.mkdir(

        parents=True,

        exist_ok=True
    )


    with open(

        SUMMARY_PATH,

        "w",

        encoding="utf-8"

    ) as file:


        file.write(

            PROJECT_TITLE + "\n"
        )

        file.write(

            "=" * 70 + "\n\n"
        )


        file.write(

            "FINAL PROJECT RESULTS SUMMARY\n\n"
        )


        # ----------------------------------------------------
        # MODEL INFORMATION
        # ----------------------------------------------------

        file.write(

            "MODEL INFORMATION\n"
        )

        file.write(

            "-" * 40 + "\n"
        )


        file.write(

            f"Model: "
            f"{MODEL_NAME}\n"
        )


        file.write(

            f"Features: "
            f"{FEATURE_INFORMATION}\n"
        )


        file.write(

            f"Total Features: "
            f"{TOTAL_FEATURES}\n"
        )


        file.write(

            f"Maximum Sequence Length: "
            f"{MAX_SEQUENCE_LENGTH}\n\n"
        )


        # ----------------------------------------------------
        # DATASET INFORMATION
        # ----------------------------------------------------

        file.write(

            "TEST DATASET INFORMATION\n"
        )

        file.write(

            "-" * 40 + "\n"
        )


        file.write(

            f"Total Test Samples: "
            f"{results['total_test_samples']}\n"
        )


        file.write(

            f"REAL Samples: "
            f"{results['real_samples']}\n"
        )


        file.write(

            f"FAKE Samples: "
            f"{results['fake_samples']}\n\n"
        )


        # ----------------------------------------------------
        # PERFORMANCE
        # ----------------------------------------------------

        file.write(

            "MODEL PERFORMANCE\n"
        )

        file.write(

            "-" * 40 + "\n"
        )


        file.write(

            f"Accuracy: "
            f"{results['accuracy']:.6f}\n"
        )


        file.write(

            f"Precision (FAKE): "
            f"{results['precision']:.6f}\n"
        )


        file.write(

            f"Recall (FAKE): "
            f"{results['recall']:.6f}\n"
        )


        file.write(

            f"F1 Score (FAKE): "
            f"{results['f1_score']:.6f}\n"
        )


        if results["roc_auc"] is not None:

            file.write(

                f"ROC-AUC: "
                f"{results['roc_auc']:.6f}\n"
            )


        file.write(

            "\n"
        )


        # ----------------------------------------------------
        # CONFUSION MATRIX
        # ----------------------------------------------------

        file.write(

            "CONFUSION MATRIX\n"
        )

        file.write(

            "-" * 40 + "\n"
        )


        file.write(

            "                 PREDICTED\n"
        )


        file.write(

            "              REAL     FAKE\n"
        )


        file.write(

            f"ACTUAL REAL   "
            f"{results['true_negatives']:5d}    "
            f"{results['false_positives']:5d}\n"
        )


        file.write(

            f"ACTUAL FAKE   "
            f"{results['false_negatives']:5d}    "
            f"{results['true_positives']:5d}\n\n"
        )


        file.write(

            f"True Negatives: "
            f"{results['true_negatives']}\n"
        )


        file.write(

            f"False Positives: "
            f"{results['false_positives']}\n"
        )


        file.write(

            f"False Negatives: "
            f"{results['false_negatives']}\n"
        )


        file.write(

            f"True Positives: "
            f"{results['true_positives']}\n\n"
        )


        # ----------------------------------------------------
        # INTERPRETATION
        # ----------------------------------------------------

        file.write(

            "RESULT INTERPRETATION\n"
        )

        file.write(

            "-" * 40 + "\n"
        )


        file.write(

            "The Multimodal LSTM model analyzes "
            "lip movement features and MFCC audio "
            "features to detect potential deepfake "
            "lip-sync anomalies.\n\n"
        )


        file.write(

            f"The model achieved a test accuracy of "
            f"{results['accuracy'] * 100:.2f}%.\n"
        )


        file.write(

            f"The ROC-AUC score is "
            f"{results['roc_auc']:.4f}.\n"
        )


        file.write(

            "The results demonstrate that the model "
            "can distinguish between REAL and FAKE "
            "videos on the test dataset.\n"
        )


    print(

        "Text summary saved:"
    )

    print(

        SUMMARY_PATH
    )

    print()


# ============================================================
# SAVE JSON SUMMARY
# ============================================================

def save_json_summary(

    results

):

    summary = {

        "project_title":

            PROJECT_TITLE,


        "model_information": {

            "model":

                MODEL_NAME,


            "features":

                FEATURE_INFORMATION,


            "total_features":

                TOTAL_FEATURES,


            "maximum_sequence_length":

                MAX_SEQUENCE_LENGTH

        },


        "results":

            results

    }


    with open(

        JSON_SUMMARY_PATH,

        "w",

        encoding="utf-8"

    ) as file:

        json.dump(

            summary,

            file,

            indent=4
        )


    print(

        "JSON summary saved:"
    )

    print(

        JSON_SUMMARY_PATH
    )

    print()


# ============================================================
# PRINT FINAL SUMMARY
# ============================================================

def print_final_summary(

    results

):

    print()

    print("=" * 70)

    print(
        "FINAL PROJECT RESULTS"
    )

    print("=" * 70)

    print()

    print(

        f"Test Accuracy: "
        f"{results['accuracy'] * 100:.2f}%"
    )

    print(

        f"Precision: "
        f"{results['precision'] * 100:.2f}%"
    )

    print(

        f"Recall: "
        f"{results['recall'] * 100:.2f}%"
    )

    print(

        f"F1 Score: "
        f"{results['f1_score'] * 100:.2f}%"
    )

    if results["roc_auc"] is not None:

        print(

            f"ROC-AUC: "
            f"{results['roc_auc']:.4f}"
        )

    print()

    print(

        "Confusion Matrix:"
    )

    print()

    print(

        f"TN = "
        f"{results['true_negatives']}"
    )

    print(

        f"FP = "
        f"{results['false_positives']}"
    )

    print(

        f"FN = "
        f"{results['false_negatives']}"
    )

    print(

        f"TP = "
        f"{results['true_positives']}"
    )

    print()


# ============================================================
# MAIN
# ============================================================

def main():

    check_required_files()


    video_ids, y_true, y_pred, probabilities = (

        load_predictions()
    )


    results = (

        calculate_results(

            y_true,

            y_pred,

            probabilities
        )
    )


    save_text_summary(

        results
    )


    save_json_summary(

        results
    )


    print_final_summary(

        results
    )


    print("=" * 70)

    print(
        "RESULTS SUMMARY COMPLETE"
    )

    print("=" * 70)

    print()

    print(

        "Output directory:"
    )

    print(

        EVALUATION_ROOT
    )

    print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()