from pathlib import Path
import json

from evaluation.classification_inference_v2 import (
    load_model,
    classify_message,
    LABELS,
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

VALIDATION_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "classification_controlled"
    / "validation.json"
)

UNSEEN_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "classification_controlled"
    / "test.json"
)


# ============================================================
# SAMPLE MESSAGES
# ============================================================

SAMPLE_MESSAGES = [
    "Your electricity bill is due tomorrow. Please pay through the official portal.",

    "Congratulations! You have won ₹50,000. Pay ₹1,000 to claim your prize.",

    "Your doctor's appointment is confirmed for Monday at 10 AM.",

    "Your bank account will be blocked today. Send your OTP immediately.",

    "The project meeting has been moved to 3 PM.",

    "Pay a small processing fee to receive your approved loan.",

    "Your monthly bank statement is now available in your official account.",

    "We detected unusual activity. Verify your card number and CVV immediately.",

    "Your membership renewal is scheduled for September 15 according to your account settings.",

    "Your refund is ready. Confirm your bank details so we can process the payment.",
]


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset(path):

    if not path.exists():

        raise FileNotFoundError(
            f"Dataset not found:\n{path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


# ============================================================
# PRINT RESULT
# ============================================================

def print_classification_result(
    index,
    total,
    message,
    result,
    expected=None,
):

    print("\n----------------------------------------")

    if expected is not None:

        print(
            f"Test {index}/{total}"
        )

        print(
            f"Expected       : {expected}"
        )

    else:

        print(
            f"Sample {index}/{total}"
        )

    print(
        f"Prediction     : "
        f"{result['prediction']}"
    )

    if expected is not None:

        print(
            f"Result         : "
            f"{'CORRECT' if result['prediction'] == expected else 'WRONG'}"
        )

    print(
        f"SCAM log-prob  : "
        f"{result['scam_log_probability']:.8f}"
    )

    print(
        f"LEGIT log-prob : "
        f"{result['legit_log_probability']:.8f}"
    )

    print(
        f"Difference     : "
        f"{result['difference']:.8f}"
    )

    print(
        f"SCAM probability  : "
        f"{result['scam_probability'] * 100:.2f}%"
    )

    print(
        f"LEGIT probability : "
        f"{result['legit_probability'] * 100:.2f}%"
    )

    print(
        f"Confidence     : "
        f"{result['confidence'] * 100:.2f}%"
    )

    print(
        f"Confidence level : "
        f"{result['confidence_level']}"
    )

    print(
        f"Input          : "
        f"{message}"
    )


# ============================================================
# DATASET EVALUATION
# ============================================================

def evaluate_dataset(
    model,
    dataset,
    test_name,
):

    print("\n========================================")
    print(f" {test_name}")
    print("========================================")

    total = len(dataset)

    correct = 0

    scam_total = 0
    scam_correct = 0

    legit_total = 0
    legit_correct = 0

    uncertain_count = 0

    for index, entry in enumerate(
        dataset,
        start=1
    ):

        message = entry.get(
            "input",
            ""
        )

        expected = entry["output"]

        result = classify_message(
            model,
            message
        )

        prediction = result[
            "prediction"
        ]

        is_correct = (
            prediction == expected
        )

        total_prediction = prediction

        if expected == "SCAM":

            scam_total += 1

            if is_correct:
                scam_correct += 1

        elif expected == "LEGIT":

            legit_total += 1

            if is_correct:
                legit_correct += 1

        if is_correct:
            correct += 1

        if prediction == "UNCERTAIN":
            uncertain_count += 1

        print_classification_result(
            index,
            total,
            message,
            result,
            expected,
        )

    accuracy = (
        correct / total
        if total > 0
        else 0.0
    )

    scam_accuracy = (
        scam_correct / scam_total
        if scam_total > 0
        else 0.0
    )

    legit_accuracy = (
        legit_correct / legit_total
        if legit_total > 0
        else 0.0
    )

    print("\n========================================")
    print(f" {test_name} SUMMARY")
    print("========================================")

    print(
        f"Total tests       : {total}"
    )

    print(
        f"Correct           : {correct}/{total}"
    )

    print(
        f"Accuracy          : {accuracy * 100:.2f}%"
    )

    print(
        f"SCAM              : "
        f"{scam_correct}/{scam_total}"
    )

    print(
        f"SCAM accuracy     : "
        f"{scam_accuracy * 100:.2f}%"
    )

    print(
        f"LEGIT             : "
        f"{legit_correct}/{legit_total}"
    )

    print(
        f"LEGIT accuracy    : "
        f"{legit_accuracy * 100:.2f}%"
    )

    print(
        f"UNCERTAIN         : "
        f"{uncertain_count}"
    )

    return {
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "scam_total": scam_total,
        "scam_correct": scam_correct,
        "scam_accuracy": scam_accuracy,
        "legit_total": legit_total,
        "legit_correct": legit_correct,
        "legit_accuracy": legit_accuracy,
        "uncertain": uncertain_count,
    }


# ============================================================
# SAMPLE EVALUATION
# ============================================================

def evaluate_samples(model):

    print("\n========================================")
    print(" SAMPLE TEST")
    print("========================================")

    total = len(SAMPLE_MESSAGES)

    scam_count = 0
    legit_count = 0
    uncertain_count = 0

    for index, message in enumerate(
        SAMPLE_MESSAGES,
        start=1
    ):

        result = classify_message(
            model,
            message
        )

        prediction = result[
            "prediction"
        ]

        if prediction == "SCAM":

            scam_count += 1

        elif prediction == "LEGIT":

            legit_count += 1

        else:

            uncertain_count += 1

        print_classification_result(
            index,
            total,
            message,
            result,
        )

    print("\n========================================")
    print(" SAMPLE TEST SUMMARY")
    print("========================================")

    print(
        f"Total samples     : {total}"
    )

    print(
        f"SCAM              : {scam_count}"
    )

    print(
        f"LEGIT             : {legit_count}"
    )

    print(
        f"UNCERTAIN         : {uncertain_count}"
    )

    return {
        "total": total,
        "scam": scam_count,
        "legit": legit_count,
        "uncertain": uncertain_count,
    }


# ============================================================
# FINAL SUMMARY
# ============================================================

def print_final_summary(
    validation_result,
    unseen_result,
    sample_result,
):

    print("\n")
    print("========================================")
    print(" FINAL CLASSIFICATION EVALUATION")
    print("========================================")

    print("\nValidation Test")

    print(
        f"  Accuracy       : "
        f"{validation_result['accuracy'] * 100:.2f}%"
    )

    print(
        f"  Correct        : "
        f"{validation_result['correct']}/"
        f"{validation_result['total']}"
    )

    print("\nUnseen Test")

    print(
        f"  Accuracy       : "
        f"{unseen_result['accuracy'] * 100:.2f}%"
    )

    print(
        f"  Correct        : "
        f"{unseen_result['correct']}/"
        f"{unseen_result['total']}"
    )

    print("\nSample Test")

    print(
        f"  Total          : "
        f"{sample_result['total']}"
    )

    print(
        f"  SCAM           : "
        f"{sample_result['scam']}"
    )

    print(
        f"  LEGIT          : "
        f"{sample_result['legit']}"
    )

    print(
        f"  UNCERTAIN      : "
        f"{sample_result['uncertain']}"
    )

    print("\n========================================")
    print(" EVALUATION COMPLETE")
    print("========================================")


# ============================================================
# MAIN
# ============================================================

def main():

    print("========================================")
    print(" CLASSIFICATION EVALUATOR V2")
    print("========================================")

    print(
        f"SCAM label tokens  : "
        f"{LABELS['SCAM']}"
    )

    print(
        f"LEGIT label tokens : "
        f"{LABELS['LEGIT']}"
    )

    model = load_model()

    print("\n===== LOADING VALIDATION DATA =====")

    validation_data = load_dataset(
        VALIDATION_DATA_PATH
    )

    print(
        f"Validation examples: "
        f"{len(validation_data)}"
    )

    print("\n===== LOADING UNSEEN TEST DATA =====")

    unseen_data = load_dataset(
        UNSEEN_DATA_PATH
    )

    print(
        f"Unseen examples: "
        f"{len(unseen_data)}"
    )

    validation_result = evaluate_dataset(
        model,
        validation_data,
        "VALIDATION TEST",
    )

    unseen_result = evaluate_dataset(
        model,
        unseen_data,
        "UNSEEN TEST",
    )

    sample_result = evaluate_samples(
        model
    )

    print_final_summary(
        validation_result,
        unseen_result,
        sample_result,
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()