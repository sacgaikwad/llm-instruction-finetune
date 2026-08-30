from pathlib import Path
import json
import math

import torch
import tiktoken

from src.model.gpt_model import Model


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "controlled"
    / "classification_controlled_v2_best.pth"
)

TEST_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "classification_controlled"
    / "test.json"
)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

CONFIG = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 1024,
    "n_layers": 24,
    "n_heads": 16,
    "drop_rate": 0.0,
    "qkv_bias": True,
}


# ============================================================
# TOKENIZER
# ============================================================

TOKENIZER = tiktoken.get_encoding("gpt2")

SCAM_TOKENS = TOKENIZER.encode(" SCAM")
LEGIT_TOKENS = TOKENIZER.encode(" LEGIT")


# ============================================================
# CLASSIFICATION THRESHOLDS
# ============================================================

HIGH_CONFIDENCE = 0.90
MEDIUM_CONFIDENCE = 0.70


# ============================================================
# PROMPT
# ============================================================

def format_prompt(message):

    return (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request."
        "\n\n"
        "### Instruction:\n"
        "Classify the following message as SCAM or LEGIT. "
        "Return exactly one label."
        "\n\n"
        "## Input:\n"
        f"{message}"
        "\n\n"
        "### Response:\n"
    )


# ============================================================
# LABEL LOG PROBABILITY
# ============================================================

def label_log_probability(
    model,
    input_ids,
    label_tokens
):

    current_ids = input_ids.clone()

    total_log_probability = 0.0

    with torch.no_grad():

        for token_id in label_tokens:

            logits = model(current_ids)

            next_token_logits = logits[:, -1, :]

            log_probs = torch.log_softmax(
                next_token_logits,
                dim=-1
            )

            token_log_probability = (
                log_probs[0, token_id]
            )

            total_log_probability += (
                token_log_probability.item()
            )

            next_token = torch.tensor(
                [[token_id]],
                dtype=torch.long,
                device=DEVICE
            )

            current_ids = torch.cat(
                [
                    current_ids,
                    next_token
                ],
                dim=1
            )

    return total_log_probability


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    print("========================================")
    print(" CLASSIFICATION UNSEEN TEST V2")
    print("========================================")

    print(f"Device: {DEVICE}")

    print("\n===== LOADING MODEL =====")
    print(f"Checkpoint: {MODEL_PATH}")

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Checkpoint not found:\n{MODEL_PATH}"
        )

    model = Model(CONFIG)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    result = model.load_state_dict(
        checkpoint,
        strict=True
    )

    print(f"Checkpoint: {result}")

    model.to(DEVICE)

    model.eval()

    print("Model loaded successfully.")

    return model


# ============================================================
# LOAD TEST DATA
# ============================================================

def load_test_data():

    print("\n===== LOADING UNSEEN TEST DATA =====")
    print(f"Dataset: {TEST_DATA_PATH}")

    if not TEST_DATA_PATH.exists():

        raise FileNotFoundError(
            f"Test dataset not found:\n{TEST_DATA_PATH}"
        )

    with open(
        TEST_DATA_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        data = json.load(file)

    print(f"Test examples: {len(data)}")

    scam_count = sum(
        1
        for item in data
        if item["output"] == "SCAM"
    )

    legit_count = sum(
        1
        for item in data
        if item["output"] == "LEGIT"
    )

    print(f"SCAM : {scam_count}")
    print(f"LEGIT: {legit_count}")

    return data


# ============================================================
# CLASSIFY MESSAGE
# ============================================================

def classify_message(
    model,
    message
):

    prompt = format_prompt(message)

    encoded = TOKENIZER.encode(prompt)

    if len(encoded) >= CONFIG["context_length"] - 2:

        raise ValueError(
            "Input is too long for the model context window."
        )

    input_ids = torch.tensor(
        [encoded],
        dtype=torch.long,
        device=DEVICE
    )

    scam_log_probability = (
        label_log_probability(
            model,
            input_ids,
            SCAM_TOKENS
        )
    )

    legit_log_probability = (
        label_log_probability(
            model,
            input_ids,
            LEGIT_TOKENS
        )
    )

    difference = (
        scam_log_probability
        - legit_log_probability
    )

    # --------------------------------------------------------
    # Convert log-probability difference into probability
    #
    # P(SCAM) =
    #     1 / (1 + exp(-difference))
    # --------------------------------------------------------

    try:

        scam_probability = (
            1.0
            / (
                1.0
                + math.exp(-difference)
            )
        )

    except OverflowError:

        scam_probability = (
            0.0
            if difference < 0
            else 1.0
        )

    legit_probability = (
        1.0 - scam_probability
    )

    # --------------------------------------------------------
    # Determine most likely class
    # --------------------------------------------------------

    if scam_probability >= 0.5:

        confidence = scam_probability
        predicted_label = "SCAM"

    else:

        confidence = legit_probability
        predicted_label = "LEGIT"

    # --------------------------------------------------------
    # Confidence level
    # --------------------------------------------------------

    if confidence >= HIGH_CONFIDENCE:

        confidence_level = "HIGH"

    elif confidence >= MEDIUM_CONFIDENCE:

        confidence_level = "MEDIUM"

    else:

        confidence_level = "LOW"

    # --------------------------------------------------------
    # Classification decision
    #
    # We do not call LOW-confidence predictions reliable.
    # --------------------------------------------------------

    if confidence < MEDIUM_CONFIDENCE:

        prediction = "UNCERTAIN"

    else:

        prediction = predicted_label

    return {
        "prediction": prediction,
        "predicted_label": predicted_label,
        "scam_log_probability": scam_log_probability,
        "legit_log_probability": legit_log_probability,
        "difference": difference,
        "scam_probability": scam_probability,
        "legit_probability": legit_probability,
        "confidence": confidence,
        "confidence_level": confidence_level,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    model = load_model()

    test_data = load_test_data()

    print("\n===== LABEL TOKENIZATION =====")

    print(
        f"SCAM  ' SCAM'  -> {SCAM_TOKENS}"
    )

    print(
        f"LEGIT ' LEGIT' -> {LEGIT_TOKENS}"
    )

    print("\n========================================")
    print(" RUNNING UNSEEN TEST")
    print("========================================")

    # --------------------------------------------------------
    # Overall statistics
    # --------------------------------------------------------

    total = 0
    correct = 0

    # --------------------------------------------------------
    # Per-class statistics
    # --------------------------------------------------------

    scam_total = 0
    scam_correct = 0

    legit_total = 0
    legit_correct = 0

    # --------------------------------------------------------
    # Confusion matrix
    #
    # UNCERTAIN is intentionally included.
    # --------------------------------------------------------

    confusion = {
        "SCAM": {
            "SCAM": 0,
            "LEGIT": 0,
            "UNCERTAIN": 0,
        },
        "LEGIT": {
            "SCAM": 0,
            "LEGIT": 0,
            "UNCERTAIN": 0,
        },
    }

    # --------------------------------------------------------
    # Confidence statistics
    # --------------------------------------------------------

    high_confidence = 0
    medium_confidence = 0
    low_confidence = 0

    uncertain_count = 0

    # ========================================================
    # TEST LOOP
    # ========================================================

    for index, entry in enumerate(
        test_data,
        start=1
    ):

        expected = entry["output"]
        message = entry.get("input", "")

        result = classify_message(
            model,
            message
        )

        prediction = result["prediction"]

        # ----------------------------------------------------
        # Correctness
        #
        # UNCERTAIN is NOT counted as correct even when
        # the underlying predicted_label matches.
        # ----------------------------------------------------

        is_correct = (
            prediction == expected
        )

        total += 1

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

        # ----------------------------------------------------
        # Confusion matrix
        # ----------------------------------------------------

        confusion[expected][prediction] += 1

        # ----------------------------------------------------
        # Confidence statistics
        # ----------------------------------------------------

        if result["confidence_level"] == "HIGH":

            high_confidence += 1

        elif result["confidence_level"] == "MEDIUM":

            medium_confidence += 1

        else:

            low_confidence += 1

        if prediction == "UNCERTAIN":

            uncertain_count += 1

        # ----------------------------------------------------
        # Print result
        # ----------------------------------------------------

        print("\n----------------------------------------")

        print(
            f"Test {index}/{len(test_data)}"
        )

        print(
            f"Expected       : {expected}"
        )

        print(
            f"Prediction     : {prediction}"
        )

        print(
            f"Result         : "
            f"{'CORRECT' if is_correct else 'WRONG'}"
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
            f"Input          : {message}"
        )

    # ========================================================
    # FINAL RESULTS
    # ========================================================

    accuracy = (
        correct / total * 100
        if total > 0
        else 0
    )

    scam_accuracy = (
        scam_correct / scam_total * 100
        if scam_total > 0
        else 0
    )

    legit_accuracy = (
        legit_correct / legit_total * 100
        if legit_total > 0
        else 0
    )

    print("\n========================================")
    print(" FINAL UNSEEN TEST RESULTS")
    print("========================================")

    print(
        f"Total tests : {total}"
    )

    print(
        f"Correct     : {correct}/{total}"
    )

    print(
        f"Accuracy    : {accuracy:.2f}%"
    )

    print("\nSCAM")

    print(
        f"Correct     : "
        f"{scam_correct}/{scam_total}"
    )

    print(
        f"Accuracy    : "
        f"{scam_accuracy:.2f}%"
    )

    print("\nLEGIT")

    print(
        f"Correct     : "
        f"{legit_correct}/{legit_total}"
    )

    print(
        f"Accuracy    : "
        f"{legit_accuracy:.2f}%"
    )

    # ========================================================
    # CONFIDENCE SUMMARY
    # ========================================================

    print("\n========================================")
    print(" CONFIDENCE SUMMARY")
    print("========================================")

    print(
        f"HIGH confidence   : {high_confidence}"
    )

    print(
        f"MEDIUM confidence : {medium_confidence}"
    )

    print(
        f"LOW confidence    : {low_confidence}"
    )

    print(
        f"UNCERTAIN         : {uncertain_count}"
    )

    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    print("\n========================================")
    print(" CONFUSION MATRIX")
    print("========================================")

    print(
        "Expected \\ Predicted"
    )

    print(
        f"{'':15}"
        f"{'SCAM':10}"
        f"{'LEGIT':10}"
        f"{'UNCERTAIN':12}"
    )

    print(
        f"{'SCAM':15}"
        f"{confusion['SCAM']['SCAM']:<10}"
        f"{confusion['SCAM']['LEGIT']:<10}"
        f"{confusion['SCAM']['UNCERTAIN']:<12}"
    )

    print(
        f"{'LEGIT':15}"
        f"{confusion['LEGIT']['SCAM']:<10}"
        f"{confusion['LEGIT']['LEGIT']:<10}"
        f"{confusion['LEGIT']['UNCERTAIN']:<12}"
    )

    print("\n========================================")
    print(" UNSEEN TEST COMPLETE")
    print("========================================")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()