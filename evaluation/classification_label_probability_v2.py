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

VALIDATION_PATH = (
    PROJECT_ROOT
    / "data"
    / "classification_controlled"
    / "validation.json"
)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# MODEL CONFIG
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


# ============================================================
# LABELS
# ============================================================

SCAM_TOKENS = TOKENIZER.encode(" SCAM")
LEGIT_TOKENS = TOKENIZER.encode(" LEGIT")


# ============================================================
# PROMPT
# ============================================================

def format_prompt(entry):

    instruction = entry["instruction"]
    input_text = entry.get("input", "")

    prompt = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request."
        "\n\n"
        "### Instruction:\n"
        f"{instruction}"
    )

    if input_text:

        prompt += (
            "\n\n"
            "## Input:\n"
            f"{input_text}"
        )

    prompt += (
        "\n\n"
        "### Response:\n"
    )

    return prompt


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    print("\n========================================")
    print(" LOADING CONTROLLED V2 MODEL")
    print("========================================")

    print(f"Checkpoint:\n{MODEL_PATH}")

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
# LOAD DATA
# ============================================================

def load_validation_data():

    print("\n========================================")
    print(" LOADING VALIDATION DATA")
    print("========================================")

    with open(
        VALIDATION_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        data = json.load(file)

    print(f"Validation examples: {len(data)}")

    scam_count = sum(
        1
        for item in data
        if item["output"].strip().upper() == "SCAM"
    )

    legit_count = sum(
        1
        for item in data
        if item["output"].strip().upper() == "LEGIT"
    )

    print(f"SCAM : {scam_count}")
    print(f"LEGIT: {legit_count}")

    return data


# ============================================================
# LABEL LOG PROBABILITY
# ============================================================

def label_log_probability(
    model,
    prompt,
    label_tokens
):

    prompt_tokens = TOKENIZER.encode(
        prompt
    )

    input_ids = torch.tensor(
        [prompt_tokens],
        dtype=torch.long,
        device=DEVICE
    )

    total_log_probability = 0.0

    current_ids = input_ids

    with torch.no_grad():

        for token_id in label_tokens:

            logits = model(
                current_ids
            )

            next_token_logits = logits[
                0,
                -1,
                :
            ]

            log_probs = torch.log_softmax(
                next_token_logits,
                dim=-1
            )

            token_log_probability = (
                log_probs[token_id].item()
            )

            total_log_probability += (
                token_log_probability
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
# MAIN
# ============================================================

def main():

    print("========================================")
    print(" CLASSIFICATION LABEL PROBABILITY V2")
    print("========================================")

    print(f"Device: {DEVICE}")

    print("\n===== LABEL TOKENIZATION =====")

    print(
        f"SCAM  ' SCAM'  -> {SCAM_TOKENS}"
    )

    print(
        f"LEGIT ' LEGIT' -> {LEGIT_TOKENS}"
    )

    model = load_model()

    data = load_validation_data()

    print("\n========================================")
    print(" RUNNING LABEL PROBABILITY TEST")
    print("========================================")

    total_correct = 0

    scam_correct = 0
    legit_correct = 0

    scam_total = 0
    legit_total = 0

    for index, entry in enumerate(data, start=1):

        expected = (
            entry["output"]
            .strip()
            .upper()
        )

        prompt = format_prompt(entry)

        scam_log_prob = label_log_probability(
            model,
            prompt,
            SCAM_TOKENS
        )

        legit_log_prob = label_log_probability(
            model,
            prompt,
            LEGIT_TOKENS
        )

        if scam_log_prob > legit_log_prob:
            prediction = "SCAM"
        else:
            prediction = "LEGIT"

        correct = (
            prediction == expected
        )

        if correct:
            total_correct += 1

        if expected == "SCAM":

            scam_total += 1

            if correct:
                scam_correct += 1

        elif expected == "LEGIT":

            legit_total += 1

            if correct:
                legit_correct += 1

        difference = (
            scam_log_prob
            - legit_log_prob
        )

        print("\n----------------------------------------")
        print(f"Test {index}/{len(data)}")
        print(f"Expected       : {expected}")
        print(
            f"SCAM log-prob  : "
            f"{scam_log_prob:.8f}"
        )
        print(
            f"LEGIT log-prob : "
            f"{legit_log_prob:.8f}"
        )
        print(
            f"Difference     : "
            f"{difference:.8f}"
        )
        print(
            f"Prediction     : {prediction}"
        )
        print(
            f"Result         : "
            f"{'CORRECT' if correct else 'WRONG'}"
        )
        print(
            f"Input          : "
            f"{entry.get('input', '')}"
        )

    overall_accuracy = (
        total_correct / len(data) * 100
    )

    scam_accuracy = (
        scam_correct / scam_total * 100
        if scam_total
        else 0.0
    )

    legit_accuracy = (
        legit_correct / legit_total * 100
        if legit_total
        else 0.0
    )

    print("\n========================================")
    print(" FINAL LABEL PROBABILITY RESULTS")
    print("========================================")

    print(
        f"Total tests : {len(data)}"
    )

    print(
        f"Correct     : "
        f"{total_correct}/{len(data)}"
    )

    print(
        f"Accuracy    : "
        f"{overall_accuracy:.2f}%"
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

    print("\n========================================")
    print(" INTERPRETATION")
    print("========================================")

    if overall_accuracy >= 90:

        print(
            "Strong classification signal detected."
        )

        print(
            "The balanced training appears to have "
            "learned the SCAM vs LEGIT distinction."
        )

    elif scam_accuracy > 0 and legit_accuracy > 0:

        print(
            "Some classification signal is present, "
            "but the model is not yet reliable."
        )

    else:

        print(
            "The model still shows a strong "
            "classification bias."
        )

    print("\n========================================")
    print(" END OF LABEL PROBABILITY TEST")
    print("========================================")


if __name__ == "__main__":
    main()