from pathlib import Path
import json

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
# LOAD MODEL
# ============================================================

def load_model():

    print("\n========================================")
    print(" LOADING CONTROLLED V2 MODEL")
    print("========================================")

    print(f"Model:\n{MODEL_PATH}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found:\n{MODEL_PATH}"
        )

    model = Model(CONFIG)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    result = model.load_state_dict(
        checkpoint
    )

    print(
        "\nCheckpoint:",
        result
    )

    model.to(DEVICE)
    model.eval()

    print("\nModel loaded successfully.")

    return model


# ============================================================
# LOAD VALIDATION DATA
# ============================================================

def load_validation_data():

    print("\n========================================")
    print(" LOADING CONTROLLED VALIDATION DATA")
    print("========================================")

    print(f"Dataset:\n{VALIDATION_PATH}")

    if not VALIDATION_PATH.exists():
        raise FileNotFoundError(
            f"Validation dataset not found:\n{VALIDATION_PATH}"
        )

    with open(
        VALIDATION_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        data = json.load(file)

    print(
        f"\nValidation examples: {len(data)}"
    )

    scam_count = sum(
        1
        for entry in data
        if entry.get("output", "").strip().upper() == "SCAM"
    )

    legit_count = sum(
        1
        for entry in data
        if entry.get("output", "").strip().upper() == "LEGIT"
    )

    print(f"SCAM : {scam_count}")
    print(f"LEGIT: {legit_count}")

    return data


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

    prompt += "\n\n### Response:\n"

    return prompt


# ============================================================
# GENERATION
# ============================================================

def generate_response(
    model,
    prompt,
    max_new_tokens=10
):

    token_ids = TOKENIZER.encode(
        prompt
    )

    input_tensor = torch.tensor(
        token_ids,
        dtype=torch.long,
        device=DEVICE
    ).unsqueeze(0)

    prompt_length = input_tensor.shape[1]

    with torch.no_grad():

        for _ in range(max_new_tokens):

            if input_tensor.shape[1] >= CONFIG["context_length"]:
                break

            logits = model(
                input_tensor
            )

            next_token_logits = (
                logits[:, -1, :]
            )

            next_token_id = torch.argmax(
                next_token_logits,
                dim=-1,
                keepdim=True
            )

            if next_token_id.item() == 50256:
                break

            input_tensor = torch.cat(
                [
                    input_tensor,
                    next_token_id
                ],
                dim=1
            )

    generated_tokens = (
        input_tensor[
            :,
            prompt_length:
        ]
    )

    response = TOKENIZER.decode(
        generated_tokens
        .squeeze(0)
        .tolist()
    )

    return response.strip()


# ============================================================
# EXTRACT CLASSIFICATION LABEL
# ============================================================

def extract_label(response):

    text = response.upper()

    if "SCAM" in text and "LEGIT" not in text:
        return "SCAM"

    if "LEGIT" in text and "SCAM" not in text:
        return "LEGIT"

    if "SCAM" in text and "LEGIT" in text:
        return "AMBIGUOUS"

    return "NONE"


# ============================================================
# MAIN EVALUATION
# ============================================================

def main():

    print("\n========================================")
    print(" CONTROLLED CLASSIFICATION VALIDATION")
    print("========================================")

    print(f"Device: {DEVICE}")

    model = load_model()

    data = load_validation_data()

    print("\n========================================")
    print(" RUNNING VALIDATION")
    print("========================================")

    total = len(data)

    correct = 0

    scam_total = 0
    scam_correct = 0

    legit_total = 0
    legit_correct = 0

    confusion = {
        "SCAM": {
            "SCAM": 0,
            "LEGIT": 0,
            "OTHER": 0
        },
        "LEGIT": {
            "SCAM": 0,
            "LEGIT": 0,
            "OTHER": 0
        }
    }

    for index, entry in enumerate(
        data,
        start=1
    ):

        expected = entry.get(
            "output",
            ""
        ).strip().upper()

        prompt = format_prompt(
            entry
        )

        generated = generate_response(
            model,
            prompt
        )

        predicted = extract_label(
            generated
        )

        is_correct = (
            predicted == expected
        )

        if is_correct:
            correct += 1

        if expected == "SCAM":

            scam_total += 1

            if is_correct:
                scam_correct += 1

            if predicted == "SCAM":
                confusion["SCAM"]["SCAM"] += 1

            elif predicted == "LEGIT":
                confusion["SCAM"]["LEGIT"] += 1

            else:
                confusion["SCAM"]["OTHER"] += 1

        elif expected == "LEGIT":

            legit_total += 1

            if is_correct:
                legit_correct += 1

            if predicted == "SCAM":
                confusion["LEGIT"]["SCAM"] += 1

            elif predicted == "LEGIT":
                confusion["LEGIT"]["LEGIT"] += 1

            else:
                confusion["LEGIT"]["OTHER"] += 1

        print("\n----------------------------------------")

        print(
            f"Test {index}/{total}"
        )

        print(
            f"Expected   : {expected}"
        )

        print(
            f"Prediction : {predicted}"
        )

        print(
            f"Result     : "
            f"{'CORRECT' if is_correct else 'WRONG'}"
        )

        print(
            f"Generated  : {generated!r}"
        )

        if not is_correct:

            input_text = entry.get(
                "input",
                ""
            )

            print(
                f"Input      : {input_text}"
            )

    # ========================================================
    # RESULTS
    # ========================================================

    accuracy = (
        correct / total * 100
        if total
        else 0
    )

    scam_accuracy = (
        scam_correct / scam_total * 100
        if scam_total
        else 0
    )

    legit_accuracy = (
        legit_correct / legit_total * 100
        if legit_total
        else 0
    )

    print("\n========================================")
    print(" FINAL CONTROLLED VALIDATION RESULTS")
    print("========================================")

    print(
        f"\nTotal tests : {total}"
    )

    print(
        f"Correct     : "
        f"{correct}/{total}"
    )

    print(
        f"Accuracy    : "
        f"{accuracy:.2f}%"
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
    # CONFUSION MATRIX
    # ========================================================

    print("\n========================================")
    print(" CONFUSION MATRIX")
    print("========================================")

    print(
        "\nExpected \\ Predicted"
    )

    print(
        f"SCAM  -> "
        f"SCAM={confusion['SCAM']['SCAM']}, "
        f"LEGIT={confusion['SCAM']['LEGIT']}, "
        f"OTHER={confusion['SCAM']['OTHER']}"
    )

    print(
        f"LEGIT -> "
        f"SCAM={confusion['LEGIT']['SCAM']}, "
        f"LEGIT={confusion['LEGIT']['LEGIT']}, "
        f"OTHER={confusion['LEGIT']['OTHER']}"
    )

    print("\n========================================")
    print(" VALIDATION COMPLETE")
    print("========================================")


if __name__ == "__main__":
    main()