from pathlib import Path

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


# ============================================================
# LABELS
# ============================================================

LABELS = {
    "SCAM": TOKENIZER.encode(" SCAM"),
    "LEGIT": TOKENIZER.encode(" LEGIT"),
}


# ============================================================
# PROMPT
# ============================================================

def format_prompt(message):

    prompt = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request."
        "\n\n"
        "### Instruction:\n"
        "Classify the following message as SCAM or LEGIT."
        "\n\n"
        "## Input:\n"
        f"{message}"
        "\n\n"
        "### Response:\n"
    )

    return prompt


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    print("\n========================================")
    print(" CLASSIFICATION INFERENCE V2")
    print("========================================")

    print(f"Device: {DEVICE}")

    print("\n===== LOADING MODEL =====")
    print(f"Checkpoint: {MODEL_PATH}")

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
        checkpoint,
        strict=True
    )

    print(f"Checkpoint: {result}")

    model.to(DEVICE)
    model.eval()

    print("Model loaded successfully.")

    return model


# ============================================================
# LABEL LOG-PROBABILITY
# ============================================================

def calculate_label_log_probability(
    model,
    prompt,
    label_tokens
):

    prompt_tokens = TOKENIZER.encode(prompt)

    input_tokens = (
        prompt_tokens
        + label_tokens
    )

    input_tensor = torch.tensor(
        input_tokens,
        dtype=torch.long,
        device=DEVICE
    ).unsqueeze(0)

    with torch.no_grad():

        logits = model(input_tensor)

    prompt_length = len(prompt_tokens)

    total_log_probability = 0.0

    for index, token_id in enumerate(label_tokens):

        prediction_position = (
            prompt_length
            + index
            - 1
        )

        token_logits = logits[
            0,
            prediction_position,
            :
        ]

        log_probabilities = torch.log_softmax(
            token_logits,
            dim=-1
        )

        token_log_probability = (
            log_probabilities[token_id]
            .item()
        )

        total_log_probability += (
            token_log_probability
        )

    return total_log_probability


# ============================================================
# CLASSIFY MESSAGE
# ============================================================

def classify_message(
    model,
    message
):

    prompt = format_prompt(message)

    scam_log_probability = (
        calculate_label_log_probability(
            model,
            prompt,
            LABELS["SCAM"]
        )
    )

    legit_log_probability = (
        calculate_label_log_probability(
            model,
            prompt,
            LABELS["LEGIT"]
        )
    )

    difference = (
        scam_log_probability
        - legit_log_probability
    )

    # --------------------------------------------------------
    # Convert the two log-probabilities into normalized
    # probabilities.
    #
    # P(SCAM) =
    #     exp(SCAM) /
    #     (exp(SCAM) + exp(LEGIT))
    # --------------------------------------------------------

    scam_probability = (
        1.0
        / (1.0 + torch.exp(
            torch.tensor(-difference)
        ).item())
    )

    legit_probability = (
        1.0 - scam_probability
    )

    # --------------------------------------------------------
    # Decision policy
    # --------------------------------------------------------

    if scam_probability >= 0.90:

        prediction = "SCAM"

    elif legit_probability >= 0.90:

        prediction = "LEGIT"

    else:

        prediction = "UNCERTAIN"

    # --------------------------------------------------------
    # Confidence
    # --------------------------------------------------------

    confidence = (
        max(
            scam_probability,
            legit_probability
        )
    )

    if confidence >= 0.90:

        confidence_level = "HIGH"

    elif confidence >= 0.75:

        confidence_level = "MEDIUM"

    else:

        confidence_level = "LOW"

    return {

        "prediction": prediction,

        "scam_log_probability":
            scam_log_probability,

        "legit_log_probability":
            legit_log_probability,

        "difference":
            difference,

        "scam_probability":
            scam_probability,

        "legit_probability":
            legit_probability,

        "confidence":
            confidence,

        "confidence_level":
            confidence_level,
    }

# ============================================================
# PRINT RESULT
# ============================================================

def print_result(
    message,
    result
):

    print("\n----------------------------------------")
    print("CLASSIFICATION RESULT")
    print("----------------------------------------")

    print(
        f"Input          : {message}"
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
        f"Prediction     : "
        f"{result['prediction']}"
    )

    print(
        f"Confidence     : "
        f"{result['confidence'] * 100:.2f}%"
    )

    print(
        f"Confidence level : "
        f"{result['confidence_level']}"
    )

# ============================================================
# INTERACTIVE MODE
# ============================================================

def interactive_mode(model):

    print("\n========================================")
    print(" INTERACTIVE CLASSIFICATION")
    print("========================================")

    print("Enter a message to classify.")
    print("Type 'exit' to stop.")

    while True:

        print()

        message = input(
            "Message: "
        ).strip()

        if message.lower() == "exit":
            break

        if not message:

            print(
                "Please enter a message."
            )

            continue

        result = classify_message(
            model,
            message
        )

        print_result(
            message,
            result
        )


# ============================================================
# MAIN
# ============================================================

def main():

    model = load_model()

    print("\n===== LABEL TOKENIZATION =====")

    print(
        f"SCAM  ' SCAM'  -> "
        f"{LABELS['SCAM']}"
    )

    print(
        f"LEGIT ' LEGIT' -> "
        f"{LABELS['LEGIT']}"
    )

    interactive_mode(model)

    print(
        "\n========================================"
    )

    print(
        " CLASSIFICATION INFERENCE COMPLETE"
    )

    print(
        "========================================"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()