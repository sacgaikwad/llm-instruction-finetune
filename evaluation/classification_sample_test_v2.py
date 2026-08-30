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

SCAM_TOKENS = TOKENIZER.encode(" SCAM")
LEGIT_TOKENS = TOKENIZER.encode(" LEGIT")


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
# LABEL PROBABILITY
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
                log_probs[
                    0,
                    token_id
                ]
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
    print(" CLASSIFICATION SAMPLE TEST V2")
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
# CLASSIFY
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

    if scam_log_probability > legit_log_probability:

        prediction = "SCAM"

    else:

        prediction = "LEGIT"

    return (
        scam_log_probability,
        legit_log_probability,
        difference,
        prediction
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
# MAIN
# ============================================================

def main():

    model = load_model()

    print("\n===== LABEL TOKENIZATION =====")

    print(
        f"SCAM  ' SCAM'  -> {SCAM_TOKENS}"
    )

    print(
        f"LEGIT ' LEGIT' -> {LEGIT_TOKENS}"
    )

    print("\n========================================")
    print(" RUNNING SAMPLE CLASSIFICATION")
    print("========================================")

    for index, message in enumerate(
        SAMPLE_MESSAGES,
        start=1
    ):

        (
            scam_log_probability,
            legit_log_probability,
            difference,
            prediction
        ) = classify_message(
            model,
            message
        )

        print("\n----------------------------------------")

        print(
            f"Sample {index}/{len(SAMPLE_MESSAGES)}"
        )

        print(
            f"SCAM log-prob  : "
            f"{scam_log_probability:.8f}"
        )

        print(
            f"LEGIT log-prob : "
            f"{legit_log_probability:.8f}"
        )

        print(
            f"Difference     : "
            f"{difference:.8f}"
        )

        print(
            f"Prediction     : {prediction}"
        )

        print(
            f"Input          : {message}"
        )

    print("\n========================================")
    print(" SAMPLE TEST COMPLETE")
    print("========================================")


if __name__ == "__main__":

    main()