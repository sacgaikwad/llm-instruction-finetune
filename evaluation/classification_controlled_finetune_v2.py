from pathlib import Path
import json

import torch
import torch.nn as nn
import tiktoken

from pathlib import Path
import json
import random

import torch
# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

V2_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "v2"
    / "instruction_finetuned_v2_best.pth"
)

TRAIN_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "classification_controlled"
    / "train.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "models"
    / "controlled"
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "classification_controlled_v2_best.pth"
)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# CONFIGURATION
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

LEARNING_RATE = 1e-5
EPOCHS = 3
BATCH_SIZE = 2
SEED = 42

random.seed(SEED)
torch.manual_seed(SEED)

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
# LOAD MODEL
# ============================================================

def load_model():

    print("\n========================================")
    print(" LOADING V2 CHECKPOINT")
    print("========================================")

    if not V2_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"V2 checkpoint not found:\n{V2_MODEL_PATH}"
        )

    from src.model.gpt_model import Model

    model = Model(CONFIG)

    checkpoint = torch.load(
        V2_MODEL_PATH,
        map_location=DEVICE
    )

    result = model.load_state_dict(checkpoint)

    print("Checkpoint:", result)

    model.to(DEVICE)

    return model


# ============================================================
# LOAD DATA
# ============================================================

def load_dataset():

    print("\n========================================")
    print(" LOADING CONTROLLED DATASET")
    print("========================================")

    if not TRAIN_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Training dataset not found:\n{TRAIN_DATA_PATH}"
        )

    with open(
        TRAIN_DATA_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        data = json.load(file)

    scam_count = sum(
        1 for item in data
        if item["output"].strip().upper() == "SCAM"
    )

    legit_count = sum(
        1 for item in data
        if item["output"].strip().upper() == "LEGIT"
    )

    print(f"Training examples: {len(data)}")
    print(f"SCAM : {scam_count}")
    print(f"LEGIT: {legit_count}")

    return data


# ============================================================
# CREATE CLASSIFICATION TARGET
# ============================================================

def prepare_example(entry):

    prompt = format_prompt(entry)

    label = entry["output"].strip().upper()

    if label not in LABELS:
        raise ValueError(
            f"Unexpected label: {label}"
        )

    prompt_tokens = TOKENIZER.encode(prompt)
    label_tokens = LABELS[label]

    # --------------------------------------------------------
    # Complete sequence:
    #
    # prompt + label
    #
    # Example:
    #
    # ### Response:
    # SCAM
    # --------------------------------------------------------

    full_tokens = (
        prompt_tokens +
        label_tokens
    )

    input_ids = full_tokens[:-1]
    target_ids = full_tokens[1:]

    # --------------------------------------------------------
    # MASK EVERYTHING EXCEPT LABEL TOKENS
    # --------------------------------------------------------

    masked_targets = [-100] * len(target_ids)

    prompt_length = len(prompt_tokens)

    # The label begins at prompt_length.
    #
    # target_ids is shifted by one position, therefore
    # the label prediction positions begin at:
    #
    # prompt_length - 1
    # --------------------------------------------------------

    label_start = prompt_length - 1

    for i in range(len(label_tokens)):

        position = label_start + i

        if position < len(masked_targets):
            masked_targets[position] = label_tokens[i]

    return (
        torch.tensor(input_ids, dtype=torch.long),
        torch.tensor(masked_targets, dtype=torch.long),
        label
    )


# ============================================================
# PAD BATCH
# ============================================================

def create_batch(examples):

    max_length = max(
        len(example[0])
        for example in examples
    )

    batch_inputs = []
    batch_targets = []

    for input_ids, target_ids, _ in examples:

        padding = max_length - len(input_ids)

        padded_inputs = torch.cat(
            [
                input_ids,
                torch.zeros(
                    padding,
                    dtype=torch.long
                )
            ]
        )

        padded_targets = torch.cat(
            [
                target_ids,
                torch.full(
                    (padding,),
                    -100,
                    dtype=torch.long
                )
            ]
        )

        batch_inputs.append(
            padded_inputs
        )

        batch_targets.append(
            padded_targets
        )

    return (
        torch.stack(batch_inputs).to(DEVICE),
        torch.stack(batch_targets).to(DEVICE)
    )


# ============================================================
# CREATE BALANCED EPOCH
# ============================================================

def create_balanced_epoch(data):

    scam_examples = [
        item
        for item in data
        if item["output"].strip().upper() == "SCAM"
    ]

    legit_examples = [
        item
        for item in data
        if item["output"].strip().upper() == "LEGIT"
    ]

    if len(scam_examples) != len(legit_examples):

        raise ValueError(
            "Balanced training requires equal "
            "numbers of SCAM and LEGIT examples."
        )

    random.shuffle(scam_examples)
    random.shuffle(legit_examples)

    balanced_data = []

    for scam, legit in zip(
        scam_examples,
        legit_examples
    ):

        balanced_data.append(scam)
        balanced_data.append(legit)

    return balanced_data


# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_epoch(
    model,
    data,
    optimizer,
    loss_function,
    epoch
):

    model.train()

    epoch_data = create_balanced_epoch(
        data
    )

    total_loss = 0.0
    steps = 0

    for start in range(
        0,
        len(epoch_data),
        BATCH_SIZE
    ):

        batch_data = epoch_data[
            start:start + BATCH_SIZE
        ]

        prepared = [
            prepare_example(item)
            for item in batch_data
        ]

        input_ids, target_ids = create_batch(
            prepared
        )

        optimizer.zero_grad()

        logits = model(
            input_ids
        )

        loss = loss_function(
            logits.reshape(
                -1,
                CONFIG["vocab_size"]
            ),
            target_ids.reshape(-1)
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()
        steps += 1

        if steps % 10 == 0:

            total_steps = (
                len(epoch_data)
                + BATCH_SIZE
                - 1
            ) // BATCH_SIZE

            print(
                f"Epoch {epoch}/{EPOCHS} "
                f"| Step {steps}/{total_steps} "
                f"| Loss {loss.item():.4f}"
            )

    return total_loss / steps


# ============================================================
# AUDIT TARGETS
# ============================================================

def audit_targets(data):

    print("\n========================================")
    print(" TARGET AUDIT")
    print("========================================")

    for label in ["SCAM", "LEGIT"]:

        examples = [
            item for item in data
            if item["output"].strip().upper() == label
        ]

        if not examples:
            continue

        input_ids, target_ids, _ = prepare_example(
            examples[0]
        )

        active_positions = (
            target_ids != -100
        ).nonzero(
            as_tuple=True
        )[0]

        print(f"\n{label}")

        print(
            "Label tokens:",
            LABELS[label]
        )

        print(
            "Active target positions:",
            active_positions.tolist()
        )

        print(
            "Active target tokens:",
            target_ids[
                active_positions
            ].tolist()
        )

        decoded = TOKENIZER.decode(
            target_ids[
                active_positions
            ].tolist()
        )

        print(
            "Decoded target:",
            repr(decoded)
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n========================================")
    print(" CONTROLLED CLASSIFICATION FINE-TUNING V2")
    print("========================================")

    print(
        f"Device       : {DEVICE}"
    )

    print(
        f"Learning rate: {LEARNING_RATE}"
    )

    print(
        f"Epochs       : {EPOCHS}"
    )

    print(
        f"Batch size   : {BATCH_SIZE}"
    )

    print(
        "\nLoss objective:"
    )

    print(
        "ONLY SCAM / LEGIT label tokens"
    )
    print(
        "Balanced batches: 1 SCAM + 1 LEGIT"
    )
    print(
        f"Random seed: {SEED}"
    )
    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = load_model()

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    data = load_dataset()

    # --------------------------------------------------------
    # Audit targets before training
    # --------------------------------------------------------

    audit_targets(data)

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE
    )

    loss_function = nn.CrossEntropyLoss(
        ignore_index=-100
    )

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    best_loss = float("inf")

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    print("\n========================================")
    print(" STARTING TRAINING")
    print("========================================")

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        average_loss = train_epoch(
            model,
            data,
            optimizer,
            loss_function,
            epoch
        )

        print(
            "\n----------------------------------------"
        )

        print(
            f"Epoch {epoch} complete"
        )

        print(
            f"Average loss: {average_loss:.4f}"
        )

        if average_loss < best_loss:

            best_loss = average_loss

            torch.save(
                model.state_dict(),
                OUTPUT_PATH
            )

            print(
                "Best checkpoint saved:"
            )

            print(
                OUTPUT_PATH
            )

    # --------------------------------------------------------
    # Complete
    # --------------------------------------------------------

    print("\n========================================")
    print(" CONTROLLED FINE-TUNING V2 COMPLETE")
    print("========================================")

    print(
        f"Best training loss: {best_loss:.4f}"
    )

    print(
        "Checkpoint:"
    )

    print(
        OUTPUT_PATH
    )


if __name__ == "__main__":
    main()