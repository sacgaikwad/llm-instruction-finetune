import torch
import tiktoken

from src.model.config import GPT_CONFIG_355M
from src.model.gpt_model import Model


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = (
    "models/v2/instruction_finetuned_v2_best.pth"
)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(
    f"Device: {device}"
)


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    print(
        "\n===== LOADING V2 BEST MODEL ====="
    )

    model = Model(
        GPT_CONFIG_355M
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device
    )

    model.load_state_dict(
        checkpoint
    )

    model.to(device)

    model.eval()

    print(
        "V2 best model loaded successfully."
    )

    return model


# ============================================================
# TOKENIZER
# ============================================================

def load_tokenizer():

    return tiktoken.get_encoding(
        "gpt2"
    )


# ============================================================
# GENERATE TEXT
# ============================================================

def generate_response(
    model,
    tokenizer,
    prompt,
    max_new_tokens=30
):

    encoded = tokenizer.encode(
        prompt
    )

    input_ids = torch.tensor(
        encoded,
        dtype=torch.long,
        device=device
    ).unsqueeze(0)

    prompt_length = input_ids.shape[1]

    with torch.no_grad():

        for _ in range(
            max_new_tokens
        ):

            logits = model(
                input_ids
            )

            next_token_logits = (
                logits[:, -1, :]
            )

            next_token = torch.argmax(
                next_token_logits,
                dim=-1,
                keepdim=True
            )

            input_ids = torch.cat(
                (
                    input_ids,
                    next_token
                ),
                dim=1
            )

            if (
                next_token.item()
                == tokenizer.eot_token
            ):
                break

            if input_ids.shape[1] >= (
                GPT_CONFIG_355M[
                    "context_length"
                ]
            ):
                break

    generated_tokens = input_ids[
        0,
        prompt_length:
    ].tolist()

    generated_text = tokenizer.decode(
        generated_tokens
    )

    generated_text = generated_text.replace(
        "<|endoftext|>",
        ""
    ).strip()

    return generated_text


# ============================================================
# EXTRACT FIRST CLASSIFICATION LABEL
# ============================================================

def extract_first_label(
    generated_text
):

    text = generated_text.upper()

    labels = [
        "AMBIGUOUS",
        "SCAM",
        "LEGIT"
    ]

    positions = []

    for label in labels:

        position = text.find(
            label
        )

        if position != -1:

            positions.append(
                (
                    position,
                    label
                )
            )

    if not positions:

        return None

    positions.sort(
        key=lambda item: item[0]
    )

    return positions[0][1]


# ============================================================
# EVALUATION CASES
# ============================================================

TEST_CASES = [

    {
        "name": "SCAM - Processing Fee",
        "instruction": (
            "Classify the message as SCAM or LEGIT. "
            "Return exactly one label."
        ),
        "input": (
            "Pay a processing fee before your prize "
            "can be released."
        ),
        "expected": "SCAM"
    },

    {
        "name": "LEGIT - Appointment",
        "instruction": (
            "Classify the message as SCAM or LEGIT. "
            "Return exactly one label."
        ),
        "input": (
            "Your dentist appointment is confirmed "
            "for Tuesday at 11 AM."
        ),
        "expected": "LEGIT"
    },

    {
        "name": "SCAM - Password Request",
        "instruction": (
            "Classify the message as SCAM or LEGIT. "
            "Return exactly one label."
        ),
        "input": (
            "Provide your password immediately or "
            "your account will be permanently suspended."
        ),
        "expected": "SCAM"
    },

    {
        "name": "LEGIT - School Notice",
        "instruction": (
            "Classify the message as SCAM or LEGIT. "
            "Return exactly one label."
        ),
        "input": (
            "The school has announced that the "
            "examination begins at 9 AM on Monday."
        ),
        "expected": "LEGIT"
    },

    {
        "name": "SCAM - Cash Reward",
        "instruction": (
            "Classify the message as SCAM or LEGIT. "
            "Return exactly one label."
        ),
        "input": (
            "Send ₹200 to claim your unexpected "
            "cash reward."
        ),
        "expected": "SCAM"
    }

]


# ============================================================
# BUILD PROMPT
# ============================================================

def build_prompt(
    instruction,
    input_text
):

    return (
        "### Instruction:\n"
        f"{instruction}\n\n"
        "### Input:\n"
        f"{input_text}\n\n"
        "### Response:\n"
    )


# ============================================================
# EVALUATE ONE CASE
# ============================================================

def evaluate_case(
    model,
    tokenizer,
    case
):

    prompt = build_prompt(
        case["instruction"],
        case["input"]
    )

    generated_text = generate_response(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=30
    )

    predicted_label = extract_first_label(
        generated_text
    )

    correct = (
        predicted_label
        == case["expected"]
    )

    print(
        "\n--------------------------------------------------"
    )

    print(
        f"Test: {case['name']}"
    )

    print(
        f"Expected: {case['expected']}"
    )

    print(
        f"Predicted first label: "
        f"{predicted_label}"
    )

    print(
        f"Correct: {correct}"
    )

    print(
        "Generated response:"
    )

    print(
        repr(generated_text)
    )

    return correct


# ============================================================
# MAIN
# ============================================================

def main():

    model = load_model()

    tokenizer = load_tokenizer()

    print(
        "\n===== V2 CLASSIFICATION EVALUATION ====="
    )

    correct_count = 0

    for case in TEST_CASES:

        correct = evaluate_case(
            model,
            tokenizer,
            case
        )

        if correct:

            correct_count += 1

    total_cases = len(
        TEST_CASES
    )

    accuracy = (
        correct_count
        / total_cases
    )

    print(
        "\n=================================================="
    )

    print(
        "V2 CLASSIFICATION RESULTS"
    )

    print(
        f"Correct: "
        f"{correct_count}/{total_cases}"
    )

    print(
        f"First-label accuracy: "
        f"{accuracy * 100:.2f}%"
    )

    print(
        "=================================================="
    )


if __name__ == "__main__":

    main()