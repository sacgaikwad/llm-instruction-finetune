"""
Interactive GPT-style SCAM / LEGIT classifier.

This is the application layer for the trained V2 classifier.
The actual model loading and classification logic remain in:

    evaluation/classification_inference_v2.py

Run from the project root:

    python -m evaluation.classification_interactive_v2
"""

from evaluation.classification_inference_v2 import (
    load_model,
    classify_message,
    print_result,
)


# ============================================================
# INTERACTIVE CLASSIFIER
# ============================================================

def interactive_mode(model):

    print("\n========================================")
    print(" GPT-STYLE SCAM CLASSIFIER V2")
    print("========================================")

    print("Enter a message to classify.")
    print("Type 'exit' to stop.")

    while True:

        print()

        message = input("Message: ").strip()

        if message.lower() == "exit":
            print("\nExiting classifier.")
            break

        if not message:
            print("Please enter a message.")
            continue

        try:

            result = classify_message(
                model,
                message
            )

            print_result(
                message,
                result
            )

        except ValueError as error:

            print("\nERROR")
            print(error)

        except RuntimeError as error:

            print("\nMODEL ERROR")
            print(error)


# ============================================================
# MAIN
# ============================================================

def main():

    model = load_model()

    interactive_mode(model)

    print("\n========================================")
    print(" CLASSIFIER SESSION COMPLETE")
    print("========================================")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
