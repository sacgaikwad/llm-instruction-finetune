import json
import random
from pathlib import Path


RANDOM_SEED = 42

TRAIN_SIZE = 935
VALIDATION_SIZE = 55
TEST_SIZE = 110


def load_json(file_path):

    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


def save_json(
    data,
    file_path
):

    file_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        file_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            data,
            file,
            indent=4,
            ensure_ascii=False
        )


def normalize_example(example):

    return {
        "instruction": example.get(
            "instruction",
            ""
        ).strip(),

        "input": example.get(
            "input",
            ""
        ).strip(),

        "output": example.get(
            "output",
            example.get(
                "response",
                ""
            )
        ).strip()
    }


def example_key(example):

    example = normalize_example(
        example
    )

    return (
        example["instruction"].lower(),
        example["input"].lower(),
        example["output"].lower()
    )


def remove_duplicates(examples):

    unique_examples = []
    seen = set()

    for example in examples:

        key = example_key(
            example
        )

        if key in seen:
            continue

        seen.add(key)

        unique_examples.append(
            normalize_example(
                example
            )
        )

    return unique_examples


def validate_example(example):

    if not isinstance(
        example,
        dict
    ):
        return False

    if not example.get(
        "instruction",
        ""
    ).strip():

        return False

    if not example.get(
        "output",
        ""
    ).strip():

        return False

    return True


def validate_examples(examples):

    valid = []
    invalid = []

    for example in examples:

        normalized = normalize_example(
            example
        )

        if validate_example(
            normalized
        ):

            valid.append(
                normalized
            )

        else:

            invalid.append(
                example
            )

    return valid, invalid


def split_original_dataset(data):

    train_portion = int(
        len(data) * 0.85
    )

    test_portion = int(
        len(data) * 0.10
    )

    train_data = data[
        :train_portion
    ]

    test_data = data[
        train_portion:
        train_portion + test_portion
    ]

    validation_data = data[
        train_portion + test_portion:
    ]

    return (
        train_data,
        validation_data,
        test_data
    )


def load_targeted_examples(
    targeted_data_paths
):

    targeted_examples = []

    for path in targeted_data_paths:

        data = load_json(
            path
        )

        print(
            f"Targeted examples from "
            f"{path}: {len(data)}"
        )

        targeted_examples.extend(
            data
        )

    return targeted_examples


def build_v2_training_dataset(
    original_data_path,
    targeted_data_paths,
    training_output_path,
    validation_output_path,
    test_output_path
):

    print(
        "\n===== BUILDING V2 DATASET ====="
    )

    # --------------------------------------------------------
    # Load original 1100 examples
    # --------------------------------------------------------

    original_data = load_json(
        original_data_path
    )

    print(
        "Original examples:",
        len(original_data)
    )

    # --------------------------------------------------------
    # Split exactly:
    #
    # 935 training
    # 55 validation
    # 110 test
    # --------------------------------------------------------

    (
        original_training,
        validation_data,
        test_data
    ) = split_original_dataset(
        original_data
    )

    print(
        "Original training:",
        len(original_training)
    )

    print(
        "Validation:",
        len(validation_data)
    )

    print(
        "Test:",
        len(test_data)
    )

    # --------------------------------------------------------
    # Load targeted V2 examples
    # --------------------------------------------------------

    targeted_examples = (
        load_targeted_examples(
            targeted_data_paths
        )
    )

    print(
        "Targeted examples:",
        len(targeted_examples)
    )

    # --------------------------------------------------------
    # Combine ONLY with training data
    # --------------------------------------------------------

    combined_training = (
        original_training
        +
        targeted_examples
    )

    print(
        "Combined training examples:",
        len(combined_training)
    )

    # --------------------------------------------------------
    # Remove duplicates
    # --------------------------------------------------------

    combined_training = (
        remove_duplicates(
            combined_training
        )
    )

    print(
        "After duplicate removal:",
        len(combined_training)
    )

    # --------------------------------------------------------
    # Validate training data
    # --------------------------------------------------------

    (
        training_data,
        invalid_examples
    ) = validate_examples(
        combined_training
    )

    print(
        "Valid training examples:",
        len(training_data)
    )

    print(
        "Invalid training examples:",
        len(invalid_examples)
    )

    if invalid_examples:

        print(
            "\n===== INVALID EXAMPLES ====="
        )

        for example in invalid_examples:

            print(example)

        raise ValueError(
            "Invalid training examples found."
        )

    # --------------------------------------------------------
    # Validate validation/test data
    # --------------------------------------------------------

    validation_data = [
        normalize_example(
            example
        )
        for example in validation_data
    ]

    test_data = [
        normalize_example(
            example
        )
        for example in test_data
    ]

    # --------------------------------------------------------
    # Shuffle ONLY training data
    # --------------------------------------------------------

    random.seed(
        RANDOM_SEED
    )

    random.shuffle(
        training_data
    )

    # --------------------------------------------------------
    # Save datasets
    # --------------------------------------------------------

    save_json(
        training_data,
        training_output_path
    )

    save_json(
        validation_data,
        validation_output_path
    )

    save_json(
        test_data,
        test_output_path
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print(
        "\n===== V2 DATASET CREATED ====="
    )

    print(
        "Training:",
        len(training_data)
    )

    print(
        "Validation:",
        len(validation_data)
    )

    print(
        "Test:",
        len(test_data)
    )

    print(
        "\nTraining output:"
    )

    print(
        training_output_path
    )

    print(
        "\nValidation output:"
    )

    print(
        validation_output_path
    )

    print(
        "\nTest output:"
    )

    print(
        test_output_path
    )


def main():

    project_root = (
        Path(__file__)
        .resolve()
        .parents[1]
    )

    original_data_path = (
        project_root
        / "data"
        / "raw"
        / "instruction-data.json"
    )

    targeted_data_paths = [

        project_root
        / "data"
        / "v2"
        / "knowledge_examples.json",

        project_root
        / "data"
        / "v2"
        / "language_examples.json",

        project_root
        / "data"
        / "v2"
        / "instruction_examples.json",

        project_root
        / "data"
        / "v2"
        / "transformation_examples.json",

        project_root
        / "data"
        / "v2"
        / "calculation_examples.json"
    ]

    training_output_path = (
        project_root
        / "data"
        / "v2"
        / "training_data_v2.json"
    )

    validation_output_path = (
        project_root
        / "data"
        / "v2"
        / "validation_data_v2.json"
    )

    test_output_path = (
        project_root
        / "data"
        / "v2"
        / "test_data_v2.json"
    )

    build_v2_training_dataset(
        original_data_path,
        targeted_data_paths,
        training_output_path,
        validation_output_path,
        test_output_path
    )


if __name__ == "__main__":
    main()