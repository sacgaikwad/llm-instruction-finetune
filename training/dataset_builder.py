import json
import random
from pathlib import Path


RANDOM_SEED = 42

ORIGINAL_TRAIN_SIZE = 935
VALIDATION_SIZE = 55
TEST_SIZE = 110


# ============================================================
# JSON HELPERS
# ============================================================

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


# ============================================================
# NORMALIZATION
# ============================================================

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


# ============================================================
# VALIDATION
# ============================================================

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


# ============================================================
# DUPLICATE REMOVAL
# ============================================================

def remove_duplicates(examples):

    unique_examples = []
    seen = set()

    for example in examples:

        normalized = normalize_example(
            example
        )

        key = example_key(
            normalized
        )

        if key in seen:
            continue

        seen.add(key)

        unique_examples.append(
            normalized
        )

    return unique_examples


# ============================================================
# OVERLAP FILTER
# ============================================================

def remove_overlaps(
    examples,
    excluded_examples
):

    excluded_keys = {
        example_key(example)
        for example in excluded_examples
    }

    filtered_examples = []
    removed_examples = []

    for example in examples:

        normalized = normalize_example(
            example
        )

        key = example_key(
            normalized
        )

        if key in excluded_keys:

            removed_examples.append(
                normalized
            )

        else:

            filtered_examples.append(
                normalized
            )

    return (
        filtered_examples,
        removed_examples
    )


# ============================================================
# OVERLAP COUNT
# ============================================================

def count_overlap(
    examples_a,
    examples_b
):

    keys_a = {
        example_key(example)
        for example in examples_a
    }

    keys_b = {
        example_key(example)
        for example in examples_b
    }

    return len(
        keys_a.intersection(
            keys_b
        )
    )


# ============================================================
# ORIGINAL DATASET SPLIT
# ============================================================

def split_original_dataset(data):

    expected_size = (
        ORIGINAL_TRAIN_SIZE
        + VALIDATION_SIZE
        + TEST_SIZE
    )

    if len(data) != expected_size:

        raise ValueError(
            "Unexpected original dataset size: "
            f"{len(data)}. "
            f"Expected {expected_size}."
        )

    train_data = data[
        :ORIGINAL_TRAIN_SIZE
    ]

    validation_data = data[
        ORIGINAL_TRAIN_SIZE:
        ORIGINAL_TRAIN_SIZE
        + VALIDATION_SIZE
    ]

    test_data = data[
        ORIGINAL_TRAIN_SIZE
        + VALIDATION_SIZE:
    ]

    return (
        train_data,
        validation_data,
        test_data
    )


# ============================================================
# TARGETED DATA
# ============================================================

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


# ============================================================
# CLASSIFICATION VALIDATION DATA
# ============================================================

def load_classification_validation_examples(
    file_path
):

    data = load_json(
        file_path
    )

    # Only SCAM and LEGIT are used for the
    # binary classification validation set.
    #
    # AMBIGUOUS examples are intentionally excluded
    # from this validation set.

    classification_validation = [
        example
        for example in data
        if example.get(
            "output",
            ""
        ).strip()
        in {
            "SCAM",
            "LEGIT"
        }
    ]

    return classification_validation


# ============================================================
# DATASET SUMMARY
# ============================================================

def print_dataset_summary(
    name,
    examples
):

    print(
        f"{name}: {len(examples)}"
    )


# ============================================================
# BUILD V2 DATASET
# ============================================================

def build_v2_training_dataset(
    original_data_path,
    targeted_data_paths,
    classification_validation_path,
    training_output_path,
    validation_output_path,
    classification_validation_output_path,
    test_output_path
):

    print(
        "\n===== BUILDING V2 DATASET ====="
    )

    # --------------------------------------------------------
    # LOAD ORIGINAL DATASET
    # --------------------------------------------------------

    original_data = load_json(
        original_data_path
    )

    print_dataset_summary(
        "Original examples",
        original_data
    )

    # --------------------------------------------------------
    # SPLIT ORIGINAL DATASET
    # --------------------------------------------------------

    (
        original_training,
        validation_data,
        test_data
    ) = split_original_dataset(
        original_data
    )

    print_dataset_summary(
        "Original training",
        original_training
    )

    print_dataset_summary(
        "Validation",
        validation_data
    )

    print_dataset_summary(
        "Test",
        test_data
    )

    # --------------------------------------------------------
    # NORMALIZE ORIGINAL DATA
    # --------------------------------------------------------

    original_training = [
        normalize_example(example)
        for example in original_training
    ]

    validation_data = [
        normalize_example(example)
        for example in validation_data
    ]

    test_data = [
        normalize_example(example)
        for example in test_data
    ]

    # --------------------------------------------------------
    # LOAD CLASSIFICATION VALIDATION
    # --------------------------------------------------------

    classification_validation_data = (
        load_classification_validation_examples(
            classification_validation_path
        )
    )

    classification_validation_data = [
        normalize_example(example)
        for example in classification_validation_data
    ]

    print_dataset_summary(
        "Classification validation",
        classification_validation_data
    )

    # --------------------------------------------------------
    # PRE-TRAINING HOLDOUT CHECK
    # --------------------------------------------------------

    overlap_with_validation = count_overlap(
        classification_validation_data,
        validation_data
    )

    overlap_with_test = count_overlap(
        classification_validation_data,
        test_data
    )

    print(
        "\n===== PRE-TRAINING OVERLAP CHECK ====="
    )

    print(
        "Classification validation <-> "
        f"general validation: "
        f"{overlap_with_validation}"
    )

    print(
        "Classification validation <-> "
        f"test: "
        f"{overlap_with_test}"
    )

    if overlap_with_validation:

        raise ValueError(
            "Classification validation overlaps "
            "with general validation."
        )

    if overlap_with_test:

        raise ValueError(
            "Classification validation overlaps "
            "with test data."
        )

    # --------------------------------------------------------
    # LOAD TARGETED EXAMPLES
    # --------------------------------------------------------

    targeted_examples = (
        load_targeted_examples(
            targeted_data_paths
        )
    )

    print_dataset_summary(
        "Targeted examples",
        targeted_examples
    )

    # --------------------------------------------------------
    # PROTECTED HOLDOUTS
    #
    # These datasets must NEVER enter training:
    #
    # 1. General validation
    # 2. General test
    # 3. Classification validation
    # --------------------------------------------------------

    protected_examples = (
        validation_data
        +
        test_data
        +
        classification_validation_data
    )

    # --------------------------------------------------------
    # REMOVE HOLDOUTS FROM TARGETED DATA
    # --------------------------------------------------------

    (
        targeted_examples,
        removed_targeted_holdouts
    ) = remove_overlaps(
        targeted_examples,
        protected_examples
    )

    print(
        "\n===== HOLDOUT PROTECTION ====="
    )

    print(
        "Removed targeted examples overlapping "
        "with holdout datasets: "
        f"{len(removed_targeted_holdouts)}"
    )

    # --------------------------------------------------------
    # REMOVE HOLDOUTS FROM ORIGINAL TRAINING
    # --------------------------------------------------------

    (
        original_training,
        removed_original_holdouts
    ) = remove_overlaps(
        original_training,
        protected_examples
    )

    print(
        "Removed original training examples "
        "overlapping with holdout datasets: "
        f"{len(removed_original_holdouts)}"
    )

    # --------------------------------------------------------
    # COMBINE TRAINING DATA
    # --------------------------------------------------------

    combined_training = (
        original_training
        +
        targeted_examples
    )

    print_dataset_summary(
        "Combined training before cleanup",
        combined_training
    )

    # --------------------------------------------------------
    # REMOVE DUPLICATES
    # --------------------------------------------------------

    combined_training = (
        remove_duplicates(
            combined_training
        )
    )

    print_dataset_summary(
        "Training after duplicate removal",
        combined_training
    )

    # --------------------------------------------------------
    # VALIDATE TRAINING DATA
    # --------------------------------------------------------

    (
        training_data,
        invalid_examples
    ) = validate_examples(
        combined_training
    )

    print_dataset_summary(
        "Valid training examples",
        training_data
    )

    print_dataset_summary(
        "Invalid training examples",
        invalid_examples
    )

    if invalid_examples:

        print(
            "\n===== INVALID TRAINING EXAMPLES ====="
        )

        for example in invalid_examples:

            print(example)

        raise ValueError(
            "Invalid training examples found."
        )

    # --------------------------------------------------------
    # VALIDATE GENERAL VALIDATION DATA
    # --------------------------------------------------------

    (
        validation_data,
        invalid_validation
    ) = validate_examples(
        validation_data
    )

    if invalid_validation:

        raise ValueError(
            "Invalid validation examples found."
        )

    # --------------------------------------------------------
    # VALIDATE TEST DATA
    # --------------------------------------------------------

    (
        test_data,
        invalid_test
    ) = validate_examples(
        test_data
    )

    if invalid_test:

        raise ValueError(
            "Invalid test examples found."
        )

    # --------------------------------------------------------
    # VALIDATE CLASSIFICATION VALIDATION
    # --------------------------------------------------------

    (
        classification_validation_data,
        invalid_classification_validation
    ) = validate_examples(
        classification_validation_data
    )

    if invalid_classification_validation:

        raise ValueError(
            "Invalid classification validation "
            "examples found."
        )

    # --------------------------------------------------------
    # FINAL LEAKAGE CHECK
    # --------------------------------------------------------

    overlap_training_validation = count_overlap(
        training_data,
        validation_data
    )

    overlap_training_test = count_overlap(
        training_data,
        test_data
    )

    overlap_training_classification = count_overlap(
        training_data,
        classification_validation_data
    )

    validation_test_overlap = count_overlap(
        validation_data,
        test_data
    )

    validation_classification_overlap = count_overlap(
        validation_data,
        classification_validation_data
    )

    test_classification_overlap = count_overlap(
        test_data,
        classification_validation_data
    )

    print(
        "\n===== FINAL DATASET LEAKAGE CHECK ====="
    )

    print(
        "Training <-> Validation: "
        f"{overlap_training_validation}"
    )

    print(
        "Training <-> Test: "
        f"{overlap_training_test}"
    )

    print(
        "Training <-> Classification validation: "
        f"{overlap_training_classification}"
    )

    print(
        "Validation <-> Test: "
        f"{validation_test_overlap}"
    )

    print(
        "Validation <-> Classification validation: "
        f"{validation_classification_overlap}"
    )

    print(
        "Test <-> Classification validation: "
        f"{test_classification_overlap}"
    )

    # --------------------------------------------------------
    # FAIL IF ANY LEAKAGE REMAINS
    # --------------------------------------------------------

    if overlap_training_validation:

        raise ValueError(
            "Training/validation leakage detected."
        )

    if overlap_training_test:

        raise ValueError(
            "Training/test leakage detected."
        )

    if overlap_training_classification:

        raise ValueError(
            "Training/classification validation "
            "leakage detected."
        )

    if validation_test_overlap:

        raise ValueError(
            "Validation/test overlap detected."
        )

    if validation_classification_overlap:

        raise ValueError(
            "Validation/classification validation "
            "overlap detected."
        )

    if test_classification_overlap:

        raise ValueError(
            "Test/classification validation "
            "overlap detected."
        )

    print(
        "STATUS: Dataset leakage check PASSED."
    )

    # --------------------------------------------------------
    # SHUFFLE TRAINING DATA
    # --------------------------------------------------------

    random.seed(
        RANDOM_SEED
    )

    random.shuffle(
        training_data
    )

    # --------------------------------------------------------
    # SAVE DATASETS
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

    save_json(
        classification_validation_data,
        classification_validation_output_path
    )

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print(
        "\n===== V2 DATASET CREATED ====="
    )

    print_dataset_summary(
        "Training",
        training_data
    )

    print_dataset_summary(
        "Validation",
        validation_data
    )

    print_dataset_summary(
        "Test",
        test_data
    )

    print_dataset_summary(
        "Classification validation",
        classification_validation_data
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

    print(
        "\nClassification validation output:"
    )

    print(
        classification_validation_output_path
    )

    print(
        "\n===== DATASET BUILD COMPLETE ====="
    )


# ============================================================
# MAIN
# ============================================================

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
        / "classification_control_examples.json",

        project_root
        / "data"
        / "v2"
        / "transformation_examples.json",

        project_root
        / "data"
        / "v2"
        / "calculation_examples.json"
    ]

    classification_validation_path = (
        project_root
        / "data"
        / "v2"
        / "classification_validation_examples.json"
    )

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

    classification_validation_output_path = (
        project_root
        / "data"
        / "v2"
        / "classification_validation_examples_v2.json"
    )

    build_v2_training_dataset(
        original_data_path,
        targeted_data_paths,
        classification_validation_path,
        training_output_path,
        validation_output_path,
        classification_validation_output_path,
        test_output_path
    )


if __name__ == "__main__":

    main()