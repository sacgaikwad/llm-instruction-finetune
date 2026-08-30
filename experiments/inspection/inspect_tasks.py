import json
from evaluation.task_classifier import classify_instruction

TEST_DATA_PATH = (
    "data/v2/test_data_v2.json"
)


def load_test_data():

    with open(
        TEST_DATA_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


def main():

    data = load_test_data()

    counts = {}

    for example in data:

        task_type = classify_instruction(
            example["instruction"]
        )

        counts[task_type] = (
            counts.get(task_type, 0) + 1
        )

    print(
        "\n===== TASK DISTRIBUTION ====="
    )

    for task_type, count in counts.items():

        print(
            f"{task_type}: {count}"
        )

    print(
        "\n===== EXAMPLES ====="
    )

    for index, example in enumerate(
        data,
        start=1
    ):

        task_type = classify_instruction(
            example["instruction"]
        )

        print(
            f"{index:3} | "
            f"{task_type:22} | "
            f"{example['instruction']}"
        )


if __name__ == "__main__":

    main()