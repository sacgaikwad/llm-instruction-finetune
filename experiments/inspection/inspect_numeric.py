import json


with open(
    "data/v2/test_data_v2.json",
    "r",
    encoding="utf-8"
) as file:

    data = json.load(file)


keywords = [
    "convert",
    "calculate",
    "formula",
    "boiling point",
    "density",
    "roman numerals",
    "sequence",
]


for index, example in enumerate(
    data,
    start=1
):

    instruction = (
        example["instruction"]
        .lower()
    )

    if any(
        keyword in instruction
        for keyword in keywords
    ):

        print(f"\n{index}.")
        print(
            "Instruction:",
            example["instruction"]
        )
        print(
            "Input:",
            example["input"]
        )
        print(
            "Expected:",
            example["output"]
        )
        print("-" * 60)