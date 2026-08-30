import json

with open(
    "data/v2/test_data_v2.json",
    "r",
    encoding="utf-8"
) as file:

    data = json.load(file)


for index, example in enumerate(data, start=1):

    print(f"{index}.")
    print("Instruction:", example["instruction"])
    print("Input:", example["input"])
    print("Expected:", example["output"])
    print("-" * 80)