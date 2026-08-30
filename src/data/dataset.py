import tiktoken
from torch.utils.data import Dataset


class InstructionDataset(Dataset):

    def __init__(
        self,
        data,
        tokenizer=None
    ):

        self.data = data

        if tokenizer is None:

            tokenizer = tiktoken.get_encoding(
                "gpt2"
            )

        self.tokenizer = tokenizer

        self.encoded_data = []

        for entry in data:

            instruction = entry[
                "instruction"
            ]

            input_text = entry.get(
                "input",
                ""
            )

            output = entry[
                "output"
            ]

            # ------------------------------------------------
            # Build instruction prompt
            # ------------------------------------------------

            if input_text:

                prompt = (
                    "Below is an instruction that describes "
                    "a task. Write a response that appropriately "
                    "completes the request.\n\n"
                    "### Instruction:\n"
                    f"{instruction}\n\n"
                    "## Input:\n"
                    f"{input_text}\n\n"
                    "### Response:\n"
                )

            else:

                prompt = (
                    "Below is an instruction that describes "
                    "a task. Write a response that appropriately "
                    "completes the request.\n\n"
                    "### Instruction:\n"
                    f"{instruction}\n\n"
                    "### Response:\n"
                )

            # ------------------------------------------------
            # Tokenize prompt and response separately
            # ------------------------------------------------

            prompt_tokens = tokenizer.encode(
                prompt
            )

            output_tokens = tokenizer.encode(
                output
            )

            # ------------------------------------------------
            # Store both parts
            # ------------------------------------------------

            self.encoded_data.append(
                {
                    "prompt_tokens": prompt_tokens,
                    "output_tokens": output_tokens
                }
            )

    def __getitem__(
        self,
        index
    ):

        return self.encoded_data[
            index
        ]

    def __len__(self):

        return len(
            self.data
        )