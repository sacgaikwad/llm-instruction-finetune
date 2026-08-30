import torch


def custom_collate(
    batch,
    pad_token_id=50256,
    ignore_index=-100,
    allowed_max_length=None,
    device="cpu"
):

    # --------------------------------------------------------
    # Find maximum sequence length in this batch
    # --------------------------------------------------------

    batch_max_length = max(
        len(item["prompt_tokens"])
        + len(item["output_tokens"])
        + 1
        for item in batch
    )

    inputs_list = []
    targets_list = []

    for item in batch:

        prompt_tokens = item[
            "prompt_tokens"
        ]

        output_tokens = item[
            "output_tokens"
        ]

        # ----------------------------------------------------
        # Full sequence
        #
        # prompt + response
        # ----------------------------------------------------

        full_tokens = (
            prompt_tokens
            +
            output_tokens
        )

        # ----------------------------------------------------
        # Add one padding token so that
        # input and target can be shifted
        # ----------------------------------------------------

        full_tokens = (
            full_tokens
            +
            [pad_token_id]
        )

        # ----------------------------------------------------
        # Padding
        # ----------------------------------------------------

        padded = (
            full_tokens
            +
            [
                pad_token_id
            ]
            * (
                batch_max_length
                - len(full_tokens)
            )
        )

        # ----------------------------------------------------
        # Shift by one token
        # ----------------------------------------------------

        inputs = torch.tensor(
            padded[:-1]
        )

        targets = torch.tensor(
            padded[1:]
        )

        # ----------------------------------------------------
        # RESPONSE-ONLY LOSS MASKING
        #
        # Prompt tokens should NOT contribute
        # to the training loss.
        # ----------------------------------------------------

        prompt_length = len(
            prompt_tokens
        )

        targets[
            :prompt_length
        ] = ignore_index

        # ----------------------------------------------------
        # Padding tokens should also NOT contribute
        # to the training loss.
        # ----------------------------------------------------

        targets[
            targets == pad_token_id
        ] = ignore_index

        # ----------------------------------------------------
        # Maximum sequence length
        # ----------------------------------------------------

        if allowed_max_length is not None:

            inputs = inputs[
                :allowed_max_length
            ]

            targets = targets[
                :allowed_max_length
            ]

        inputs_list.append(
            inputs
        )

        targets_list.append(
            targets
        )

    # --------------------------------------------------------
    # Stack batches
    # --------------------------------------------------------

    inputs_tensor = torch.stack(
        inputs_list
    ).to(device)

    target_tensor = torch.stack(
        targets_list
    ).to(device)

    return (
        inputs_tensor,
        target_tensor
    )