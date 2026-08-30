import os
import json
from pathlib import Path

import numpy as np
import requests
import tensorflow as tf
from tqdm import tqdm


ALLOWED_MODEL_SIZES = (
    "124M",
    "355M",
    "774M",
    "1558M"
)


def download_and_load_gpt2(
    model_size,
    models_dir
):

    if model_size not in ALLOWED_MODEL_SIZES:

        raise ValueError(
            f"Model size not in "
            f"{ALLOWED_MODEL_SIZES}"
        )

    model_dir = os.path.join(
        models_dir,
        model_size
    )

    base_url = (
        "https://openaipublic.blob.core.windows.net/"
        "gpt-2/models"
    )

    filenames = [
        "checkpoint",
        "encoder.json",
        "hparams.json",
        "model.ckpt.data-00000-of-00001",
        "model.ckpt.index",
        "model.ckpt.meta",
        "vocab.bpe"
    ]

    os.makedirs(
        model_dir,
        exist_ok=True
    )

    for filename in filenames:

        file_url = os.path.join(
            base_url,
            model_size,
            filename
        )

        file_path = os.path.join(
            model_dir,
            filename
        )

        download_file(
            file_url,
            file_path
        )

    # --------------------------------------------------------
    # Load checkpoint
    # --------------------------------------------------------

    tf_ckpt_path = (
        tf.train.latest_checkpoint(
            model_dir
        )
    )

    settings_path = os.path.join(
        model_dir,
        "hparams.json"
    )

    with open(
        settings_path,
        "r",
        encoding="utf-8"
    ) as file:

        settings = json.load(
            file
        )

    params = (
        load_gpt2_params_from_tf_ckpt(
            tf_ckpt_path,
            settings
        )
    )

    return settings, params


def download_file(
    url,
    destination
):

    try:

        response = requests.get(
            url,
            stream=True,
            verify=True
        )

        response.raise_for_status()

        file_size = int(
            response.headers.get(
                "content-length",
                0
            )
        )

        if os.path.exists(
            destination
        ):

            file_size_local = (
                os.path.getsize(
                    destination
                )
            )

            if (
                file_size > 0
                and file_size == file_size_local
            ):

                return

        block_size = 1024

        progress_bar_description = (
            url.split("/")[-1]
        )

        with tqdm(
            total=file_size,
            unit="iB",
            unit_scale=True,
            desc=progress_bar_description
        ) as progress_bar:

            with open(
                destination,
                "wb"
            ) as file:

                for chunk in response.iter_content(
                    block_size
                ):

                    if not chunk:
                        continue

                    progress_bar.update(
                        len(chunk)
                    )

                    file.write(
                        chunk
                    )

    except requests.exceptions.RequestException as e:

        raise RuntimeError(
            f"Error downloading GPT-2 file: {url}"
        ) from e


def load_gpt2_params_from_tf_ckpt(
    ckpt_path,
    settings
):

    params = {
        "blocks": [
            {}
            for _ in range(
                settings["n_layer"]
            )
        ]
    }

    for name, _ in tf.train.list_variables(
        ckpt_path
    ):

        variable_array = np.squeeze(
            tf.train.load_variable(
                ckpt_path,
                name
            )
        )

        variable_name_parts = (
            name.split("/")[1:]
        )

        target_dict = params

        if variable_name_parts[
            0
        ].startswith("h"):

            layer_number = int(
                variable_name_parts[
                    0
                ][1:]
            )

            target_dict = (
                params["blocks"][
                    layer_number
                ]
            )

        for key in variable_name_parts[
            1:-1
        ]:

            target_dict = (
                target_dict.setdefault(
                    key,
                    {}
                )
            )

        last_key = (
            variable_name_parts[-1]
        )

        target_dict[
            last_key
        ] = variable_array

    return params