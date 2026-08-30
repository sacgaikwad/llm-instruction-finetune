import json
from pathlib import Path

import torch
import tiktoken
from torch.utils.data import DataLoader

from src.model.config import GPT_CONFIG_355M
from src.model.gpt_model import Model
from src.model.weights import load_weights_into_model

from src.data.dataset import InstructionDataset
from src.data.collate import custom_collate

from src.training.config import TrainingConfig
from src.training.trainer import Trainer

from src.training.pretrained_loader import download_and_load_gpt2


def load_json(file_path):

    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


def main():

    # ========================================================
    # PROJECT ROOT
    # ========================================================

    project_root = (
        Path(__file__)
        .resolve()
        .parents[1]
    )

    # ========================================================
    # CONFIGURATION
    # ========================================================

    config = TrainingConfig()

    torch.manual_seed(
        config.random_seed
    )

    # ========================================================
    # DEVICE
    # ========================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Device:",
        device
    )

    # ========================================================
    # PATHS
    # ========================================================

    training_path = (
        project_root
        / "data"
        / "v2"
        / "training_data_v2.json"
    )

    validation_path = (
        project_root
        / "data"
        / "v2"
        / "validation_data_v2.json"
    )

    models_dir = (
        project_root
        / "models"
        / "pretrained"
    )

    output_dir = (
        project_root
        / "models"
        / "v2"
    )

    model_output_path = (
        output_dir
        / "instruction_finetuned_v2.pth"
    )

    history_output_path = (
        output_dir
        / "training_history_v2.json"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # ========================================================
    # LOAD DATA
    # ========================================================

    train_data = load_json(
        training_path
    )

    validation_data = load_json(
        validation_path
    )

    print(
        "Training examples:",
        len(train_data)
    )

    print(
        "Validation examples:",
        len(validation_data)
    )

    # ========================================================
    # TOKENIZER
    # ========================================================

    tokenizer = tiktoken.get_encoding(
        "gpt2"
    )

    # ========================================================
    # DATASETS
    # ========================================================

    train_dataset = InstructionDataset(
        train_data,
        tokenizer
    )

    validation_dataset = InstructionDataset(
        validation_data,
        tokenizer
    )

    # ========================================================
    # COLLATE FUNCTION
    # ========================================================

    collate_function = lambda batch: (
        custom_collate(
            batch,
            allowed_max_length=(
                config.allowed_max_length
            ),
            device=device
        )
    )

    # ========================================================
    # DATA LOADERS
    # ========================================================

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=config.num_workers,
        collate_fn=collate_function
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=config.num_workers,
        collate_fn=collate_function
    )

    print(
        "Training batches:",
        len(train_loader)
    )

    print(
        "Validation batches:",
        len(validation_loader)
    )

    # ========================================================
    # DOWNLOAD / LOAD PRETRAINED GPT-2
    # ========================================================

    print(
        "\n===== LOADING PRETRAINED GPT-2 ====="
    )

    settings, params = (
        download_and_load_gpt2(
            model_size="355M",
            models_dir=str(
                models_dir
            )
        )
    )

    print(
        "Pretrained GPT-2 checkpoint loaded."
    )

    # ========================================================
    # MODEL
    # ========================================================

    model = Model(
        GPT_CONFIG_355M
    )

    print(
        "Model parameters:",
        sum(
            p.numel()
            for p in model.parameters()
        )
    )

    # ========================================================
    # LOAD PRETRAINED WEIGHTS
    # ========================================================

    load_weights_into_model(
        model,
        params
    )

    print(
        "Pretrained weights assigned."
    )

    model.to(
        device
    )

    # ========================================================
    # OPTIMIZER
    # ========================================================

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    # ========================================================
    # TRAINER
    # ========================================================

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=validation_loader,
        device=device,
        config=config
    )

    # ========================================================
    # TRAIN
    # ========================================================

    history = trainer.train()


    best_model_path = (
    output_dir
    / "instruction_finetuned_v2_best.pth"
    )

    if trainer.best_model_state is not None:

        torch.save(
            trainer.best_model_state,
            best_model_path
        )

        print(
            "\nBest V2 model saved to:"
        )

        print(
            best_model_path
        )

    # ========================================================
    # SAVE MODEL
    # ========================================================

    torch.save(
        model.state_dict(),
        model_output_path
    )

    print(
        "\nV2 model saved to:"
    )

    print(
        model_output_path
    )

    # ========================================================
    # SAVE TRAINING HISTORY
    # ========================================================

    with open(
        history_output_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            history,
            file,
            indent=4
        )

    print(
        "Training history saved to:"
    )

    print(
        history_output_path
    )


if __name__ == "__main__":

    main()