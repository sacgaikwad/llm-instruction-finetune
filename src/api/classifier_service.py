from pathlib import Path
import math

import torch
import tiktoken

from src.model.gpt_model import Model


class ClassifierService:

    def __init__(self):

        self.project_root = (
            Path(__file__).resolve().parent.parent.parent
        )

        self.model_path = (
            self.project_root
            / "models"
            / "controlled"
            / "classification_controlled_v2_best.pth"
        )

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.config = {
            "vocab_size": 50257,
            "context_length": 1024,
            "emb_dim": 1024,
            "n_layers": 24,
            "n_heads": 16,
            "drop_rate": 0.0,
            "qkv_bias": True,
        }

        self.tokenizer = tiktoken.get_encoding("gpt2")

        self.labels = {
            "SCAM": self.tokenizer.encode(" SCAM"),
            "LEGIT": self.tokenizer.encode(" LEGIT"),
        }

        self.model = self._load_model()

    # ========================================================
    # LOAD MODEL
    # ========================================================

    def _load_model(self):

        print("Loading classification model...")

        if not self.model_path.exists():

            raise FileNotFoundError(
                f"Model checkpoint not found:\n"
                f"{self.model_path}"
            )

        model = Model(self.config)

        checkpoint = torch.load(
            self.model_path,
            map_location=self.device
        )

        model.load_state_dict(
            checkpoint,
            strict=True
        )

        model.to(self.device)

        model.eval()

        print(
            f"Model loaded successfully on "
            f"{self.device}"
        )

        return model

    # ========================================================
    # PROMPT
    # ========================================================

    def _format_prompt(self, message):

        return (
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request."
            "\n\n"
            "### Instruction:\n"
            "Classify the following message as SCAM or LEGIT."
            "\n\n"
            "## Input:\n"
            f"{message}"
            "\n\n"
            "### Response:\n"
        )

    # ========================================================
    # LABEL LOG PROBABILITY
    # ========================================================

    def _calculate_label_log_probability(
        self,
        prompt,
        label_tokens
    ):

        prompt_tokens = self.tokenizer.encode(prompt)

        input_tokens = (
            prompt_tokens
            + label_tokens
        )

        input_tensor = torch.tensor(
            input_tokens,
            dtype=torch.long,
            device=self.device
        ).unsqueeze(0)

        with torch.no_grad():

            logits = self.model(input_tensor)

        prompt_length = len(prompt_tokens)

        total_log_probability = 0.0

        for index, token_id in enumerate(label_tokens):

            prediction_position = (
                prompt_length
                + index
                - 1
            )

            token_logits = logits[
                0,
                prediction_position,
                :
            ]

            log_probabilities = torch.log_softmax(
                token_logits,
                dim=-1
            )

            total_log_probability += (
                log_probabilities[token_id].item()
            )

        return total_log_probability

    # ========================================================
    # CLASSIFY
    # ========================================================

    def classify(self, message):

        prompt = self._format_prompt(message)

        scam_log_probability = (
            self._calculate_label_log_probability(
                prompt,
                self.labels["SCAM"]
            )
        )

        legit_log_probability = (
            self._calculate_label_log_probability(
                prompt,
                self.labels["LEGIT"]
            )
        )

        difference = (
            scam_log_probability
            - legit_log_probability
        )

        # ----------------------------------------------------
        # Convert log-probability difference to probability
        # ----------------------------------------------------

        scam_probability = (
            1.0
            / (
                1.0
                + math.exp(-difference)
            )
        )

        legit_probability = (
            1.0 - scam_probability
        )

        # ----------------------------------------------------
        # Decision policy
        # ----------------------------------------------------

        if scam_probability >= 0.90:

            prediction = "SCAM"

        elif legit_probability >= 0.90:

            prediction = "LEGIT"

        else:

            prediction = "UNCERTAIN"

        # ----------------------------------------------------
        # Confidence
        # ----------------------------------------------------

        confidence = max(
            scam_probability,
            legit_probability
        )

        if confidence >= 0.90:
            confidence_level = "HIGH"
        elif confidence >= 0.75:

            confidence_level = "MEDIUM"
        else:
            confidence_level = "LOW"
        return {
            "prediction": prediction,
            "scam_probability": scam_probability,
            "legit_probability": legit_probability,
            "confidence": confidence,
            "confidence_level": confidence_level,
        }