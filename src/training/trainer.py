import torch
import torch.nn.functional as F


class Trainer:

    def __init__(
        self,
        model,
        optimizer,
        train_loader,
        val_loader,
        device,
        config
    ):

        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config

        self.train_losses = []
        self.val_losses = []

        self.examples_seen = 0
        self.global_step = -1

        self.train_accs = []
        self.val_accs = []

        self.best_val_loss = float("inf")
        self.best_step = None
        self.best_model_state = None

    # ========================================================
    # LOSS
    # ========================================================

    def calc_loss_batch(
        self,
        input_batch,
        target_batch
    ):

        input_batch = input_batch.to(
            self.device
        )

        target_batch = target_batch.to(
            self.device
        )

        logits = self.model(
            input_batch
        )

        logits = logits.flatten(
            0,
            1
        )

        target_batch = target_batch.flatten()

        loss = F.cross_entropy(
            logits,
            target_batch,
            ignore_index=-100
        )

        return loss

    # ========================================================
    # LOSS LOADER
    # ========================================================

    def calc_loss_loader(
        self,
        data_loader,
        num_batches=None
    ):

        total_loss = 0.0

        if len(data_loader) == 0:

            return float("nan")

        if num_batches is None:

            num_batches = len(
                data_loader
            )

        else:

            num_batches = min(
                num_batches,
                len(data_loader)
            )

        for i, (
            input_batch,
            target_batch
        ) in enumerate(
            data_loader
        ):

            if i >= num_batches:

                break

            loss = self.calc_loss_batch(
                input_batch,
                target_batch
            )

            total_loss += loss.item()

        return (
            total_loss
            / num_batches
        )

    # ========================================================
    # EVALUATION
    # ========================================================

    def evaluate(self):

        self.model.eval()

        with torch.no_grad():

            train_loss = (
                self.calc_loss_loader(
                    self.train_loader,
                    num_batches=(
                        self.config.eval_iter
                    )
                )
            )

            val_loss = (
                self.calc_loss_loader(
                    self.val_loader,
                    num_batches=(
                        self.config.eval_iter
                    )
                )
            )

        self.model.train()

        return (
            train_loss,
            val_loss
        )

    # ========================================================
    # TRAINING
    # ========================================================
    def train(self):
        print(
            "\n===== V2 TRAINING ====="
        )

        for epoch in range(
            self.config.num_epochs
        ):

            self.model.train()

            for input_batch, target_batch in (
                self.train_loader
            ):

                self.optimizer.zero_grad()

                loss = self.calc_loss_batch(
                    input_batch,
                    target_batch
                )

                loss.backward()

                self.optimizer.step()

                self.examples_seen += (
                    input_batch.shape[0]
                )

                self.global_step += 1

                if (
                    self.global_step
                    % self.config.eval_freq
                    == 0
                ):

                    (
                        train_loss,
                        val_loss
                    ) = self.evaluate()


                    if val_loss < self.best_val_loss:

                        self.best_val_loss = val_loss
                        self.best_step = self.global_step

                        self.best_model_state = {
                            key: value.detach().cpu().clone()
                            for key, value
                            in self.model.state_dict().items()
                        }

                        print(
                            f"⭐ New best model: "
                            f"step={self.best_step}, "
                            f"val_loss={self.best_val_loss:.4f}"
                        )

                    self.train_losses.append(
                        train_loss
                    )

                    self.val_losses.append(
                        val_loss
                    )

                    print(
                        f"Ep {epoch + 1} "
                        f"(Step "
                        f"{self.global_step:06d}): "
                        f"Train loss "
                        f"{train_loss:.3f}, "
                        f"Val loss "
                        f"{val_loss:.3f}"
                    )

            # ====================================================
            # END-OF-EPOCH ACCURACY
            # ====================================================

            train_accuracy = (
                self.calc_accuracy_loader(
                    self.train_loader,
                    num_batches=self.config.eval_iter
                )
            )

            val_accuracy = (
                self.calc_accuracy_loader(
                    self.val_loader,
                    num_batches=self.config.eval_iter
                )
            )

            print(
                f"Training token accuracy: "
                f"{train_accuracy * 100:.2f}% | "
                f"Validation token accuracy: "
                f"{val_accuracy * 100:.2f}%"
            )

            self.train_accs.append(
                train_accuracy
            )

            self.val_accs.append(
                val_accuracy
            )

        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "train_accs": self.train_accs,
            "val_accs": self.val_accs,
            "examples_seen": self.examples_seen,
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
            "best_step": self.best_step
        }


    def calc_accuracy_loader(
    self,
    data_loader,
    num_batches=None):

        self.model.eval()

        correct_predictions = 0
        num_tokens = 0

        if num_batches is None:

            num_batches = len(
                data_loader
            )

        else:

            num_batches = min(
                num_batches,
                len(data_loader)
            )

        with torch.no_grad():

            for i, (
                input_batch,
                target_batch
            ) in enumerate(
                data_loader
            ):

                if i >= num_batches:
                    break

                input_batch = input_batch.to(
                    self.device
                )

                target_batch = target_batch.to(
                    self.device
                )

                logits = self.model(
                    input_batch
                )

                predicted_tokens = torch.argmax(
                    logits,
                    dim=-1
                )

                mask = (
                    target_batch != -100
                )

                correct_predictions += (
                    (
                        predicted_tokens == target_batch
                    )
                    .masked_select(mask)
                    .sum()
                    .item()
                )

                num_tokens += (
                    mask.sum().item()
                )

        self.model.train()

        if num_tokens == 0:
            return 0.0

        return (
            correct_predictions
            / num_tokens
        )