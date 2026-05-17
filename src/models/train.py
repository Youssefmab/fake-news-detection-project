"""Training pipeline for transformer-based fake news classifiers."""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup


class Trainer:
    """Handles the training, evaluation, and persistence of transformer models."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        learning_rate: float = 2e-5,
        num_epochs: int = 5,
        batch_size: int = 16,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.1,
        patience: int = 3,
        class_weights: Optional[List[float]] = None,
    ):
        """Initialize the trainer.

        Args:
            model: PyTorch model to train.
            device: Device to train on (cpu / cuda).
            learning_rate: Peak learning rate for AdamW.
            num_epochs: Maximum number of training epochs.
            batch_size: Default batch size for data loaders.
            weight_decay: L2 regularization coefficient.
            warmup_ratio: Fraction of total steps used for LR warmup.
            patience: Number of epochs without improvement before early stop.
            class_weights: Optional class weights for imbalanced data.
        """
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio
        self.patience = patience

        # Loss function (with optional class weights)
        if class_weights is not None:
            weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)
            self.criterion = nn.CrossEntropyLoss(weight=weights_tensor)
        else:
            self.criterion = nn.CrossEntropyLoss()

        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        # Scheduler and training history are set during training
        self.scheduler = None
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": [],
        }

    # ------------------------------------------------------------------
    # Data loaders
    # ------------------------------------------------------------------

    def create_data_loaders(
        self,
        train_dataset: TensorDataset,
        val_dataset: TensorDataset,
        batch_size: Optional[int] = None,
    ) -> Tuple[DataLoader, DataLoader]:
        """Create training and validation DataLoaders.

        Args:
            train_dataset: Training TensorDataset.
            val_dataset: Validation TensorDataset.
            batch_size: Override default batch size if provided.

        Returns:
            Tuple of (train_loader, val_loader).
        """
        bs = batch_size or self.batch_size
        train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=bs, shuffle=False)
        return train_loader, val_loader

    # ------------------------------------------------------------------
    # Single-epoch training
    # ------------------------------------------------------------------

    def train_epoch(self, train_loader: DataLoader) -> float:
        """Run one training epoch.

        Args:
            train_loader: DataLoader yielding (input_ids, attention_mask, labels).

        Returns:
            Average training loss for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(train_loader, desc="Training", leave=False)
        for batch in progress_bar:
            input_ids = batch[0].to(self.device)
            attention_mask = batch[1].to(self.device)
            labels = batch[2].to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
            loss = self.criterion(logits, labels)
            loss.backward()

            # Gradient clipping to avoid exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

            total_loss += loss.item()
            num_batches += 1
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        return total_loss / max(num_batches, 1)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self, val_loader: DataLoader
    ) -> Tuple[float, float, np.ndarray, np.ndarray]:
        """Evaluate the model on a validation set.

        Args:
            val_loader: DataLoader yielding (input_ids, attention_mask, labels).

        Returns:
            Tuple of (val_loss, val_accuracy, all_predictions, all_labels).
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        all_preds: List[np.ndarray] = []
        all_labels: List[np.ndarray] = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating", leave=False):
                input_ids = batch[0].to(self.device)
                attention_mask = batch[1].to(self.device)
                labels = batch[2].to(self.device)

                logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
                loss = self.criterion(logits, labels)

                total_loss += loss.item()
                num_batches += 1

                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_preds.append(preds)
                all_labels.append(labels.cpu().numpy())

        all_preds_arr = np.concatenate(all_preds)
        all_labels_arr = np.concatenate(all_labels)
        avg_loss = total_loss / max(num_batches, 1)
        accuracy = (all_preds_arr == all_labels_arr).mean()

        return avg_loss, accuracy, all_preds_arr, all_labels_arr

    # ------------------------------------------------------------------
    # Full training loop
    # ------------------------------------------------------------------

    def train(
        self, train_loader: DataLoader, val_loader: DataLoader
    ) -> Dict[str, List[float]]:
        """Full training loop with early stopping and linear warmup scheduling.

        Args:
            train_loader: Training DataLoader.
            val_loader: Validation DataLoader.

        Returns:
            Dictionary with training history (train_loss, val_loss, val_accuracy).
        """
        total_steps = len(train_loader) * self.num_epochs
        warmup_steps = int(total_steps * self.warmup_ratio)

        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        best_val_loss = float("inf")
        epochs_without_improvement = 0
        best_model_state = None

        for epoch in range(1, self.num_epochs + 1):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch}/{self.num_epochs}")
            print(f"{'='*60}")

            # Train
            train_loss = self.train_epoch(train_loader)
            self.history["train_loss"].append(train_loss)

            # Validate
            val_loss, val_accuracy, _, _ = self.evaluate(val_loader)
            self.history["val_loss"].append(val_loss)
            self.history["val_accuracy"].append(val_accuracy)

            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss:   {val_loss:.4f}")
            print(f"  Val Acc:    {val_accuracy:.4f}")

            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0
                best_model_state = {
                    k: v.clone() for k, v in self.model.state_dict().items()
                }
                print("  >> New best model saved.")
            else:
                epochs_without_improvement += 1
                print(
                    f"  >> No improvement for {epochs_without_improvement} epoch(s)."
                )

            if epochs_without_improvement >= self.patience:
                print(f"\nEarly stopping triggered after {epoch} epochs.")
                break

        # Restore best model weights
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print("Restored best model weights.")

        return self.history

    # ------------------------------------------------------------------
    # Model persistence
    # ------------------------------------------------------------------

    def save_model(self, path: str) -> None:
        """Save the model state dict to disk.

        Args:
            path: File path for the saved model.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

    def load_model(self, path: str) -> None:
        """Load a model state dict from disk.

        Args:
            path: File path of the saved model.
        """
        self.model.load_state_dict(
            torch.load(path, map_location=self.device)
        )
        self.model.to(self.device)
        print(f"Model loaded from {path}")
