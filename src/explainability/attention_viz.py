"""Attention weight extraction and visualization for transformer models."""

from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from transformers import PreTrainedTokenizer


class AttentionVisualizer:
    """Extracts and visualizes attention weights from BERT-family models."""

    def extract_attention(
        self,
        model: nn.Module,
        tokenizer: PreTrainedTokenizer,
        text: str,
        device: torch.device,
    ) -> Tuple[torch.Tensor, List[str]]:
        """Extract attention weights from the model for a single input text.

        Args:
            model: A BERT/RoBERTa classifier with a `.bert` or `.roberta` attribute.
            tokenizer: HuggingFace tokenizer matching the model.
            text: Input text string.
            device: Torch device.

        Returns:
            Tuple of:
                - attention_weights: Tensor of shape (num_layers, num_heads, seq_len, seq_len)
                - tokens: List of token strings corresponding to the input.
        """
        model.eval()
        encoding = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=256,
            padding="max_length",
        )
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        # Determine the underlying transformer backbone
        backbone = getattr(model, "bert", None) or getattr(model, "roberta", None)
        if backbone is None:
            raise AttributeError(
                "Model must have a 'bert' or 'roberta' attribute to extract attention."
            )

        with torch.no_grad():
            outputs = backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True,
            )

        # outputs.attentions is a tuple of tensors, one per layer
        # Each tensor has shape (batch, num_heads, seq_len, seq_len)
        attention_weights = torch.stack(outputs.attentions).squeeze(1)  # (layers, heads, seq, seq)

        # Decode tokens (trim padding)
        seq_len = int(attention_mask.sum().item())
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0][:seq_len])
        attention_weights = attention_weights[:, :, :seq_len, :seq_len]

        return attention_weights.cpu(), tokens

    def plot_attention_heatmap(
        self,
        attention_weights: torch.Tensor,
        tokens: List[str],
        layer: int = -1,
        head: int = 0,
        save_path: Optional[str] = None,
    ) -> None:
        """Visualize an attention heatmap for a specific layer and head.

        Args:
            attention_weights: Tensor (num_layers, num_heads, seq_len, seq_len).
            tokens: List of token strings.
            layer: Layer index (negative indexing supported).
            head: Attention head index.
            save_path: If provided, save the figure to this path.
        """
        attn = attention_weights[layer, head].numpy()

        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(
            attn,
            xticklabels=tokens,
            yticklabels=tokens,
            cmap="viridis",
            ax=ax,
            square=True,
        )
        ax.set_title(f"Attention Heatmap  (Layer {layer}, Head {head})")
        ax.set_xlabel("Key Tokens")
        ax.set_ylabel("Query Tokens")
        plt.xticks(rotation=90, fontsize=7)
        plt.yticks(rotation=0, fontsize=7)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Attention heatmap saved to {save_path}")

        plt.show()

    def get_important_tokens(
        self,
        attention_weights: torch.Tensor,
        tokens: List[str],
        top_k: int = 10,
    ) -> List[Tuple[str, float]]:
        """Return the top-k tokens ranked by aggregated attention received.

        Attention is averaged across all layers and heads, then summed over the
        query dimension to get a per-token importance score.

        Args:
            attention_weights: Tensor (num_layers, num_heads, seq_len, seq_len).
            tokens: List of token strings.
            top_k: Number of top tokens to return.

        Returns:
            List of (token, score) tuples sorted by descending importance.
        """
        # Average over layers and heads -> (seq_len, seq_len)
        avg_attention = attention_weights.mean(dim=(0, 1)).numpy()

        # Sum attention received from all query positions for each key position
        token_importance = avg_attention.sum(axis=0)

        # Pair tokens with their scores and sort
        token_scores = list(zip(tokens, token_importance.tolist()))
        token_scores.sort(key=lambda x: x[1], reverse=True)

        # Filter out special tokens
        special = {"[CLS]", "[SEP]", "[PAD]", "<s>", "</s>", "<pad>"}
        filtered = [(t, s) for t, s in token_scores if t not in special]

        return filtered[:top_k]
