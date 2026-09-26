"""
CaptionAI — Phase 5: Bahdanau Attention Decoder (PyTorch)

Implements Bahdanau attention over spatial DenseNet201 features.

Encoder: DenseNet201 spatial features (7x7x1920) - frozen, precomputed.
Decoder: LSTM with Bahdanau attention over spatial locations.

Full-sequence teacher forcing: the decoder consumes the whole input
sequence [START] + caption[:-1] and emits logits for every position
(batch, seq_len, vocab_size). The target is caption[1:] (incl. END).
This is ~10x fewer training rows than per-token prefix expansion and
trains much faster.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_spatial_encoder(device: str = "cpu") -> nn.Module:
    """Build the frozen DenseNet201 spatial encoder.

    Returns a module that maps (batch, 3, 224, 224) -> (batch, 1920, 7, 7).
    Output is BEFORE the final BatchNorm (norm5) + activation, matching the
    original AttentionCaptionModel definition. Must be identical in
    precompute_spatial_features.py and training so feature files line up.
    """
    from torchvision import models
    densenet = models.densenet201(weights=models.DenseNet201_Weights.IMAGENET1K_V1)
    encoder = nn.Sequential(*list(densenet.features.children())[:-1])
    for param in encoder.parameters():
        param.requires_grad = False
    encoder.eval()
    encoder.to(device)
    return encoder


class BahdanauAttention(nn.Module):
    """Bahdanau additive attention over spatial features."""

    def __init__(self, encoder_dim: int, decoder_dim: int, attention_dim: int):
        super().__init__()
        self.encoder_proj = nn.Linear(encoder_dim, attention_dim)
        self.decoder_proj = nn.Linear(decoder_dim, attention_dim)
        self.v = nn.Linear(attention_dim, 1)

    def forward(self, encoder_features: torch.Tensor, decoder_hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            encoder_features: (batch, num_pixels, encoder_dim)
            decoder_hidden: (batch, decoder_dim)
        Returns:
            context: (batch, encoder_dim)
            alpha: (batch, num_pixels)
        """
        encoder_proj = self.encoder_proj(encoder_features)              # (B, P, att)
        decoder_proj = self.decoder_proj(decoder_hidden).unsqueeze(1)   # (B, 1, att)
        combined = torch.tanh(encoder_proj + decoder_proj)              # (B, P, att)
        scores = self.v(combined).squeeze(-1)                           # (B, P)
        alpha = F.softmax(scores, dim=1)                                # (B, P)
        context = (encoder_features * alpha.unsqueeze(-1)).sum(dim=1)  # (B, enc)
        return context, alpha


class AttentionDecoder(nn.Module):
    """LSTM decoder with Bahdanau attention over spatial DenseNet201 features."""

    def __init__(
        self,
        vocab_size: int,
        max_length: int,
        encoder_dim: int = 1920,
        embed_dim: int = 256,
        decoder_dim: int = 512,
        attention_dim: int = 512,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.attention = BahdanauAttention(encoder_dim, decoder_dim, attention_dim)
        self.lstm = nn.LSTM(embed_dim + encoder_dim, decoder_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(decoder_dim, vocab_size)

    def forward(
        self,
        encoder_features: torch.Tensor,
        captions: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Full-sequence teacher forcing.

        Args:
            encoder_features: (batch, num_pixels, encoder_dim)
            captions: (batch, seq_len) input tokens: [START] + caption[:-1]
            hidden: optional (h, c) initial state
        Returns:
            logits: (batch, seq_len, vocab_size)
            hidden: final (h, c)
        """
        batch, max_len = captions.shape
        embeddings = self.embedding(captions)  # (B, L, embed)

        if hidden is None:
            h = torch.zeros(1, batch, self.decoder_dim, device=encoder_features.device)
            c = torch.zeros(1, batch, self.decoder_dim, device=encoder_features.device)
            hidden = (h, c)

        logits_list = []
        for t in range(max_len):
            context, _ = self.attention(encoder_features, hidden[0].squeeze(0))  # (B, enc)
            embed = embeddings[:, t:t + 1, :]                                    # (B, 1, embed)
            lstm_input = torch.cat([embed, context.unsqueeze(1)], dim=-1)        # (B, 1, embed+enc)
            output, hidden = self.lstm(lstm_input, hidden)                       # (B, 1, dec)
            logit = self.fc(self.dropout(output.squeeze(1)))                     # (B, V)
            logits_list.append(logit)

        logits = torch.stack(logits_list, dim=1)  # (B, L, V)
        return logits, hidden

    @torch.no_grad()
    def generate(
        self,
        encoder_features: torch.Tensor,
        start_id: int,
        end_id: int,
        max_len: int,
        device: torch.device,
    ):
        """Greedy caption generation for a single image feature.

        Returns list of token ids (excluding START/END).
        """
        self.eval()
        batch = encoder_features.shape[0]
        h = torch.zeros(1, batch, self.decoder_dim, device=device)
        c = torch.zeros(1, batch, self.decoder_dim, device=device)
        hidden = (h, c)

        tokens = torch.full((batch, 1), start_id, dtype=torch.long, device=device)
        generated = []
        for _ in range(max_len):
            context, _ = self.attention(encoder_features, hidden[0].squeeze(0))
            embed = self.embedding(tokens)  # (B, 1, embed)
            lstm_input = torch.cat([embed, context.unsqueeze(1)], dim=-1)
            output, hidden = self.lstm(lstm_input, hidden)
            logit = self.fc(self.dropout(output.squeeze(1)))  # (B, V)
            next_tok = logit.argmax(dim=1, keepdim=True)        # (B, 1)
            generated.append(next_tok)
            tokens = next_tok
            if int(next_tok[0, 0]) == end_id:
                break
        # concat along time -> (B, T)
        seq = torch.cat(generated, dim=1)
        return seq[0].tolist()

    @torch.no_grad()
    def beam_search(
        self,
        encoder_features: torch.Tensor,
        start_id: int,
        end_id: int,
        max_len: int,
        device: torch.device,
        beam_width: int = 3,
        length_penalty: float = 0.7,
        return_score: bool = False,
    ):
        """Beam-search caption generation for a single image feature.

        Returns list of token ids (excluding START/END).
        """
        self.eval()
        batch = encoder_features.shape[0]

        def init_hidden():
            h = torch.zeros(1, batch, self.decoder_dim, device=device)
            c = torch.zeros(1, batch, self.decoder_dim, device=device)
            return (h, c)

        def norm_score(log_prob, seq):
            return log_prob / (seq.shape[1] ** length_penalty)

        # beams: list of (log_prob, seq_tensor(B, t), hidden)
        beams = [(0.0, torch.full((batch, 1), start_id, dtype=torch.long, device=device), init_hidden())]
        finished = []

        for _ in range(max_len):
            if not beams:
                break
            new_beams = []
            for log_prob, seq, hidden in beams:
                if int(seq[0, -1]) == end_id:
                    finished.append((log_prob, seq))
                    continue
                tokens = seq[:, -1:]
                context, _ = self.attention(encoder_features, hidden[0].squeeze(0))
                embed = self.embedding(tokens)
                lstm_input = torch.cat([embed, context.unsqueeze(1)], dim=-1)
                output, new_hidden = self.lstm(lstm_input, hidden)
                logit = self.fc(self.dropout(output.squeeze(1)))   # (B, V)
                log_probs = torch.log_softmax(logit, dim=1)[0]      # (V,)
                topk = torch.topk(log_probs, beam_width)
                for score, idx in zip(topk.values, topk.indices):
                    nxt = idx.item()
                    n_seq = torch.cat([seq, torch.full((batch, 1), nxt, dtype=torch.long, device=device)], dim=1)
                    new_beams.append((log_prob + score.item(), n_seq, new_hidden))
            new_beams.sort(key=lambda x: norm_score(x[0], x[1]), reverse=True)
            beams = new_beams[:beam_width]
            if all(int(b[1][0, -1]) == end_id for b in beams):
                for b in beams:
                    finished.append((norm_score(b[0], b[1]), b[1]))
                break

        if finished:
            finished.sort(key=lambda x: x[0], reverse=True)
            best = finished[0][1]
            best_score = finished[0][0]
        else:
            best = beams[0][1]
            best_score = beams[0][0]
        if return_score:
            return best[0].tolist(), float(best_score)
        return best[0].tolist()


class AttentionCaptionModel(nn.Module):
    """Encoder+decoder wrapper that accepts raw images (for inference only)."""

    def __init__(
        self,
        vocab_size: int,
        max_length: int = 35,
        encoder_dim: int = 1920,
        embed_dim: int = 256,
        decoder_dim: int = 512,
        attention_dim: int = 512,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.encoder = build_spatial_encoder()
        self.decoder = AttentionDecoder(
            vocab_size=vocab_size,
            max_length=max_length,
            embed_dim=embed_dim,
            decoder_dim=decoder_dim,
            attention_dim=attention_dim,
            dropout=dropout,
        )

    def forward(self, images: torch.Tensor, captions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: (batch, 3, 224, 224)
            captions: (batch, seq_len)
        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        with torch.no_grad():
            spatial = self.encoder(images)  # (B, 1920, 7, 7)
            b, ch, h, w = spatial.shape
            spatial = spatial.view(b, ch, h * w).permute(0, 2, 1)  # (B, 49, 1920)
        logits, _ = self.decoder(spatial, captions)
        return logits


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AttentionCaptionModel(vocab_size=8427, max_length=35).to(device)
    model.eval()
    images = torch.randn(2, 3, 224, 224).to(device)
    captions = torch.randint(0, 8427, (2, 35)).to(device)
    with torch.no_grad():
        logits = model(images, captions)
        print(f"Output shape: {logits.shape}")  # (2, 35, 8427)
        print("Model test passed!")
