"""
SILK: Smooth InterpoLation frameworK

Complete Transformer implementation for motion in-betweening.

Architecture:
- Input projection: din → d_model
- 6 Transformer encoder layers
- Output projection: d_model → dout

Key features:
- Relative positional embeddings (learned)
- No masking (bidirectional attention)
- Multi-head attention (8 heads)
- Large model (d_model=1024, d_ff=4096)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class RelativePositionalEmbedding(nn.Module):
    """
    Learned relative positional embeddings for Transformer attention.

    Instead of absolute positions, encodes the DISTANCE between positions.
    This allows generalization to sequence lengths beyond training.

    Args:
        n_heads: Number of attention heads
        max_relative_position: Maximum relative distance to encode
    """
    def __init__(self, n_heads, max_relative_position=128):
        super().__init__()
        self.n_heads = n_heads
        self.max_relative_position = max_relative_position

        # Learnable relative position bias for each head
        # Shape: (n_heads, 2 * max_relative_position + 1)
        # Index: [0] = distance -max, [max] = distance 0, [2*max] = distance +max
        self.relative_position_bias = nn.Parameter(
            torch.zeros(n_heads, 2 * max_relative_position + 1)
        )

        # Initialize with small random values
        nn.init.normal_(self.relative_position_bias, std=0.02)

    def forward(self, seq_len):
        """
        Generate relative position bias for attention scores.

        Args:
            seq_len: Length of sequence

        Returns:
            bias: (n_heads, seq_len, seq_len) relative position bias
        """
        # Create position indices
        # positions[i] = i for i in range(seq_len)
        positions = torch.arange(seq_len, device=self.relative_position_bias.device)

        # Compute relative distances
        # relative[i, j] = i - j (distance from position i to position j)
        relative_positions = positions[:, None] - positions[None, :]  # (seq_len, seq_len)

        # Clip to max relative distance
        relative_positions = torch.clamp(
            relative_positions,
            -self.max_relative_position,
            self.max_relative_position
        )

        # Convert to indices (shift by max to make non-negative)
        relative_position_indices = relative_positions + self.max_relative_position

        # Gather biases for each head
        # bias[h, i, j] = relative_position_bias[h, relative_distance(i, j)]
        bias = self.relative_position_bias[:, relative_position_indices]  # (n_heads, seq_len, seq_len)

        return bias


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention with relative positional embeddings.

    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        dropout: Dropout probability
        max_relative_position: Maximum relative distance for positional bias
    """
    def __init__(self, d_model, n_heads, dropout=0.1, max_relative_position=128):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # Q, K, V projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # Output projection
        self.W_o = nn.Linear(d_model, d_model)

        # Relative positional embeddings
        self.relative_position = RelativePositionalEmbedding(n_heads, max_relative_position)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        Apply multi-head self-attention.

        Args:
            x: (batch, seq_len, d_model) input
            mask: (batch, seq_len, seq_len) attention mask (NOT USED in SILK)

        Returns:
            output: (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape

        # 1. Project to Q, K, V and split into heads
        # (batch, seq_len, d_model) -> (batch, n_heads, seq_len, d_k)
        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # 2. Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)  # (batch, n_heads, seq_len, seq_len)

        # 3. Add relative positional bias
        relative_bias = self.relative_position(seq_len)  # (n_heads, seq_len, seq_len)
        scores = scores + relative_bias.unsqueeze(0)  # Broadcast across batch

        # 4. Apply mask if provided (SILK doesn't use masking!)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # 5. Apply softmax and dropout
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # 6. Apply attention to values
        output = torch.matmul(attention_weights, V)  # (batch, n_heads, seq_len, d_k)

        # 7. Concatenate heads and project
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.W_o(output)

        return output


class FeedForward(nn.Module):
    """
    Position-wise feed-forward network.

    Two linear layers with GELU activation:
    FFN(x) = W2(GELU(W1(x)))

    Args:
        d_model: Model dimension
        d_ff: Feed-forward dimension (typically 4 * d_model)
        dropout: Dropout probability
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = self.linear1(x)
        x = F.gelu(x)  # GELU activation
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class TransformerEncoderLayer(nn.Module):
    """
    Single Transformer encoder layer.

    Architecture:
        x -> [Multi-Head Attention] -> [Add & Norm] ->
             [Feed-Forward] -> [Add & Norm] -> output

    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        d_ff: Feed-forward dimension
        dropout: Dropout probability
    """
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()

        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        Apply one Transformer encoder layer.

        Args:
            x: (batch, seq_len, d_model)
            mask: (batch, seq_len, seq_len) attention mask

        Returns:
            output: (batch, seq_len, d_model)
        """
        # Pre-norm: LayerNorm before sublayers
        
        # Multi-head attention with residual connection
        attn_output = self.attention(self.norm1(x), mask)
        x = x + self.dropout(attn_output)

        # Feed-forward with residual connection
        ff_output = self.feed_forward(self.norm2(x))
        x = x + self.dropout(ff_output)

        return x


class SILK(nn.Module):
    """
    SILK: Smooth InterpoLation frameworK

    Complete Transformer model for motion in-betweening.

    Architecture:
        Input (batch, seq_len, din=404)
            ↓
        Input Projection: din → d_model=1024
            ↓
        6× Transformer Encoder Layers
            ├─ Multi-Head Attention (8 heads)
            ├─ Relative Positional Bias
            ├─ Add & Norm
            ├─ Feed-Forward (d_ff=4096)
            └─ Add & Norm
            ↓
        Output Projection: d_model → dout=202
            ↓
        Output (batch, seq_len, dout=202)

    Args:
        din: Input feature dimension (404 for LAFAN1 with 22 joints)
        dout: Output feature dimension (202 for LAFAN1 with 22 joints)
        d_model: Model dimension (default: 1024)
        n_layers: Number of Transformer layers (default: 6)
        n_heads: Number of attention heads (default: 8)
        d_ff: Feed-forward dimension (default: 4096)
        dropout: Dropout probability (default: 0.1)
        max_seq_len: Maximum sequence length for positional encoding (default: 128)
    """
    def __init__(
        self,
        din=404,
        dout=202,
        d_model=1024,
        n_layers=6,
        n_heads=8,
        d_ff=4096,
        dropout=0.0,
        max_seq_len=128
    ):
        super().__init__()

        self.din = din
        self.dout = dout
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads

        # Input projection: din → d_model
        self.input_projection = nn.Linear(din, d_model)

        # Transformer encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # Output projection: d_model → dout
        self.output_projection = nn.Linear(d_model, dout)

        # Dropout
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, mask=None):
        """
        Forward pass through SILK model.

        Args:
            x: (batch, seq_len, din) input features
               Should have transition frames zero-filled!
            mask: (batch, seq_len, seq_len) attention mask
                  NOT USED in SILK (all frames attend to all frames)

        Returns:
            output: (batch, seq_len, dout) predicted features
        """
        # Project input to model dimension
        x = self.input_projection(x)  # (batch, seq_len, d_model)
        x = self.dropout(x)

        # Apply Transformer encoder layers
        for layer in self.encoder_layers:
            x = layer(x, mask=None)  # SILK doesn't use masking!

        # Project to output dimension
        output = self.output_projection(x)  # (batch, seq_len, dout)

        return output

    def count_parameters(self):
        """Count total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_silk_model(num_joints=22, d_model=1024, n_layers=6, n_heads=8, d_ff=4096, dropout=0.1):
    """
    Create SILK model with configurable hyperparameters.

    Args:
        num_joints: Number of joints in skeleton (22 for LAFAN1)
        d_model: Model dimension (default: 1024)
        n_layers: Number of Transformer layers (default: 6)
        n_heads: Number of attention heads (default: 8)
        d_ff: Feed-forward dimension (default: 4096)
        dropout: Dropout probability (default: 0.1)

    Returns:
        model: SILK model instance
    """
    # Compute feature dimensions
    din = 18 * num_joints + 8   # 404 for 22 joints
    dout = 9 * num_joints + 4   # 202 for 22 joints

    model = SILK(
        din=din,
        dout=dout,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_ff=d_ff,
        dropout=dropout,
        max_seq_len=128
    )

    print(f"SILK Model Created:")
    print(f"  Input dimension: {din}")
    print(f"  Output dimension: {dout}")
    print(f"  Model dimension: {d_model}")
    print(f"  Layers: {n_layers}")
    print(f"  Heads: {n_heads}")
    print(f"  Feed-forward dim: {d_ff}")
    print(f"  Dropout: {dropout}")
    print(f"  Total parameters: {model.count_parameters():,}")

    return model


if __name__ == "__main__":
    # Test model creation
    model = create_silk_model(num_joints=22)

    # Test forward pass
    batch_size = 4
    seq_len = 20
    din = 404

    x = torch.randn(batch_size, seq_len, din)
    output = model(x)

    print(f"\nTest forward pass:")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {output.shape}")
    print(f"  Model works!")
