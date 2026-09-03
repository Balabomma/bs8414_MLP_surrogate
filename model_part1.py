"""MLP-Attention-LSTM for the BS 8414 Part1 geometry-variant corpus.

This is the **MLP ablation control** for `bs8414_KAN_surrogate/model_part1.py`,
carried onto the Part1 corpus. The question is unchanged: does the learnable
B-spline edge activation buy anything over a conventional MLP once inputs,
temporal backbone, loss, split and eval are held fixed?

Held fixed vs the Part1 KAN model (same shared files, byte-identical):
  - `config_part1.py`, `data_loader_part1.py`, `train_part1.py`,
    `evaluate_part1.py` — verified by `verify_parity_part1.py`
  - conditioning: cladding(12) + insulation(5) + geometry(8) embeddings + 13
    material features, identical embedding widths
  - backbone: sinusoidal TimeEncoding, input_proj, MultiScaleConv (k=3/9/27),
    2-layer BiLSTM(96), 4-head self-attention + LayerNorm, param->output skips
  - two thermocouple group decoders + the HRR head, same shapes and dropout
  - hard ambient / zero-HRR output clamps

Changed (the one variable):
  - every `KANLinear` edge block -> `MLPLinear` = LayerNorm(W2·GELU(W1x+b1)+b2)

`MLPLinear` is imported from this project's `model.py`, not re-declared, so the
block under test is the same one the 60-sim ablation used. Capacity is matched,
not reduced: each block sizes its hidden width so its parameter count matches the
KANLinear it replaces (see `matched_hidden` in `model.py`). `--plain` equivalent:
construct with `match_params=False`.

Disclosed second difference, unchanged from the 60-sim ablation: the spline
regulariser has no MLP counterpart, so `LAMBDA_REG = 0.0` here.
"""
import torch
import torch.nn as nn

from config_part1 import (
    N_CLADDING, N_INSULATION, N_GEOMETRY, N_MATERIAL_FEATURES,
    CLADDING_EMBED_DIM, INSULATION_EMBED_DIM, GEOMETRY_EMBED_DIM,
    N_SENSORS, GROUP_SIZES, N_HRR_CHANNELS, N_TIMESTEPS,
    EMBEDDING_DIM, LSTM_HIDDEN_SIZE, ATTENTION_HEADS, DROPOUT, NUM_KNOTS,
    T_AMBIENT, COL_CLADDING, COL_INSULATION, COL_GEOM,
)
from layers_part1 import MLPLinear, TimeEncoding, MultiScaleConv

MATCH_PARAMS = True


class Part1ParameterEncoder(nn.Module):
    """[cladding_id, insulation_id, geom_id, 13 material] -> embedding."""

    def __init__(self, output_dim=EMBEDDING_DIM, num_knots=NUM_KNOTS,
                 match_params=MATCH_PARAMS):
        super().__init__()
        self.cladding_embedding = nn.Embedding(N_CLADDING, CLADDING_EMBED_DIM)
        self.insulation_embedding = nn.Embedding(N_INSULATION, INSULATION_EMBED_DIM)
        self.geometry_embedding = nn.Embedding(N_GEOMETRY, GEOMETRY_EMBED_DIM)

        input_dim = (CLADDING_EMBED_DIM + INSULATION_EMBED_DIM
                     + GEOMETRY_EMBED_DIM + N_MATERIAL_FEATURES)
        self.block1 = MLPLinear(input_dim, 48, num_knots=num_knots,
                                match_params=match_params)
        self.block2 = MLPLinear(48, output_dim, num_knots=num_knots,
                                match_params=match_params)
        self.output_dim = output_dim

    def forward(self, params):
        clad = self.cladding_embedding(params[:, COL_CLADDING].long())
        ins = self.insulation_embedding(params[:, COL_INSULATION].long())
        geom = self.geometry_embedding(params[:, COL_GEOM].long())
        x = torch.cat([clad, ins, geom, params[:, 3:]], dim=-1)
        return self.block2(self.block1(x))


class Part1SensorDecoder(nn.Module):
    """Per-group decoder — same shapes and dropout placement as the KAN's."""

    def __init__(self, input_dim, n_outputs, num_knots=NUM_KNOTS, dropout=0.15,
                 match_params=MATCH_PARAMS):
        super().__init__()
        hidden = max(32, n_outputs * 2)
        self.block1 = MLPLinear(input_dim, hidden, num_knots=num_knots,
                                match_params=match_params)
        self.dropout1 = nn.Dropout(dropout)
        self.block2 = MLPLinear(hidden, n_outputs, num_knots=num_knots,
                                match_params=match_params)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        return self.block2(self.dropout1(self.block1(self.dropout2(x))))


class Part1MLPAttentionLSTM(nn.Module):
    """Shared temporal backbone, two output heads (thermocouples and HRR)."""

    def __init__(self, n_sensors=N_SENSORS, n_hrr_channels=N_HRR_CHANNELS,
                 hidden_size=LSTM_HIDDEN_SIZE, embedding_dim=EMBEDDING_DIM,
                 n_heads=ATTENTION_HEADS, dropout=DROPOUT, num_knots=NUM_KNOTS,
                 match_params=MATCH_PARAMS):
        super().__init__()
        self.n_sensors = n_sensors
        self.n_hrr_channels = n_hrr_channels

        self.param_encoder = Part1ParameterEncoder(
            output_dim=embedding_dim, num_knots=num_knots, match_params=match_params)
        self.time_encoding = TimeEncoding(d_model=embedding_dim)

        self.input_proj = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_size), nn.GELU(), nn.Dropout(dropout),
        )
        self.multi_scale = MultiScaleConv(hidden_size, hidden_size)

        self.lstm = nn.LSTM(
            input_size=hidden_size, hidden_size=hidden_size,
            num_layers=2, batch_first=True, bidirectional=True, dropout=dropout,
        )

        lstm_out = hidden_size * 2
        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_out, num_heads=n_heads, dropout=dropout, batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(lstm_out)

        self.sensor_decoders = nn.ModuleList([
            Part1SensorDecoder(lstm_out, size, num_knots=num_knots,
                               match_params=match_params)
            for size in GROUP_SIZES
        ])
        self.hrr_decoder = Part1SensorDecoder(lstm_out, n_hrr_channels,
                                              num_knots=num_knots,
                                              match_params=match_params)

        self.skip_proj = nn.Linear(embedding_dim, n_sensors)
        self.hrr_skip_proj = nn.Linear(embedding_dim, n_hrr_channels)

        self.register_buffer("ambient_scaled", torch.full((n_sensors,), -1e9))
        self.register_buffer("hrr_floor_scaled", torch.full((n_hrr_channels,), -1e9))

    def set_output_scaling(self, tc_scaler, hrr_scaler, hrr_nonnegative_idx=(0,),
                           t_ambient=T_AMBIENT):
        amb = (t_ambient - torch.as_tensor(tc_scaler.mean, dtype=torch.float32)) \
            / torch.as_tensor(tc_scaler.scale, dtype=torch.float32)
        self.ambient_scaled.copy_(amb.to(self.ambient_scaled.device))

        floor = torch.full((self.n_hrr_channels,), -1e9)
        mean = torch.as_tensor(hrr_scaler.mean, dtype=torch.float32)
        scale = torch.as_tensor(hrr_scaler.scale, dtype=torch.float32)
        for i in hrr_nonnegative_idx:
            floor[i] = (0.0 - mean[i]) / scale[i]
        self.hrr_floor_scaled.copy_(floor.to(self.hrr_floor_scaled.device))

    def forward(self, params, time_array):
        B, T = params.shape[0], len(time_array)

        param_embed = self.param_encoder(params)
        time_embed = self.time_encoding(time_array)

        combined = torch.cat([
            param_embed.unsqueeze(1).expand(-1, T, -1),
            time_embed.unsqueeze(0).expand(B, -1, -1),
        ], dim=-1)

        x = self.input_proj(combined)
        x = x + self.multi_scale(x)

        lstm_out, _ = self.lstm(x)
        attn_out, attn_w = self.attention(lstm_out, lstm_out, lstm_out)
        features = self.attn_norm(lstm_out + attn_out)

        tc = torch.cat([d(features) for d in self.sensor_decoders], dim=-1)
        tc = tc + self.skip_proj(param_embed).unsqueeze(1).expand(-1, T, -1)
        tc = torch.maximum(tc, self.ambient_scaled)

        hrr = self.hrr_decoder(features)
        hrr = hrr + self.hrr_skip_proj(param_embed).unsqueeze(1).expand(-1, T, -1)
        hrr = torch.maximum(hrr, self.hrr_floor_scaled)

        return tc, hrr, attn_w


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ── uniform interface (see the KAN project's model_part1.py) ──────────────
MODEL_NAME = "MLP-Attention-LSTM (Part1)"
Part1Surrogate = Part1MLPAttentionLSTM

# No spline weights to regularise. Disclosed difference, not an omission.
LAMBDA_REG = 0.0


def regularization(model):
    return torch.zeros((), device=next(model.parameters()).device)


if __name__ == "__main__":
    import numpy as np
    from config_part1 import N_INPUT_PARAMS
    from data_loader_part1 import ChannelScaler

    matched = Part1Surrogate()
    plain = Part1Surrogate(match_params=False)
    print(f"{MODEL_NAME}")
    print(f"  param-matched : {count_parameters(matched):,}")
    print(f"  plain (h=out) : {count_parameters(plain):,}")
    print(f"  KAN Part1 ref : 850,765")

    tc_scaler = ChannelScaler().load_state_dict(
        {"mean": np.full(N_SENSORS, 200.0, np.float32),
         "scale": np.full(N_SENSORS, 150.0, np.float32)})
    hrr_scaler = ChannelScaler().load_state_dict(
        {"mean": np.full(N_HRR_CHANNELS, 800.0, np.float32),
         "scale": np.full(N_HRR_CHANNELS, 600.0, np.float32)})
    matched.set_output_scaling(tc_scaler, hrr_scaler)

    p = torch.rand(4, N_INPUT_PARAMS)
    p[:, COL_CLADDING] = torch.randint(0, N_CLADDING, (4,)).float()
    p[:, COL_INSULATION] = torch.randint(0, N_INSULATION, (4,)).float()
    p[:, COL_GEOM] = torch.randint(0, N_GEOMETRY, (4,)).float()

    tc, hrr, _ = matched(p, torch.linspace(0, 1, N_TIMESTEPS))
    print(f"  tc  {tuple(tc.shape)}  hrr {tuple(hrr.shape)}")
