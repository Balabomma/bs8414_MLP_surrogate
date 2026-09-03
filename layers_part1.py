"""Layer primitives for the Part1 surrogate, vendored verbatim.

Extracted from this project's `model.py` (the 60-sim corpus model) so the Part1
pipeline stands alone: the published repository is the Part1 geometry-corpus
surrogate and carries no 60-sim code.

VERBATIM, not re-implemented. The blocks below are byte-for-byte the ones every
existing result was produced with — a re-implementation differing in an
initialisation constant or a basis width would silently invalidate comparison
against those runs. Same reasoning, and the same pattern, as
`kan_layers_part1.py` in the FunDiff-KAN project.

Verify a symbol against the original with:

    python -c "import inspect, hashlib, layers_part1 as L; \
               print(hashlib.sha256(inspect.getsource(L.MLPLinear).encode()).hexdigest()[:16])"
"""

import math
import torch
import torch.nn as nn

def matched_hidden(in_features, out_features, num_knots):
    """Hidden width making MLPLinear match KANLinear(in, out, num_knots) in params."""
    target = out_features * in_features * (num_knots + 1) + in_features
    return max(1, int(round(target / (in_features + out_features + 1))))

class MLPLinear(nn.Module):
    """Conventional MLP block: LayerNorm(W2 · GELU(W1 x + b1) + b2).

    Same interface as the KAN block it replaces — accepts (B, F) or (B, T, F) —
    and keeps the output LayerNorm so training stability is not a confound.
    Fixed GELU on nodes instead of learnable B-splines on edges.
    """

    def __init__(self, in_features, out_features, num_knots=8, match_params=True,
                 hidden=None):
        super().__init__()
        if hidden is None:
            hidden = (matched_hidden(in_features, out_features, num_knots)
                      if match_params else out_features)
        self.in_features = in_features
        self.out_features = out_features
        self.hidden = hidden
        self.fc1 = nn.Linear(in_features, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, out_features)
        self.ln = nn.LayerNorm(out_features)

    def forward(self, x):
        return self.ln(self.fc2(self.act(self.fc1(x))))

class TimeEncoding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model

    def forward(self, time_array):
        pe = torch.zeros(len(time_array), self.d_model, device=time_array.device)
        pos = time_array.unsqueeze(1) * 1000
        div = torch.exp(torch.arange(0, self.d_model, 2, device=time_array.device).float()
                        * -(math.log(10000.0) / self.d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe

class MultiScaleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv3 = nn.Conv1d(in_ch, out_ch // 3, kernel_size=3, padding=1)
        self.conv9 = nn.Conv1d(in_ch, out_ch // 3, kernel_size=9, padding=4)
        self.conv27 = nn.Conv1d(in_ch, out_ch - 2 * (out_ch // 3), kernel_size=27,
                                padding=13)
        self.norm = nn.LayerNorm(out_ch)

    def forward(self, x):
        xt = x.transpose(1, 2)
        return self.norm(torch.cat([self.conv3(xt), self.conv9(xt), self.conv27(xt)],
                                   dim=1).transpose(1, 2))
