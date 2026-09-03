"""Causal-structure variant: temperature driven by PREDICTED total heat release.

The baseline drives its plume channels from the prescribed crib source alone,

    q(t) = hrr_mw * ramp(t)          Q(t) = hrr_mw * cumulative_ramp(t)
    q23  = q(t)^(2/3)                (plume dT ~ HRR^(2/3))

which makes every plume channel a smooth, monotone function of a single scalar.
That is exactly why it fails at the top of the sweep: between 2100 and
2333 kW m-2 the FDS response accelerates by up to 9.9x as the cladding core
ignites, and no smooth function of the prescribed source can turn a corner it has
never been shown.

This variant inserts the physical mediator. A non-negative cladding head predicts
the cladding's own contribution over time, and the plume channels are rebuilt from
the TOTAL:

    q_tot(t) = q_crib(t) + softplus(cladding_head(param_embed, t))

The additive, non-negative form is the inductive bias, and it is the physically
correct one: burning cladding adds heat and never removes it, so q_tot >= q_crib
by construction. The regime change becomes a state the network can represent --
cladding_head stays near zero while the cladding is inert and rises steeply once
it is involved -- rather than a curvature it must fake.

q_tot is supervised where FDS _hrr.csv output exists (48 of 60 runs, see
hrr_targets.py), so the mediator is anchored to measured physics rather than left
as a free latent.

forward() returns (prediction, attention_weights, q_tot) -- one element more than
the baseline. Callers must unpack three.
"""
import torch
import torch.nn as nn

from model import MLPAttentionLSTM, HRR_MW_COL
from features_v4 import HRR_MW_SCALE


class CausalMLPAttentionLSTM(MLPAttentionLSTM):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        e = self.param_encoder.output_dim if hasattr(self.param_encoder, "output_dim") \
            else self.skip_proj.in_features
        d_time = self.time_encoding.d_model if hasattr(self.time_encoding, "d_model") else e
        hidden = max(32, e)
        # Per-timestep head over (design embedding, time embedding). Deliberately
        # small: it has to represent one transition, not a temperature field.
        self.cladding_head = nn.Sequential(
            nn.Linear(e + d_time + 1, hidden), nn.GELU(),
            nn.Linear(hidden, hidden // 2), nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )
        # Start small so the model begins near the baseline's behaviour, but not
        # so small that the auxiliary gradient cannot lift it. An earlier version
        # used zero weights with bias -3.0 and, at lambda_HRR = 0.05, the head
        # never left its initialisation: the learned cladding contribution stayed
        # at softplus(-3) = 0.049 MW for every source level while the true excess
        # ranged 0.57-0.98 MW. Weight init is now small-random, not zero.
        nn.init.normal_(self.cladding_head[-1].weight, std=1e-2)
        nn.init.constant_(self.cladding_head[-1].bias, -1.0)

    def forward(self, params, time_array, anchor):
        B, T = params.shape[0], len(time_array)

        param_embed = self.param_encoder(params, anchor)
        time_embed = self.time_encoding(time_array)

        delta = time_array.unsqueeze(0) - anchor.unsqueeze(1)
        bump = torch.exp(-(delta / 0.08) ** 2)

        hrr_mw = params[:, HRR_MW_COL:HRR_MW_COL + 1] * HRR_MW_SCALE
        q_crib = hrr_mw * self.ramp_f.unsqueeze(0)                  # (B, T)

        # --- the mediator -------------------------------------------------
        pe = param_embed.unsqueeze(1).expand(-1, T, -1)
        te = time_embed.unsqueeze(0).expand(B, -1, -1)
        head_in = torch.cat([pe, te, delta.unsqueeze(-1)], dim=-1)
        q_clad = nn.functional.softplus(self.cladding_head(head_in).squeeze(-1))
        q_tot = q_crib + q_clad                                     # (B, T), >= q_crib

        Q = torch.cumsum(q_tot, dim=1) / float(T)
        q23 = q_tot.clamp(min=1e-6) ** (2.0 / 3.0)

        combined = torch.cat([
            pe, te,
            delta.unsqueeze(-1),
            bump.unsqueeze(-1),
            q_tot.unsqueeze(-1),
            Q.unsqueeze(-1),
            q23.unsqueeze(-1),
        ], dim=-1)

        x = self.input_proj(combined)
        x = x + self.multi_scale(x)

        lstm_out, _ = self.lstm(x)
        attn_out, attn_w = self.attention(lstm_out, lstm_out, lstm_out)
        features = self.attn_norm(lstm_out + attn_out)

        groups = [self.decoder_ext_lv1(features), self.decoder_ext_lv2(features)]
        if self.decoder_ins_lv2 is not None:
            groups.append(self.decoder_ins_lv2(features))
        preds = torch.cat(groups, dim=-1)

        skip = self.skip_proj(param_embed).unsqueeze(1).expand(-1, T, -1)
        out = torch.maximum(preds + skip, self.ambient_scaled)
        return out, attn_w, q_tot
