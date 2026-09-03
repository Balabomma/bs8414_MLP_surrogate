"""Grid convergence index for the validated FDS configurations (manuscript Sec 3.2).

Celik et al. (2008), ASME J. Fluids Eng. 130(7):078001 -- the standard GCI
procedure, applied to the peak temperature rise at the two BR 135-relevant
thermocouples for each of the four validated DCLG configurations.

Meshes (coarse -> fine): h1 = 0.100 m, h2 = 0.091 m, h3 = 0.080 m
    r21 = h1/h2 = 1.0989      r32 = h2/h3 = 1.1375

Reports for each configuration and channel:
    eps21, eps32   solution changes
    s              sign(eps32/eps21); s < 0 => oscillatory (non-monotone) convergence,
                   which invalidates Richardson extrapolation
    p              observed order of convergence (iterated)
    phi_ext        Richardson-extrapolated value
    e_a            approximate relative error
    GCI21, GCI32   grid convergence indices with Fs = 1.25
"""
import os
import numpy as np

os.environ.setdefault("MLP_SPLIT", "70_15_15")
import grouped_split  # noqa: E402
from data_loader import build_dataset  # noqa: E402

FS = 1.25
H = {"M010": 0.100, "M009": 0.091, "M008": 0.080}
R21 = H["M010"] / H["M009"]
R32 = H["M009"] / H["M008"]
CHANNELS = ["External_LV1_main02(1003)", "External_LV2_main02(1029)"]
LABEL = {"External_LV1_main02(1003)": "DG1_1003 (L1)",
         "External_LV2_main02(1029)": "DG1_1029 (L2)"}


def mesh_of(chid):
    for k in ("M008", "M009", "M010"):
        if k in chid:
            return k
    if chid.endswith("_0_08") or chid.endswith("_08"):
        return "M008"
    if chid.endswith("_0_09"):
        return "M009"
    if chid.endswith("_0_1"):
        return "M010"
    return None


def observed_order(e21, e32, r21, r32, tol=1e-10, itmax=200):
    """Iteratively solve Celik Eq. (2)-(3) for the observed order p."""
    ratio = e32 / e21
    if ratio <= 0:
        return None  # oscillatory: p undefined
    p = np.log(abs(ratio)) / np.log(r21)
    for _ in range(itmax):
        s = 1.0 * np.sign(ratio)
        q = np.log((r21 ** p - s) / (r32 ** p - s))
        p_new = abs(np.log(abs(ratio)) + q) / np.log(r21)
        if abs(p_new - p) < tol:
            return p_new
        p = p_new
    return p


def main():
    params, outputs, masks, meta, names = build_dataset()
    ci = {c: names.index(c) for c in CHANNELS}

    # peak temperature rise for every anchor (HRRPUA = 2333.3) run
    peaks = {}
    for i, m in enumerate(meta):
        clad, hrr = grouped_split.config_key(m["chid"])
        if hrr != "2333":
            continue
        mk = mesh_of(m["chid"])
        for c in CHANNELS:
            s = outputs[i][:, ci[c]]
            peaks[(clad, mk, c)] = float(s.max() - s[0])

    print("=" * 92)
    print("  Grid convergence index -- peak temperature rise, validated configurations")
    print(f"  h1=0.100  h2=0.091  h3=0.080 m   r21={R21:.4f}  r32={R32:.4f}   Fs={FS}")
    print("=" * 92)
    hdr = (f"{'configuration':<22}{'channel':<16}{'phi1':>8}{'phi2':>8}{'phi3':>8}"
           f"{'p':>7}{'GCI21%':>9}{'GCI32%':>9}  conv")
    print(hdr); print("-" * len(hdr))

    rows = []
    for clad in sorted({k[0] for k in peaks}):
        for c in CHANNELS:
            try:
                phi1 = peaks[(clad, "M010", c)]
                phi2 = peaks[(clad, "M009", c)]
                phi3 = peaks[(clad, "M008", c)]
            except KeyError:
                continue
            e21 = phi2 - phi1
            e32 = phi3 - phi2
            if e21 == 0:
                continue
            ratio = e32 / e21
            p = observed_order(e21, e32, R21, R32)
            if ratio < 0 or p is None:
                conv = "OSCILLATORY"
                rows.append((clad, c, phi1, phi2, phi3, None, None, None, conv))
                print(f"{clad:<22}{LABEL[c]:<16}{phi1:>8.0f}{phi2:>8.0f}{phi3:>8.0f}"
                      f"{'--':>7}{'--':>9}{'--':>9}  {conv}")
                continue
            ext21 = (R21 ** p * phi1 - phi2) / (R21 ** p - 1)
            ea21 = abs((phi1 - phi2) / phi1)
            ea32 = abs((phi2 - phi3) / phi2)
            gci21 = FS * ea21 / (R21 ** p - 1) * 100
            gci32 = FS * ea32 / (R32 ** p - 1) * 100
            conv = "monotone"
            rows.append((clad, c, phi1, phi2, phi3, p, gci21, gci32, conv))
            print(f"{clad:<22}{LABEL[c]:<16}{phi1:>8.0f}{phi2:>8.0f}{phi3:>8.0f}"
                  f"{p:>7.2f}{gci21:>9.1f}{gci32:>9.1f}  {conv}")

    print("-" * len(hdr))
    osc = [r for r in rows if r[8] == "OSCILLATORY"]
    mono = [r for r in rows if r[8] == "monotone"]
    print(f"  {len(mono)}/{len(rows)} channel-configuration pairs converge monotonically; "
          f"{len(osc)}/{len(rows)} oscillate")
    if mono:
        ps = [r[5] for r in mono]
        gs = [r[6] for r in mono]
        print(f"  observed order p: {min(ps):.2f} - {max(ps):.2f} (mean {np.mean(ps):.2f})")
        print(f"  GCI21: {min(gs):.1f}% - {max(gs):.1f}% (mean {np.mean(gs):.1f}%)")
        print(f"  pairs with GCI21 < 5%: {sum(1 for g in gs if g < 5)}/{len(gs)}")
    if osc:
        print("  NOTE: oscillatory pairs invalidate Richardson extrapolation; the solution")
        print("        is not in the asymptotic range for those quantities.")


if __name__ == "__main__":
    main()
