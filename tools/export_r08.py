"""Export our own model of the lab cell R08 for r08.html.

Usage: python tools/export_r08.py <research_repo> <measured_curve.txt>
  research_repo        the project folder with data/ours/v1, data/training/ours_v1 and data/simulations/stage1
  measured_curve.txt   R08's J-V scan: columns V, I (mA), J (mA/cm2), P

Writes model/r08_encoder_0.js, model/r08_decoder_0.js (float16 ONNX as base64) and model/r08_data.js.
"""
import base64
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from export_onnx import Decoder, Encoder, OUT  # noqa: E402

FIT_FROM = -0.12  # V; RMS runs from here (skipping the start-of-scan dip) to the measured Voc, as in the calibration
VOC = 1.299


def read_oghma_jv(path):
    """OghmaNano jv csv (V, J in A/m2) -> V, J in mA/cm2."""
    a = np.loadtxt(path, comments="#")
    return a[:, 0], a[:, 1] / 10


def main(repo, measured):
    import onnx
    import onnxruntime as ort
    from onnxconverter_common import float16

    norm = json.load(open(os.path.join(repo, "data", "ours", "v1", "norm.json")))
    res = json.load(open(os.path.join(repo, "data", "training", "ours_v1", "results.json")))
    v = np.array(norm["voltages"])
    L, names = len(v), norm["params"]
    state = torch.load(os.path.join(repo, "data", "training", "ours_v1", "model.pt"), map_location="cpu")

    nets = {"encoder": (Encoder(L, len(names)), torch.zeros(1, L), "x", "y"),
            "decoder": (Decoder(L, len(names)), torch.zeros(1, len(names)), "y", "x")}
    sizes = {}
    for name, (net, dummy, i, o) in nets.items():
        net.load_state_dict({k[len(name) + 1:]: t for k, t in state.items() if k.startswith(name + ".")})
        net.eval()
        fp32 = os.path.join(OUT, f"r08_{name}_fp32.onnx")
        torch.onnx.export(net, dummy, fp32, input_names=[i], output_names=[o], dynamic_axes={i: {0: "n"}, o: {0: "n"}},
                          opset_version=17, dynamo=False)
        m = float16.convert_float_to_float16(onnx.load(fp32), keep_io_types=True)
        os.remove(fp32)
        b = m.SerializeToString()
        b64 = base64.b64encode(b).decode()
        with open(os.path.join(OUT, f"r08_{name}_0.js"), "w") as fh:
            fh.write(f'window.R08 = window.R08 || {{}}; R08.{name} = "{b64}";\n')
        sizes[name] = os.path.getsize(os.path.join(OUT, f"r08_{name}_0.js"))
        nets[name] = ort.InferenceSession(b)

    # check the float16 export on the held-out simulated devices of the training set
    d = os.path.join(repo, "data", "ours", "v1")
    x = np.load(os.path.join(d, "x_test.npy")).astype(np.float32)
    y = np.load(os.path.join(d, "y_test.npy")).astype(np.float32)
    yp = nets["encoder"].run(None, {"x": x})[0]
    xp = nets["decoder"].run(None, {"y": y})[0]
    j0, j1 = norm["J_range_mA_cm2"]
    r2 = 1 - ((y - yp) ** 2).sum(0) / ((y - y.mean(0)) ** 2).sum(0)

    meas = np.loadtxt(measured)
    jm = np.interp(v, meas[:, 0], meas[:, 2])
    fits = []
    for label, path, rms in [
        ("Solver fit, ETL traps (Stage 1)", os.path.join(repo, "data", "simulations", "stage1", "refit_phase3_Rsh_free_jv.csv"), None),
        ("Solver fit, HTL traps (Stage 1)", os.path.join(repo, "data", "simulations", "stage1", "refit_htl_Rsh_free_jv.csv"), None),
        ("Decoder fit, re-simulated in the solver", os.path.join(repo, "data", "training", "ours_v1", "resim_fit_jv.csv"), None)]:
        fv, fj = read_oghma_jv(path)
        fj_on_v = np.interp(v, fv, fj)
        keep = (v >= FIT_FROM) & (v <= VOC)
        fits.append({"label": label, "V": np.round(fv, 4).tolist(), "J": np.round(fj, 4).tolist(),
                     "rms": round(float(np.sqrt(np.mean((fj_on_v - jm)[keep] ** 2))), 3)})
    refit = {r["name"]: r for r in json.load(open(os.path.join(repo, "data", "simulations", "stage1", "refit.json")))}
    est = json.load(open(os.path.join(repo, "data", "training", "ours_v1", "estimate.json")))[0]

    def to_unit(p):
        lo, hi = np.array(norm["log10_min"]), np.array(norm["log10_max"])
        return ((np.log10([p[n] for n in names]) - lo) / (hi - lo)).round(4).tolist()

    data = {
        "device": {"name": "R08", "gap_eV": 1.82, "structure": "p-i-n",
                   "stack": "ITO 150 nm / PTAA + Me4PACz (< 5 nm) / perovskite 270 nm / PCBM 55 nm / BCP 5 nm / Ag 100 nm",
                   "scan": "one forward scan, -0.2 to 1.35 V, 1 sun", "Voc_V": 1.299, "Jsc_mA_cm2": 14.11, "FF_pct": 78.2,
                   "PCE_pct": 14.34},
        "params": names, "log10_min": norm["log10_min"], "log10_max": norm["log10_max"],
        "J_range_mA_cm2": [j0, j1], "voltages": np.round(v, 4).tolist(), "fit_from_V": FIT_FROM, "fit_to_V": VOC,
        "measured": {"V": np.round(meas[:, 0], 4).tolist(), "J": np.round(meas[:, 2], 4).tolist()},
        "fits": fits,
        "presets": {"Decoder fit (64 starts)": to_unit(est["fit"]),
                    "Solver fit, ETL traps": to_unit({n: refit["phase3_Rsh_free"]["params"][n] for n in names})},
        "model_check": {"test_devices": int(len(x)), "param_R2": dict(zip(names, np.round(r2, 3).tolist())),
                        "decoder_MAE_mA_cm2": round(float(np.abs(xp - x).mean() * (j1 - j0)), 3),
                        "epochs": res["epochs_run"], "training_devices": norm["train"]},
        "sizes": sizes,
    }
    with open(os.path.join(OUT, "r08_data.js"), "w", encoding="utf-8") as fh:
        fh.write("window.R08 = window.R08 || {}; R08.data = " + json.dumps(data, ensure_ascii=False) + ";\n")
    print(json.dumps(data["model_check"]), sizes, [(f["label"], f["rms"]) for f in fits])


if __name__ == "__main__":
    main(*sys.argv[1:3])
