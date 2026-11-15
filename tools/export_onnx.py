"""Export a trained PyTorch port of the paper's conditional autoencoder to ONNX for the browser demo.

Usage: python tools/export_onnx.py <checkpoint_dir> <zenodo_dir>
  checkpoint_dir  holds encoder.pt and decoder.pt (state dicts)
  zenodo_dir      the paper's data set (x_test.npy, y_test.npy, current_df_info.csv, log_param_df_info.csv)

Writes model/encoder.onnx, model/decoder.onnx (int8 weights; encoder float16), model/meta.json and model/examples.json.
"""
import csv
import json
import os
import sys

import numpy as np
import torch
from torch import nn

L = 732  # 6 scan speeds x (0 -> 1.2 -> 0 V in 0.02 V steps)
N_PAR = 6
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model")


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv1d(1, 128, 7, padding=3), nn.LeakyReLU(0.2),
                                  nn.Conv1d(128, 64, 5, padding=2), nn.LeakyReLU(0.2))
        self.dense = nn.Sequential(nn.Flatten(), nn.Linear(64 * L, 512), nn.ReLU(), nn.Linear(512, 64), nn.ReLU(),
                                   nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, N_PAR), nn.ReLU())

    def forward(self, x):
        return self.dense(self.conv(x.unsqueeze(1)))


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense = nn.Sequential(nn.Linear(N_PAR, 32), nn.ReLU(), nn.Linear(32, 64), nn.ReLU(),
                                   nn.Linear(64, 512), nn.ReLU(), nn.Linear(512, 64 * L), nn.ReLU())
        self.deconv = nn.Sequential(nn.ConvTranspose1d(64, 128, 5, padding=2), nn.LeakyReLU(0.2),
                                    nn.ConvTranspose1d(128, 1, 7, padding=3), nn.LeakyReLU(0.2))

    def forward(self, y):
        return self.deconv(self.dense(y).view(-1, 64, L)).squeeze(1)


def main(ckpt, zenodo):
    from onnxruntime.quantization import QuantType, quantize_dynamic
    import onnx
    import onnxruntime as ort
    from onnxconverter_common import float16

    os.makedirs(OUT, exist_ok=True)
    nets = {"encoder": (Encoder(), torch.zeros(1, L), "x", "y"), "decoder": (Decoder(), torch.zeros(1, N_PAR), "y", "x")}
    for name, (net, dummy, i, o) in nets.items():
        net.load_state_dict(torch.load(os.path.join(ckpt, f"{name}.pt"), map_location="cpu"))
        net.eval()
        fp32 = os.path.join(OUT, f"{name}_fp32.onnx")
        torch.onnx.export(net, dummy, fp32, input_names=[i], output_names=[o], dynamic_axes={i: {0: "n"}, o: {0: "n"}},
                          opset_version=17, dynamo=False)
        if name == "decoder":  # int8 weights: curve error barely changes (0.42 -> 0.43 mA/cm2 mean absolute error)
            quantize_dynamic(fp32, os.path.join(OUT, f"{name}.onnx"), weight_type=QuantType.QInt8, per_channel=True)
        else:  # int8 costs the encoder up to 0.1 in R2, so float16 instead (no measurable loss)
            onnx.save(float16.convert_float_to_float16(onnx.load(fp32), keep_io_types=True), os.path.join(OUT, f"{name}.onnx"))
        os.remove(fp32)

    # normalisation, exactly as the authors' decoder_for_data_generation.py
    def rows(name):
        with open(os.path.join(zenodo, name)) as fh:
            return [r for r in csv.DictReader(fh) if r["min"]]
    jn = rows("current_df_info.csv")[-1]
    pn = rows("log_param_df_info.csv")[2:]
    meta = {
        "source": "Zbinden, Comi, Knapp & Tress, npj Comput. Mater. 12, 7 (2026), doi:10.1038/s41524-025-01875-0; "
                  "data: Zenodo doi:10.5281/zenodo.15639230 (CC BY 4.0)",
        "J_min": float(jn["min"]), "J_max": float(jn["max"]),
        "params": [{"key": k, "name": n, "unit": u, "log10_min": float(lo), "log10_max": float(hi)}
                   for (k, n, u), lo, hi in zip(
                       [("S", "Surface recombination velocity S", "cm/s"), ("mu", "Perovskite mobility μ", "cm²/Vs"),
                        ("tau", "SRH lifetime τ", "ns"), ("mu_TiO2", "TiO₂ electron mobility μ_TiO₂", "cm²/Vs"),
                        ("mu_C", "Cation mobility μ_C", "cm²/Vs"), ("rho_ion", "Ion density ρ_ion", "cm⁻³")],
                       [r["min"] for r in pn], [r["max"] for r in pn])],
        "scan_speeds_V_s": [100, 50, 5, 0.5, 0.05, 0.005],
        "voltage": "0 -> 1.2 -> 0 V in 0.02 V steps, 122 points per speed",
    }

    # a few test-set devices (never used in training) for the estimate tab, plus an accuracy check of the int8 export
    x = np.load(os.path.join(zenodo, "x_test.npy")).astype(np.float32)
    y = np.load(os.path.join(zenodo, "y_test.npy")).astype(np.float32)
    rng = np.random.default_rng(7)
    pick = rng.choice(len(x), 12, replace=False)
    ex = [{"id": int(k), "x": np.round(x[k], 5).tolist(), "y": np.round(y[k], 5).tolist()} for k in pick]
    json.dump(ex, open(os.path.join(OUT, "examples.json"), "w"))

    enc = ort.InferenceSession(os.path.join(OUT, "encoder.onnx"))
    dec = ort.InferenceSession(os.path.join(OUT, "decoder.onnx"))
    sub = rng.choice(len(x), 2000, replace=False)
    yp = enc.run(None, {"x": x[sub]})[0]
    xp = dec.run(None, {"y": y[sub]})[0]
    r2 = 1 - ((y[sub] - yp) ** 2).sum(0) / ((y[sub] - y[sub].mean(0)) ** 2).sum(0)
    meta["check_export_on_2000_test_devices"] = {
        "param_R2": dict(zip([p["key"] for p in meta["params"]], np.round(r2, 3).tolist())),
        "decoder_MAE_mA_cm2": round(float(np.abs(xp - x[sub]).mean() * (meta["J_max"] - meta["J_min"])), 3)}
    json.dump(meta, open(os.path.join(OUT, "meta.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(meta["check_export_on_2000_test_devices"]))
    for f in sorted(os.listdir(OUT)):
        print(f, round(os.path.getsize(os.path.join(OUT, f)) / 1e6, 1), "MB")


if __name__ == "__main__":
    main(*sys.argv[1:3])
