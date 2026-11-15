# Perovskite solar cell autoencoder: browser demo

**Live page: https://cheyneleokueh.github.io/perovskite-autoencoder-paper-simulation/**

An interactive page for the conditional autoencoder of

> O. Zbinden, E. L. Comi, E. Knapp & W. Tress, *Autoencoder for parameter estimation and current-voltage curve simulation of perovskite solar cells*, npj Computational Materials 12, 7 (2026). doi:[10.1038/s41524-025-01875-0](https://doi.org/10.1038/s41524-025-01875-0)

The network is an independent PyTorch re-implementation, trained on the authors' public data set (Zenodo, doi:[10.5281/zenodo.15639230](https://doi.org/10.5281/zenodo.15639230)). It runs in the browser with ONNX Runtime Web; nothing is sent to a server.

- **Simulate:** six sliders set the device parameters, and the decoder draws the transient J–V curves at six scan speeds, forward and reverse, in about 30 ms.
- **Estimate:** pick one of 12 devices from the paper's test set, and the encoder estimates its parameters; the decoder then rebuilds its curves from the estimate.

## Our lab cell: `r08.html`

A second page applies the same method to one real device, the wide-bandgap (1.82 eV) p-i-n cell R08: ITO / PTAA + Me4PACz / perovskite 270 nm / PCBM 55 nm / BCP 5 nm / Ag. Its one measured J–V curve (forward scan, 1 sun) comes from Prof. Fang-Chung Chen's lab (NYCU) and is shared with permission; no other lab data is included. The network was retrained on 3,598 of our own OghmaNano simulations of this cell.

- Sliders for τ, μ, ρ_ion, ETL interface traps, Rs and Rsh redraw the curve, with the live RMS against the measurement.
- Buttons load the encoder's estimate from the measured curve, the best decoder fit, and a solver fit.
- Dashed lines show full solver simulations for comparison (ETL-side traps RMS 0.10, HTL-side traps 0.07, the decoder fit re-simulated 0.56 mA/cm²).

The page states its limits: one curve cannot decide between these fits, this network has no HTL-side traps yet, and the decoder's own error (0.67 mA/cm² on average) is larger than the differences between the fits. Files: `model/r08_*.js`, rebuilt by `tools/export_r08.py`.

## Run it

Open the live page, or download the repository and open `index.html` in a browser. No server or install is needed, but the page loads two libraries (Chart.js and ONNX Runtime Web) from jsDelivr, so it needs an internet connection.

Both networks start downloading as soon as the page opens, with a progress bar for each above the chart. The decoder is about 24 MB and the encoder about 48 MB on the wire (GitHub Pages compresses the base64 back to roughly the ONNX size). The sliders work once the decoder is ready, and the Estimate tab once both are.

## Contents

| Path | What |
|---|---|
| `index.html` | the paper-model page |
| `r08.html`, `model/r08_*.js` | our lab cell page and its float16 model (9 MB each half) |
| `style.css` | shared styles |
| `model/decoder_*.js` | decoder, int8 weights: 24.6 MB as ONNX, 32.8 MB as base64 in 2 script files |
| `model/encoder_*.js` | encoder, float16 weights: 48.1 MB as ONNX, 64.2 MB as base64 in 3 script files |
| `model/data.js` | normalisation (from the data set), parameter names and units, the 12 test devices, the test-set check of the export, and file sizes for the progress bars |
| `tools/export_onnx.py` | rebuilds the paper-model files from a PyTorch checkpoint and the Zenodo data |
| `tools/export_r08.py` | rebuilds the R08 files from our training run, fits and the measured curve |

The model is stored as script files rather than `.onnx` files because browsers block a page opened from disk from reading other local files. Each file stays under 30 MB.

To rebuild `model/`, install `torch`, `numpy`, `onnx`, `onnxruntime` and `onnxconverter-common`, then run `python tools/export_onnx.py <checkpoint_dir> <zenodo_dir>`. The checkpoint folder holds `encoder.pt` and `decoder.pt` state dicts.

## Model

The architecture follows the paper and the authors' published code: two convolutional layers, dense layers of 512, 64 and 32 units, and a 6-value latent layer, with the decoder mirroring the encoder. This is one training run (one of ten validation splits, early-stopping patience 200).

Weights are compressed for the web. The decoder uses int8, which raises the mean curve error only from 0.42 to 0.43 mA/cm². The encoder uses float16, because int8 lowered its R² by up to 0.12.

Test-set check of the exported model on 2,000 devices: R² S 0.52, μ 0.98, τ 0.87, μ_TiO₂ 0.58, μ_C 0.83, ρ_ion 0.93; mean curve error 0.45 mA/cm².

## Limits

The model covers only the simulated device of the paper (FTO / TiO₂ / m-TiO₂-MAPI / m-ZrO₂-MAPI / carbon, 1 sun, transient scans) and the parameter ranges of its Table 3. Estimates are point values without uncertainty, and different parameter sets can give very similar curves.

## Credit

Method, data and parameter ranges come from the paper and its data set, both under CC BY 4.0. This repository is not affiliated with the authors. The repository's own code has no licence file yet.
