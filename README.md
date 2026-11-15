# Perovskite solar cell autoencoder: browser demo

An interactive page for the conditional autoencoder of

> O. Zbinden, E. L. Comi, E. Knapp & W. Tress, *Autoencoder for parameter estimation and current-voltage curve simulation of perovskite solar cells*, npj Computational Materials 12, 7 (2026). doi:[10.1038/s41524-025-01875-0](https://doi.org/10.1038/s41524-025-01875-0)

The network is an independent PyTorch re-implementation, trained on the authors' public data set (Zenodo, doi:[10.5281/zenodo.15639230](https://doi.org/10.5281/zenodo.15639230)). It runs entirely in the browser with ONNX Runtime Web.

- **Simulate:** six sliders set the device parameters, and the decoder draws the transient J–V curves at six scan speeds in about 30 ms.
- **Estimate:** pick a device from the paper's test set, and the encoder estimates its parameters; the decoder rebuilds its curves from the estimate.

## Run it

Download or clone the repository and open `index.html` in a browser. No server or install is needed, but the page loads two libraries (Chart.js and ONNX Runtime Web) from jsDelivr, so it needs an internet connection. It also works on GitHub Pages.

## Contents

| Path | What |
|---|---|
| `index.html` | the page |
| `model/decoder_*.js` | decoder, int8 weights (24.6 MB), base64 in 2 script files |
| `model/encoder_*.js` | encoder, float16 weights (48.1 MB), base64 in 3 script files, loaded only for the Estimate tab |
| `model/data.js` | normalisation (from the data set), parameter names and units, test-set check of the export, 12 devices from the paper's test set |
| `tools/export_onnx.py` | rebuilds the files above from a PyTorch checkpoint and the Zenodo data |

The model is stored as scripts rather than `.onnx` files because browsers block reading local files from a page opened from disk.

## Model

Two convolutional and three dense layers per half, six latent parameters, as in the paper. One training run (fold 3, early-stopping patience 200). Test-set R² of the exported model on 2,000 devices: S 0.52, μ 0.98, τ 0.87, μ_TiO₂ 0.58, μ_C 0.83, ρ_ion 0.93; mean curve error 0.45 mA/cm².

## Limits

The model covers only the simulated device of the paper (FTO / TiO₂ / m-TiO₂-MAPI / m-ZrO₂-MAPI / carbon, 1 sun, transient scans) and the parameter ranges of its Table 3. Estimates are point values without uncertainty, and different parameter sets can give very similar curves.

## Credit and licence

Method, data and parameter ranges come from the paper and its data set, both under CC BY 4.0. This repository is not affiliated with the authors.
