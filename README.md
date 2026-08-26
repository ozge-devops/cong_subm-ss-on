# Label Leakage in Chart-Image Financial Prediction

Measuring how much of the reported performance in chart-image financial models comes from the label being computable from the model's own input window.

**Short version:** when the anomaly label sits inside the price window the model sees, a gradient-boosted classifier on 12 hand-crafted features reaches 0.88–0.92 AUC. Move the label outside that window and the same model drops to 0.61–0.65. Across three markets and three threshold settings the gap never falls below 23 AUC points.

---

## What this repository contains

| File | Contents |
|---|---|
| `vesta_sizinti_deneyi.ipynb` | Full experiment: data download, label construction, embargo sweep, threshold sensitivity, ViT comparison, bootstrap CIs |

Everything runs top to bottom in Google Colab. No private data, no credentials, no preprocessed files — the notebook downloads its own data from Yahoo Finance and reproduces every number below.

---

## Data

Daily OHLCV, downloaded via `yfinance`, 2010-01-01 to 2026-08-24:

| Market | Ticker | Bars |
|---|---|---|
| BIST 100 | `XU100.IS` | 4167 |
| S&P 500 | `^GSPC` | 4184 |
| Nikkei 225 | `^N225` | 4068 |

Raw price data is not redistributed here. The notebook fetches it directly, so results are reproducible without us hosting anything.

**Note on granularity.** The original study described 15-minute bars. Yahoo Finance serves intraday data only for roughly the trailing 60 days, so a retrospective 2024–2025 intraday corpus cannot be assembled from it. This repository uses daily bars throughout and says so rather than describing a resolution it does not use.

---

## Task setup

An anomaly is a bar whose 5-day realised volatility exceeds the mean of its trailing 250-bar distribution by *k* standard deviations. The threshold is computed from past bars only (`shift(1)`), so the threshold itself carries no forward information.

The model sees a 60-bar window ending at *t*. The `embargo` parameter controls the gap between that window and the labelled bar:

```
embargo = 0    [-- 60-bar window --][label]     label falls inside the window
embargo = 10   [-- 60-bar window --][ gap ][label]   label sits outside
```

Because realised volatility is computed over 5 bars, an embargo of 5 or more removes the labelled bar from the feature window entirely. The performance drop is expected to sharpen around that point, and it does.

Splits are chronological (80/20). No shuffling.

---

## Results

### Effect size with 95% bootstrap confidence intervals

Numerical features (LightGBM), k = 2.0, 1000 bootstrap resamples:

| Market | Label inside | Label outside | Difference (95% CI) | Test positives |
|---|---|---|---|---|
| BIST 100 | 0.883 [0.794, 0.959] | 0.653 [0.537, 0.759] | +23.0 pts [+9.9, +37.7] | 25 |
| S&P 500 | 0.922 [0.881, 0.956] | 0.615 [0.514, 0.716] | +30.7 pts [+19.8, +40.8] | 34 |
| Nikkei 225 | 0.900 [0.837, 0.957] | 0.627 [0.536, 0.724] | +27.3 pts [+15.9, +37.9] | 34 |

None of the three difference intervals contains zero.

### Threshold sensitivity

Difference in AUC points between embargo 0 and embargo 10:

| Market | k = 1.0 | k = 1.5 | k = 2.0 |
|---|---|---|---|
| BIST 100 | 41.9 | 32.9 | 23.0 |
| S&P 500 | 30.7 | 32.4 | 30.7 |
| Nikkei 225 | 35.2 | 31.1 | 27.3 |

Mean 31.7 points, range 23.0–41.9. The effect does not depend on a particular threshold choice.

The `embargo = 0` column is the more telling one: across all nine settings it stays between 0.874 and 0.922. Whatever the market and whatever the threshold, a model whose input contains the label scores about 0.89.

### Vision Transformer

4092 candlestick charts rendered from the BIST 100 series, encoded with a frozen ViT-B/16, classified with logistic regression on the CLS embedding:

| k | ViT: label inside | ViT: label outside | ViT gap | Numerical gap |
|---|---|---|---|---|
| 1.0 | 0.679 | 0.520 | 15.8 | 41.9 |
| 1.5 | 0.670 | 0.432 | 23.7 | 32.9 |
| 2.0 | 0.740 | 0.358 | 38.2 | 23.0 |

Two observations. The leakage is present in the image modality as well, so it is not an artefact of hand-crafted features. And with the label removed, ViT lands at or below chance — on this task the image model has no residual predictive signal once leakage is closed.

Charts are rendered with `axisoff=True`. Leaving axis labels visible would let the encoder read dates and price levels off the image, which is a second leakage channel specific to the rendering step and easy to miss.

---

## Scope and limitations

Stated plainly, because they bound what the numbers support.

**Purging and embargoing are not new.** López de Prado (2018) formalised both, together with Purged K-Fold and Combinatorial Purged Cross-Validation, and they are standard practice in quantitative finance. This repository does not propose them. What it measures is the size of the effect in the chart-image subfield specifically, where those tools are not in common use.

**Positive counts are small.** Between 25 and 132 positive events per test split depending on threshold. This is why the confidence intervals are wide, and it is the main reason the individual point estimates should not be read too precisely. The direction and the ordering are stable; the exact magnitudes are not.

**Single label family.** Only volatility-threshold anomaly labels are tested. Return-direction and pattern-classification labels — both common in this literature — may behave differently.

**Daily bars only.** See the note above.

**One rendering configuration.** A single chart style, window length, and image size. Rendering choices could plausibly affect how much leakage an image encoder can exploit.

**No claim about any specific published result.** These experiments characterise a setup, not other people's papers. Determining how widely the condition appears in the literature is a separate audit, not yet done.

---

## Reproducing

Open the notebook in Colab and run all cells. GPU is only needed for the ViT section; everything else runs on CPU.

Approximate timings: data download under a minute, embargo sweep about three minutes, chart rendering eight minutes, ViT embeddings one minute on a T4, bootstrap about a minute.

Fixed seeds throughout. Reported numbers are single runs, not seed averages.

---

## References

- López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley. — purging, embargoing, and leakage-resistant cross-validation.
- Dosovitskiy, A., et al. (2021). An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale. *ICLR*. arXiv:2010.11929
- Ke, G., et al. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. *NeurIPS*.

---

## Status

Work in progress. A literature audit and the write-up are in preparation. Results here are current as of 26 August 2026.
