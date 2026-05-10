# Validation Report (v0.7.0)

658 spots, SSI >= 65. Validated against 4 independent sources (1,935 points).

| Source | N | MAE (mag) | Bias (mag) | Correlation |
|--------|---|-----------|------------|-------------|
| Published SQM | 7 | 0.286 | -0.021 | -- |
| TESS (best-night max) | 8 | 0.268 | +0.185 | 0.99 |
| darkskysites.com grid | 1,920 | 0.473 | +0.472 | 0.937 |
| Expert detection | 20 | -- | -- | 100% within 10km |
| **Cross-source mean** | **1,935** | **0.342** | | |

## Published SQM (7 sites)

Sources: Lima et al. 2016, IDA certification 2018, Barbosa et al. 2024, Lima 2015 PhD, Dark Sky Alqueva network.

| Metric | Value |
|--------|-------|
| MAE | 0.286 mag |
| Bias | -0.021 mag |
| Max error | 0.666 mag |
| Sites within +/- 0.5 mag | 6/7 |

## TESS Network (8 stations)

Comparison against best-night maximum SQM (July 2020) from the Stars4All TESS-W network.

| Metric | Value |
|--------|-------|
| MAE (vs max) | 0.268 mag |
| Pearson r | 0.99 |

## darkskysites.com Grid (1,920 points)

Comparison against the LUMIX Garstang-class full RT engine at 0.1-degree spacing.

| Metric | Value |
|--------|-------|
| MAE | 0.473 mag |
| Bias | +0.472 mag |
| Pearson r | 0.937 |
| Points within +/- 1.0 mag | 98.4% |

The +0.47 mag offset reflects physics limitations of the simplified single-scatter PSF vs LUMIX's multiple scattering, non-isotropic emission, and CAMS aerosol ingestion.

## Expert Detection (20 spots)

| Metric | Value |
|--------|-------|
| Detected within 5 km | 70% |
| Detected within 10 km | 100% |
| Mean distance to nearest spot | 4.4 km |

## PSF Model

```
PSF(d) = (d + 1.0)^(-2.5) * exp(-0.0187 * d)
```

CALIB_FACTOR = 0.04237, fitted against 27 ground-truth points spanning SQM 18.3-21.6. Duriscoe softening (d_0 = 1km) eliminates suburban over-prediction artefacts from v0.5.0.

## SSI Weight Robustness

| Perturbation | Spearman rho vs baseline |
|--------------|--------------------------|
| Each weight +/-50% | > 0.97 |
| All-equal weights (1/6) | 0.89 |

## Flagged Spots

14 spots diverge >1 mag from the LUMIX reference (all predicting darker). Common characteristics: proximity to moderate towns (10-30km), valley locations, coastal sites with marine aerosol.

## Limitations

1. Systematic +/-0.5 mag uncertainty vs full Garstang RT
2. Only 7+8 truly independent quantitative validation points for Portugal
3. Temporal mismatch between ground-truth (2015-2024) and VIIRS (2023-2025)
4. PSF sensitive to position at 500m scale near bright sources
5. Fixed extinction coefficient (beta = 0.0187) across entire domain
6. Calibration specific to Portuguese atmospheric conditions

## References

1. Lima, R.C. et al. (2016). J. Tox. Environ. Health A, 79(7), 307-319.
2. Barbosa, D. et al. (2024). arXiv:2404.04090.
3. IDA (2018). Castle of Noudar certification report.
4. Falchi, F. et al. (2016). Science Advances 2(6), e1600377.
5. Duriscoe, D.M. et al. (2018). JQSRT 214, 133-145.

---

*Pipeline v0.7.0. Validation data: `apps/stargazing_spots/output/portugal/validation_report.json`.*
