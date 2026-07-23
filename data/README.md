# Data Split Structure for Research Validation

This directory implements 4 isolated layers for rigorous research evaluation:
- `qualification/`: Qualification dataset (200-500 samples/domain) to verify expert specialization ($G_d = L_{base,d} - L_{expert,d} > 0$).
- `calibration/`: Calibration dataset (500+ samples/domain) for Fisher diagonal estimation and router calibration.
- `validation/`: Validation dataset (500-1000 samples/domain) for hyperparameter selection ($\lambda$, DARE $p$, TIES $t$, Fisher $\alpha$).
- `test/`: Isolated test dataset (1000+ samples/domain) evaluated exactly once for final benchmark comparisons.
