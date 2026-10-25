# Report: sensitivity

3840 runs on 32 datasets. Error = 1 - balanced accuracy (classification) or RMSE (regression); relative reduction = (e_ref - e) / max(e_ref, e) of errors averaged over a dataset's runs; positive favours the method.

## Sensitivity: each variant vs base under the same setting (descriptive)
Base runs from the same check, the check named in `base_from`, or the reference experiment.

| check | method | base_from | datasets | median_reduction | wins | losses | zeros |
|---|---|---|---|---|---|---|---|
| k_half | random_xx | main | 32 | -0.0045 | 14 | 18 | 0 |
| k_half | residual_xx | main | 32 | -0.0046 | 11 | 21 | 0 |
| k_half | target_xx | main | 32 | -0.0071 | 11 | 21 | 0 |
| k_double | random_xx | main | 32 | -0.008 | 8 | 23 | 1 |
| k_double | residual_xx | main | 32 | -0.0104 | 12 | 20 | 0 |
| k_double | target_xx | main | 32 | -0.0121 | 10 | 22 | 0 |
| forests_4x4 | monotone | forests_4x4 | 32 | 0 | 8 | 13 | 11 |
| forests_4x4 | nonlinear | forests_4x4 | 32 | 0 | 11 | 11 | 10 |
| forests_4x4 | residual_xx | forests_4x4 | 32 | -0.0082 | 11 | 21 | 0 |
| trees_500 | monotone | trees_500 | 32 | -0.001 | 7 | 17 | 8 |
| trees_500 | nonlinear | trees_500 | 32 | 0 | 10 | 14 | 8 |
| trees_500 | residual_xx | trees_500 | 32 | -0.0067 | 12 | 20 | 0 |
| no_binning | monotone | no_binning | 32 | 0 | 10 | 15 | 7 |
| no_binning | nonlinear | no_binning | 32 | -0.0002 | 9 | 17 | 6 |
| no_binning | residual_xx | no_binning | 32 | -0.0059 | 14 | 18 | 0 |
| no_binning_relu | monotone | no_binning | 32 | -0.000141 | 10 | 16 | 6 |
| no_binning_gelu | monotone | no_binning | 32 | -0.000597 | 7 | 18 | 7 |
| oof_5 | monotone | oof_5 | 32 | 0 | 12 | 15 | 5 |
| oof_5 | nonlinear | oof_5 | 32 | 0 | 13 | 15 | 4 |
| oof_5 | residual_xx | oof_5 | 32 | 0.0012 | 17 | 15 | 0 |

## Sensitivity: each check vs `main` (descriptive)
Same method, dataset, seed and fold in the reference experiment.

| check | method | datasets | median_reduction | wins | losses |
|---|---|---|---|---|---|
| k_half | random_xx | 32 | 0.0059 | 20 | 11 |
| k_half | residual_xx | 32 | 0.0027 | 19 | 12 |
| k_half | target_xx | 32 | -0.0017 | 15 | 16 |
| k_double | random_xx | 32 | -0.0016 | 15 | 17 |
| k_double | residual_xx | 32 | -0.0026 | 13 | 19 |
| k_double | target_xx | 32 | -0.0043 | 9 | 23 |
| forests_4x4 | base | 32 | -3.56e-06 | 16 | 16 |
| forests_4x4 | monotone | 32 | -3.56e-06 | 16 | 16 |
| forests_4x4 | nonlinear | 32 | -0.000378 | 15 | 17 |
| forests_4x4 | residual_xx | 32 | 0.000134 | 16 | 16 |
| trees_500 | base | 32 | 0.000302 | 17 | 15 |
| trees_500 | monotone | 32 | 0.000178 | 17 | 15 |
| trees_500 | nonlinear | 32 | 0.000482 | 19 | 13 |
| trees_500 | residual_xx | 32 | 0.001 | 19 | 13 |
| no_binning | base | 32 | 0.0011 | 18 | 13 |
| no_binning | monotone | 32 | 0.000128 | 16 | 15 |
| no_binning | nonlinear | 32 | 0.000143 | 16 | 15 |
| no_binning | residual_xx | 32 | 0.0028 | 19 | 13 |
| no_binning_relu | monotone | 32 | 0.000594 | 18 | 13 |
| no_binning_gelu | monotone | 32 | 0.000539 | 19 | 12 |
| oof_5 | base | 32 | 0.0065 | 21 | 11 |
| oof_5 | monotone | 32 | 0.0077 | 24 | 8 |
| oof_5 | nonlinear | 32 | 0.0105 | 22 | 10 |
| oof_5 | residual_xx | 32 | 0.0118 | 27 | 5 |
