# Feature influence analysis (success=1 vs fail=0)

## Methods

- **Cohen's d**: standardized mean difference `(μ_success − μ_fail) / s_pooled`.
- **Point-biserial r**: Pearson correlation between feature and binary label.
- **Welch t-test** + Benjamini–Hochberg FDR on p-values.
- **Mann–Whitney U** + rank-biserial correlation.
- **Mutual information** (feature quantile-binned vs label).
- **L2 logistic regression** on standardized top-|d| features; `odds_ratio_per_std = exp(β)` for +1σ change.

## Phase-group effect summary

| phase_group | n_features | mean_abs_d | max_abs_d | mean_abs_r | n_sig_fdr05 |
| --- | --- | --- | --- | --- | --- |
| grasp | 38 | 0.5237 | 0.8907 | 0.2515 | 0 |
| carry_right | 38 | 0.4798 | 1.0373 | 0.2307 | 0 |
| retreat | 38 | 0.3896 | 0.8886 | 0.1904 | 0 |
| global | 39 | 0.3714 | 0.8363 | 0.1818 | 0 |
| approach | 38 | 0.3175 | 1.2847 | 0.1526 | 0 |
| place | 38 | 0.3171 | 0.5993 | 0.1572 | 0 |
| lift | 38 | 0.2911 | 0.8560 | 0.1428 | 0 |

## Top 25 features by |Cohen's d|

| feature | mean_success | mean_fail | cohens_d | point_biserial_r | welch_p_fdr | rank_biserial_r | mutual_info_bits |
| --- | --- | --- | --- | --- | --- | --- | --- |
| approach_track_err_std | 0.0310 | 0.0379 | -1.2847 | -0.5456 | 0.0545 | 0.6261 | 0.2198 |
| carry_right_vy_mean | -0.0547 | -0.0266 | -1.0373 | -0.4653 | 0.0795 | 0.5414 | 0.1273 |
| approach_joint_speed_std | 0.1114 | 0.1430 | -1.0014 | -0.4525 | 0.1201 | 0.4921 | 0.1208 |
| approach_joint_speed_maxabs | 0.6679 | 0.8159 | -0.8976 | -0.4140 | 0.1201 | 0.4250 | 0.0885 |
| grasp_acc_mag_mean | 0.9828 | 0.7097 | 0.8907 | 0.4114 | 0.1201 | -0.4780 | 0.0813 |
| carry_right_track_err_maxabs | 0.2127 | 0.1511 | 0.8903 | 0.4112 | 0.1201 | -0.5026 | 0.1194 |
| retreat_vy_mean | 0.0976 | 0.0837 | 0.8886 | 0.4106 | 0.1201 | -0.5026 | 0.1515 |
| grasp_acc_mag_rms | 1.1061 | 0.8036 | 0.8660 | 0.4019 | 0.1201 | -0.4921 | 0.1194 |
| lift_duration | 1.7025 | 2.3556 | -0.8560 | -0.3980 | 0.1201 | 0.4656 | 0.1654 |
| carry_right_track_err_rms | 0.1635 | 0.1157 | 0.8534 | 0.3969 | 0.1201 | -0.5026 | 0.1246 |
| carry_right_vz_mean | -0.0382 | -0.0557 | 0.8501 | 0.3956 | 0.1298 | -0.5026 | 0.1339 |
| carry_right_track_err_std | 0.0353 | 0.0209 | 0.8487 | 0.3951 | 0.1201 | -0.5414 | 0.2557 |
| global_track_err_std | 0.0664 | 0.0580 | 0.8363 | 0.3902 | 0.1201 | -0.4286 | 0.1288 |
| carry_right_track_err_mean | 0.1589 | 0.1134 | 0.8261 | 0.3862 | 0.1201 | -0.4780 | 0.1194 |
| approach_speed_maxabs | 0.1455 | 0.1761 | -0.8241 | -0.3854 | 0.1404 | 0.4780 | 0.0885 |
| grasp_speed_rms | 0.0895 | 0.0673 | 0.8096 | 0.3796 | 0.1201 | -0.4392 | 0.0883 |
| grasp_jerk_mag_rms | 23.5484 | 17.4279 | 0.7963 | 0.3742 | 0.1282 | -0.4427 | 0.1344 |
| grasp_jerk_mag_mean | 20.9274 | 15.3723 | 0.7950 | 0.3737 | 0.1280 | -0.4321 | 0.1344 |
| grasp_joint_acc_mean | 5.8786 | 4.6711 | 0.7945 | 0.3735 | 0.1201 | -0.3898 | 0.0483 |
| grasp_speed_mean | 0.0810 | 0.0618 | 0.7900 | 0.3717 | 0.1201 | -0.4109 | 0.0749 |
| lift_closure_mean | 1.4038 | 1.4283 | -0.7835 | -0.3690 | 0.1298 | 0.5097 | 0.1913 |
| lift_closure_rms | 1.4042 | 1.4285 | -0.7789 | -0.3671 | 0.1298 | 0.5097 | 0.1913 |
| lift_closure_std | 0.0295 | 0.0208 | 0.7719 | 0.3643 | 0.1201 | -0.4145 | 0.0943 |
| grasp_speed_maxabs | 0.1488 | 0.1117 | 0.7630 | 0.3606 | 0.1222 | -0.4356 | 0.1220 |
| grasp_acc_mag_maxabs | 1.9666 | 1.4719 | 0.7450 | 0.3532 | 0.1404 | -0.5379 | 0.1309 |

## Logistic coefficients (standardized features)

CV accuracy: 0.849 ± 0.115

| feature | logit_coef_std | odds_ratio_per_std |
| --- | --- | --- |
| lift_closure_std | 0.9632 | 2.6201 |
| approach_track_err_std | -0.9631 | 0.3817 |
| approach_speed_maxabs | -0.8402 | 0.4316 |
| retreat_vy_mean | 0.7951 | 2.2146 |
| carry_right_vz_mean | 0.7549 | 2.1274 |
| lift_duration | -0.6383 | 0.5282 |
| grasp_joint_acc_mean | -0.4878 | 0.6140 |
| carry_right_vy_mean | -0.4340 | 0.6479 |
| approach_joint_speed_maxabs | -0.4183 | 0.6582 |
| grasp_speed_maxabs | 0.4165 | 1.5167 |
| carry_right_track_err_mean | 0.4011 | 1.4935 |
| grasp_acc_mag_mean | 0.3845 | 1.4688 |
| carry_right_track_err_rms | 0.3695 | 1.4470 |
| approach_joint_speed_std | -0.3224 | 0.7244 |
| carry_right_track_err_maxabs | 0.2557 | 1.2914 |
| global_track_err_std | -0.2469 | 0.7812 |
| grasp_acc_mag_rms | 0.2164 | 1.2417 |
| grasp_jerk_mag_rms | -0.1226 | 0.8846 |
| grasp_acc_mag_maxabs | 0.0803 | 1.0836 |
| carry_right_track_err_std | 0.0754 | 1.0783 |
| grasp_speed_mean | -0.0686 | 0.9337 |
| grasp_speed_rms | 0.0661 | 1.0684 |
| lift_closure_mean | -0.0126 | 0.9875 |
| grasp_jerk_mag_mean | -0.0058 | 0.9942 |
| lift_closure_rms | -0.0022 | 0.9978 |

Interpretation: `logit_coef_std > 0` increases log-odds of **success** when the feature rises by 1 standard deviation (other top features held constant).


## Figures

- `viz_phase_duration_by_label.png`
- `viz_kinematics_boxplot_by_phase.png`
- `viz_episode_*_timeline.png`
- `viz_all_palm_paths.png`
- `viz_feature_cohens_d_top20.png`
- `viz_phase_group_influence.png`
- `viz_logistic_coefs.png`
- `viz_top2_feature_scatter.png`
