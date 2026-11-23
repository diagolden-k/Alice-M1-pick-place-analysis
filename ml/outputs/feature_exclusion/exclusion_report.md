# Feature exclusion test report

## Best by composite (acc + physics coverage + low redundancy)

- **physics_allowlist** n=17 SVM=0.875 LR=0.938 mean|corr|=0.26 phys_cov=1.00

### Features

- `approach_track_err_std` (|d|=1.285)
- `approach_joint_speed_std` (|d|=1.001)
- `approach_speed_maxabs` (|d|=0.824)
- `grasp_acc_mag_mean` (|d|=0.891)
- `lift_duration` (|d|=0.856)
- `carry_right_vy_mean` (|d|=1.037)
- `carry_right_vz_mean` (|d|=0.850)
- `retreat_vy_mean` (|d|=0.889)
- `cand_open_duration` (|d|=0.652)
- `cand_open_duration_over_hold` (|d|=0.652)
- `prop_retreat_burst_jspeed` (|d|=0.951)
- `prop_retreat_burst_speed` (|d|=0.902)
- `prop_tilt_pitch_release` (|d|=0.777)
- `prop_z_release` (|d|=0.541)
- `prop_sparc_palm` (|d|=0.635)
- `prop_open_dur_ratio` (|d|=0.684)
- `prop_grasp_tight_std` (|d|=0.671)

## All strategies

                     strategy  n_features  svm_acc  logreg_acc  mean_pairwise_abs_corr  physics_coverage
              corr_prune_0.70          14 0.895833    0.895833                0.276171          0.666667
            physics_allowlist          17 0.875000    0.937500                0.258315          1.000000
            physics+corr_0.85          14 0.854167    0.895833                0.261894          1.000000
                 family_cap_1          20 0.854167    0.854167                0.346057          0.833333
              corr_prune_0.85          22 0.854167    0.875000                0.319675          0.666667
             hybrid_must+corr          16 0.854167    0.875000                0.283911          0.666667
                   none_top25          25 0.833333    0.812500                0.386671          0.666667
              vif_prune_top25          18 0.833333    0.854167                0.347417          0.666667
                 family_cap_2          20 0.812500    0.833333                0.361694          0.666667
drop_global_maxabs_then_top15          15 0.791667    0.854167                0.356747          0.500000
         stability_freq>=0.55          12 0.770833    0.791667                0.365421          0.500000