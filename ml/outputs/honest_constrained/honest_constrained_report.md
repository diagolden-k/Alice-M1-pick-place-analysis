# Honest CV with selection constraints

- Baseline honest top25 from full pool: **0.625**
- Best constrained policy: **fixed_physics17** = **0.875**
- Fixed physics17 (pre-specified): **0.875**
- Mean fraction of physics17 inside unconstrained fold top25: **0.30**
- Mean family-clone fraction in fold top25: **0.30**

## Ranking

                    policy  accuracy       f1  mean_n_features  feature_jaccard
           fixed_physics17  0.875000 0.888889             17.0         1.000000
  honest_corr085_physics17  0.854167 0.867925             12.0         0.759560
    honest_top12_physics17  0.812500 0.830189             12.0         0.755604
  honest_family1_physics17  0.812500 0.830189             12.0         0.755604
    honest_top10_physics17  0.750000 0.785714             10.0         0.704662
honest_top15_broad_corr085  0.708333 0.766667             15.0         0.278152
 honest_top25_corr085_full  0.666667 0.724138             25.0         0.336156
      honest_top10_full267  0.625000 0.700000             10.0         0.173891
      honest_top25_full267  0.625000 0.700000             25.0         0.250091
      honest_top15_full267  0.604167 0.688525             15.0         0.161680
   honest_top15_phase_only  0.583333 0.655172             15.0         0.179843
honest_top15_physics_broad  0.583333 0.666667             15.0         0.198689
 honest_top15_family1_full  0.562500 0.644068             15.0         0.218294
