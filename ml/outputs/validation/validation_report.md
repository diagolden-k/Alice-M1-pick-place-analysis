# Generalization / Overfitting Validation

Question: does the selected SVM-RBF fit **only** these 48 episodes, or is there evidence it could work on new data?

## Protocols

1. **Optimistic 5-fold / Repeated K-fold / LOO** — features = global top-25 |Cohen's d| (same as original pipeline; mild selection leakage).
2. **Honest fold-wise selection** — top-25 rechosen on each training fold only.
3. **Nested CV** — outer test fold never sees C/γ tuning.
4. **Permutation test** — labels shuffled; real score must beat null.
5. **Bootstrap CI** — resample OOF predictions.
6. **Learning curve** — train vs validation gap.

## Headline results

- **optimistic_5fold_acc**: 0.9375
- **optimistic_5fold_f1**: 0.9434
- **honest_5fold_acc**: 0.7083
- **honest_5fold_f1**: 0.7500
- **honest_LOO_acc**: 0.6875
- **nested_CV_acc**: 0.8750
- **permutation_p_accuracy**: 0.0020
- **permutation_p_f1**: 0.0020
- **honest_permutation_p_accuracy**: 0.0100
- **domain10_5fold_acc**: 0.8125
- **feature_jaccard_across_folds**: 0.2596
- **learning_curve_final_gap**: 0.0539
- **repeated_kfold_acc_mean**: 0.8617
- **repeated_kfold_acc_q05**: 0.6983
- **repeated_kfold_acc_q95**: 1.0000
- **verdict**: signal_present_but_uncertain

## Interpretation

Optimistic permutation p(acc)=0.0020 (null mean≈0.52); honest-selection permutation p≈0.0100 (null mean≈0.51). Scores beat chance under label shuffle.

Optimistic 5-fold acc=0.938 vs honest fold-wise feature selection acc=0.708 (drop=0.229). Global top-25 ranking on all 48 episodes **inflated** the earlier ~94% figure (selection leakage). Feature Jaccard across folds=0.26 (top features reshuffle a lot).

Fixed 10 domain features (phase/metric terms) 5-fold acc=0.812. Nested CV (tune C/γ inside folds) acc=0.875. Honest LOO acc=0.688. Learning-curve final train–val gap=0.054.

**Bottom line (signal_present_but_uncertain):** There is a real success/fail signal, but the originally reported ~94% is optimistic. A fairer expectation on **new episodes from the same setup** is roughly **~70% accuracy** (honest CV), not 94%. Without an external holdout set we cannot claim deploy-ready generalization; re-validate on the demo/test release before trusting the score.

## Detailed tables

### Protocol comparison

                    protocol  accuracy       f1  roc_auc  feature_jaccard
           5-fold optimistic  0.937500 0.943396 0.931217              NaN
              LOO optimistic  0.895833 0.909091 0.897707              NaN
        5-fold honest select  0.708333 0.750000 0.735450         0.259645
  nested CV optimistic feats  0.875000 0.892857 0.913580              NaN
           LOO honest select  0.687500 0.716981 0.753086              NaN
        5-fold honest LogReg  0.687500 0.745763 0.701940         0.259645
5-fold fixed-10 domain feats  0.812500 0.836364 0.867725              NaN

### Repeated K-fold distribution

    name     mean      std      q05      q50  q95      min  max
accuracy 0.861667 0.101073 0.698333 0.888889  1.0 0.555556  1.0
      f1 0.876578 0.090012 0.727273 0.888889  1.0 0.666667  1.0
 roc_auc 0.907967 0.107918 0.700000 0.950000  1.0 0.500000  1.0

### Bootstrap 95% CI (optimistic OOF)

{
  "accuracy": {
    "mean": 0.9369479166666667,
    "lo": 0.8541666666666666,
    "hi": 1.0
  },
  "f1": {
    "mean": 0.9417452795152776,
    "lo": 0.8636363636363636,
    "hi": 1.0
  },
  "roc_auc": {
    "mean": 0.9318120589131351,
    "lo": 0.8254261919821693,
    "hi": 1.0
  }
}

### Permutation test

{
  "accuracy_score": 0.9355555555555555,
  "accuracy_pvalue": 0.001996007984031936,
  "accuracy_perm_mean": 0.5161644444444444,
  "f1_score": 0.9414141414141415,
  "f1_pvalue": 0.001996007984031936,
  "f1_perm_mean": 0.5552392141192142,
  "n_permutations": 500,
  "honest_foldwise": {
    "honest_accuracy": 0.7083333333333334,
    "null_mean": 0.511875,
    "null_std": 0.08407250301905426,
    "pvalue": 0.009950248756218905,
    "n_permutations": 200
  }
}

### Learning curve (final row = full data)

 train_size  train_mean  train_std  val_mean  val_std      gap
         11    1.000000   0.000000  0.611111 0.150062 0.388889
         16    1.000000   0.000000  0.808889 0.107589 0.191111
         22    1.000000   0.000000  0.768889 0.081771 0.231111
         27    1.000000   0.000000  0.871111 0.084503 0.128889
         32    0.993750   0.012500  0.891111 0.121797 0.102639
         38    0.989474   0.012892  0.935556 0.052775 0.053918