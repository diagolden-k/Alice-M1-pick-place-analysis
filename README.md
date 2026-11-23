# Alice M1 Pick-and-Place Analysis (IC-PBL)

한양대 ERICA **MMT3007 Machine Learning** IC-PBL  
휴머노이드 **Alice M1**의 pick-and-place(병을 집어 박스에 넣기) CSV 궤적을 단계 분할·운동학 피처·영향도 분석까지 정리한 저장소입니다.

산업 파트너: Aei Robot

## 과제 요약

- **데이터**: 48 episodes (`success` 27 / `fail` 21), 30 fps CSV
- **입력**: `state.*` / `action.*` 각 26관절 (양팔 7-DoF + 손 6-DoF)
- **목표**: 전처리 → 분류 → 클러스터링 → 품질 메트릭 (본 저장소는 1차: 위상·피처·영향 분석)

## 폴더 구조

```text
Alice/
├── IC-PBL Project Description.pdf   # 과제 설명
├── ALICE_M1_ROBOT_INFO.md           # URDF 기반 로봇 특성 정리
├── alice_m1_urdf/                   # Alice M1 URDF + meshes
├── 26_2_machinelearning_1st_data/   # episode CSV (success/fail)
├── 26_2_machinelearning_1st_video/  # 참고용 left_view mp4
└── viz/                             # FK · 위상 · 피처 · 분석 · 시각화
    ├── alice_arm_fk.py
    ├── phases.py
    ├── visualize_arm_motion.py
    ├── plot_palm_path.py
    ├── build_phase_features.py
    ├── analyze_feature_influence.py
    ├── visualize_analysis.py
    └── outputs/                     # CSV 결과 + PNG 그림
```

## 동작 단계 (phase)

왼팔 기준 torso frame (`+X` 앞, `+Y` 왼쪽, `+Z` 위):

1. **approach** — 왼쪽에서 병으로 접근  
2. **grasp** — 손 닫음  
3. **lift** — 상승  
4. **carry_right** — 오른쪽으로 이동  
5. **place** — 하강하며 놓음  
6. **retreat** — 손 펴고 왼쪽으로 치움  

## 빠른 실행

```bash
pip install numpy pandas matplotlib scipy scikit-learn pillow

# 단일 에피소드 3D 스틱 / 손바닥 경로
python viz/visualize_arm_motion.py --episode 2
python viz/plot_palm_path.py --episode 2

# 전 에피소드 피처 생성 → 영향 분석 → 시각화
python viz/build_phase_features.py
python viz/analyze_feature_influence.py
python viz/visualize_analysis.py
```

## 주요 산출물 (`viz/outputs/`)

| 파일 | 설명 |
|---|---|
| `frame_phase_kinematics.csv` | 프레임별 phase + vel/acc/jerk |
| `episode_phase_features.csv` | 에피소드 집약 피처 (269열) |
| `feature_influence_ranking.csv` | Cohen's d, point-biserial, FDR 등 |
| `logistic_feature_coefs.csv` | 표준화 로지스틱 계수 |
| `feature_influence_report.md` | 분석 리포트 |
| `viz_*.png` | 위상·운동학·효과크기 시각화 |

## 분석 한줄 요약 (피처 영향)

- 단계별 평균 효과크기: **grasp > carry_right > retreat > …**
- 실패와 연관: 접근 시 tracking 흔들림·과속 (`approach_track_err_std`, `approach_*speed*`)
- 성공과 연관: 또렷한 grasp 가속/jerk, retreat 시 왼쪽 치우기 (`retreat_vy_mean`)

## ML 파이프라인 (Classification / Clustering / Metric)

입력: `|Cohen's d|` 상위 25개 에피소드 피처. N=48이므로 딥시퀀스 대신 집약 피처 + 고전 ML.

```bash
python ml/run_pipeline.py
python ml/predict_episode.py --episode 2
```

### 선정 결과

| 과제 | 선정 | 성능 / 근거 |
|---|---|---|
| **Classification** | `SVM-RBF` (`C=3`, `gamma=scale`) | OOF F1 **0.943**, Acc **0.938** (logreg/RF/GB/kNN 등 대비 최고) |
| **Clustering** | `GMM` **k=4** (PCA-8 임베딩) | 균형·해석성 반영 선정; success/fail보다 세분 |
| **Quality Metric** | 해석 가능 가중합 + 0.25·P(success) | label 상관 **0.90**, success−fail gap **≈39점** |

### 클러스터 이름

- `CleanSuccess_FirmGrasp` — 또렷한 grasp, 짧은 lift (성공률 ≈87%)
- `CleanSuccess_SmoothApproach` — 접근 tracking/속도 안정 (성공률 ≈81%)
- `Fail_RoughApproach` — 접근 과속·불안정 (성공률 ≈10%)
- `Fail_SlowLift_RoughApproach` — 거친 접근 + 긴 lift (성공률 0%)

산출물: `ml/outputs/` (`ml_pipeline_report.md`, `episode_results.csv`, 모델 joblib, 그림)

## 좌표계 참고

CSV에 허리/베이스 관절이 없어 FK는 **torso_link 기준**입니다. 로봇 상세는 `ALICE_M1_ROBOT_INFO.md` 참고.
