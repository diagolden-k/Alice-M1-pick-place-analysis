# Alice M1 로봇 — URDF 분석 정리

## 1. 분석 기준

- 원본: [alice_m1.urdf](alice_m1.urdf), 로봇 이름: `alice_m1`.
- 이 문서는 해당 파일의 XML 정의와 그로부터 계산한 값만 정리한다. 실물 제품의 공식 사양과 다를 수 있다.
- 길이·위치는 m, 질량은 kg, 관성은 kg·m², 각도는 rad, 각속도는 rad/s, 회전 관절 effort는 N·m 기준이다. 표의 도(°) 값은 rad에서 변환하여 반올림했다.
- `origin xyz/rpy`는 부모 링크에 대한 관절 좌표계의 변환이며, `axis`는 관절 좌표계 기준이다. RPY는 roll·pitch·yaw 순서다.

## 2. 전체 구성과 자유도

Alice M1은 바퀴형 이동 베이스, 다단 몸통, 양팔, 양손, 머리 및 카메라·IMU 프레임으로 구성되어 있다. 각 팔은 어깨 3축, 팔꿈치 1축, 손목 3축의 7자유도 구조이며 각 손에는 다섯 손가락이 있다.

| 항목 | URDF 기준 값 |
|---|---:|
| 링크 | 77개 |
| 전체 관절 | 76개 |
| `revolute` — 범위가 제한된 회전 관절 | 43개 |
| `continuous` — 연속 회전 관절 | 2개 |
| `fixed` — 고정 관절 | 31개 |
| mimic 종속 관절 | 12개 |
| 독립 관절 변수 | 33개 = 43 + 2 − 12 |
| 선언된 링크 질량 합계 | 115.4414 kg |
| 루트 링크 | `base_footprint` |

독립 관절 변수 33개는 양팔 14개, 양손 12개, 몸통 3개, 머리 2개, 바퀴 2개로 나뉜다. 바퀴를 제외한 독립 회전 관절은 31개다. 이는 URDF의 기구학적 변수 수이며 실제 모터 수나 제어 인터페이스 수를 보증하지 않는다. 베이스의 월드 좌표 이동을 나타내는 floating/planar 관절은 정의되어 있지 않다.

## 3. 주요 연결 구조

아래는 핵심 연결을 요약한 트리다. 화살표의 괄호는 관절 이름이다. 손가락·센서의 세부 연결은 뒤의 전체 관절 표에 수록했다.

```text
base_footprint
└─ (base_joint, fixed) base_link
   ├─ (left_wheel_joint) left_wheel_link
   ├─ (right_wheel_joint) right_wheel_link
   ├─ (imu_fixed) imu_link
   └─ (waist_lower_pitch) lower_torso_link
      └─ (lower_torso_mimic) lower_torso_mimic_link
         └─ (upper_torso_mimic) upper_torso_mimic_link
            └─ (waist_upper_pitch) upper_torso_link
               └─ (waist_y) torso_link
                  ├─ (l_sh_p → l_sh_r → l_sh_y → l_el_p
                  │    → l_wr_y → l_wr_p → l_wr_r) left_arm_link_7
                  │  └─ (l_hand, fixed) left_base_link → 왼손 다섯 손가락
                  ├─ (r_sh_p → r_sh_r → r_sh_y → r_el_p
                  │    → r_wr_y → r_wr_p → r_wr_r) right_arm_link_7
                  │  └─ (r_hand, fixed) right_base_link → 오른손 다섯 손가락
                  └─ (head_y) head_yaw
                     └─ (head_p) head_pitch
                        └─ (zed_cam) zed_cam_link → 카메라·IMU 프레임
```

## 4. 팔과 머리·몸통 관절 사양

이름의 `l/r`은 좌/우, `sh/el/wr`은 어깨/팔꿈치/손목, `p/r/y`는 pitch/roll/yaw를 뜻한다. 두 팔의 어깨 roll 및 손목 roll 범위는 좌우가 반대다.

원본 주석은 팔·머리·독립 몸통 관절의 최대 토크를 **피크 토크** 기준이라고 명시한다. 따라서 아래 effort를 연속 정격 토크로 해석하지 않는다.

| 관절 | 축 (x y z) | 하한~상한 (°) | effort (N·m) | 속도 (rad/s) |
|---|---|---:|---:|---:|
| `l_sh_p` | `0 1 0` | -180 ~ 180 | 120 | 13.3 |
| `l_sh_r` | `1 0 0` | -15 ~ 180 | 120 | 13.3 |
| `l_sh_y` | `0 0 1` | -80 ~ 80 | 60 | 16.02 |
| `l_el_p` | `0 1 0` | -120 ~ 0 | 60 | 16.02 |
| `l_wr_y` | `0 0 1` | -80 ~ 80 | 34 | 8.69 |
| `l_wr_p` | `0 1 0` | -40 ~ 40 | 34 | 8.69 |
| `l_wr_r` | `1 0 0` | -90 ~ 20 | 34 | 8.69 |
| `r_sh_p` | `0 1 0` | -180 ~ 180 | 120 | 13.3 |
| `r_sh_r` | `1 0 0` | -180 ~ 15 | 120 | 13.3 |
| `r_sh_y` | `0 0 1` | -80 ~ 80 | 60 | 16.02 |
| `r_el_p` | `0 1 0` | -120 ~ 0 | 60 | 16.02 |
| `r_wr_y` | `0 0 1` | -80 ~ 80 | 34 | 8.69 |
| `r_wr_p` | `0 1 0` | -40 ~ 40 | 34 | 8.69 |
| `r_wr_r` | `1 0 0` | -20 ~ 90 | 34 | 8.69 |
| `head_y` | `0 0 1` | -45 ~ 45 | 10 | 24.92 |
| `head_p` | `0 1 0` | -50 ~ 50 | 10 | 24.92 |
| `waist_y` | `0 0 1` | -45 ~ 45 | 120 | 13.3 |
| `waist_upper_pitch` | `0 1 0` | -85 ~ 0 | 320 | 10.47 |
| `upper_torso_mimic` | `0 1 0` | 0 ~ 85 | 1000 | 10.47 |
| `lower_torso_mimic` | `0 1 0` | 0 ~ 85 | 1000 | 10.47 |
| `waist_lower_pitch` | `0 1 0` | -85 ~ 0 | 320 | 10.47 |

### 팔의 주요 치수

- 어깨 관절 위치는 `torso_link` 기준 왼쪽 `(0, 0.24, 0)`, 오른쪽 `(0, −0.24, 0)`이며, 두 어깨 원점 간 거리는 0.48 m다.
- 어깨 yaw → 팔꿈치 pitch 원점 간 거리는 0.26 m, 팔꿈치 pitch → 손목 yaw는 0.255 m다.
- 손목 roll → 손바닥 베이스 고정 연결은 `(0, 0, −0.077)` m이며 RPY는 왼손 `(0, 3.1416, 1.5708)`, 오른손 `(0, 3.1416, −1.5708)` rad다.
- 위 값은 관절 원점 사이 거리다. 손끝까지의 작업 반경, 충돌 없는 가동 범위, 가반하중을 직접 의미하지 않는다.

### 몸통의 mimic 관계

```text
q(lower_torso_mimic) = −q(waist_lower_pitch)
q(upper_torso_mimic) = −q(waist_upper_pitch)
```

두 mimic 관절은 각각 대응 pitch 관절을 반대 방향으로 따라간다. 네 pitch 관절 축이 모두 Y축이고 중간 고정 RPY가 0이므로, mimic 관계를 적용하면 이 pitch 체인의 누적 회전은 상쇄된다. 몸통 위치는 달라질 수 있으며, 상부의 `waist_y`는 별도로 yaw 회전을 제공한다. mimic 관절에 적힌 effort 1000은 별도의 독립 구동기 정격으로 볼 수 없다.

## 5. 손 구조

원본 주석에는 `brainco hand version`, 메시 경로에는 `revo2_left_hand` 및 `revo2_right_hand`라는 이름이 사용된다. 이 명칭만으로 실물 손의 정확한 제품 사양을 확정할 수는 없다.

각 손에는 회전 관절 11개가 있으며, 이 중 독립 관절은 6개, mimic 관절은 5개다. 엄지는 metacarpal·proximal·distal 3관절, 나머지 네 손가락은 proximal·distal 각 2관절 구조다. 각 손가락에는 고정된 `tip_link`와 `touch_link`도 있다.

아래 표는 양손 공통 한계다. 실제 관절 이름에는 `left_` 또는 `right_` 접두사가 붙는다.

| 관절 종류 | 범위 (rad) | 범위 (°, 약) | effort (N·m) | 속도 (rad/s) | 종속 관계 |
|---|---:|---:|---:|---:|---|
| thumb_metacarpal | 0 ~ 1.57 | 0 ~ 89.954 | 0.5 | 2.6175 | 독립 |
| thumb_proximal | 0 ~ 1.03 | 0 ~ 59.0147 | 1.1 | 2.5303 | 독립 |
| thumb_distal | 0 ~ 1.03 | 0 ~ 59.0147 | 1.1 | 2.5303 | 같은 손 thumb_proximal × 1 |
| index/middle/ring/pinky proximal | 0 ~ 1.41 | 0 ~ 80.786 | 2 | 2.2685 | 각각 독립 |
| index/middle/ring/pinky distal | 0 ~ 1.63 | 0 ~ 93.3921 | 2 | 2.2685 | 같은 손가락 proximal × 1.155 |

모든 손가락 mimic offset은 0이다. 네 손가락은 proximal 최대값에서 distal이 `1.41 × 1.155 = 1.62855 rad`로 distal 상한 1.63 rad 안에 들어간다. 엄지 축 방향은 좌우가 반대이며 전체 축·원점은 부록을 따른다. `touch_link`라는 이름만으로 촉각 센서의 종류나 출력이 정의되는 것은 아니다.

## 6. 이동 베이스와 바퀴

| 항목 | 파일에 정의된 값 |
|---|---|
| 바퀴 관절 | `left_wheel_joint`, `right_wheel_joint` — continuous |
| 회전축 | 양쪽 모두 `(0, 1, 0)` |
| 왼쪽 바퀴 원점 | base_link 기준 `(0, 0.222, 0)` m |
| 오른쪽 바퀴 원점 | base_link 기준 `(0, −0.222, 0)` m |
| 원점 기준 바퀴 간격 | 0.444 m |
| 각 바퀴 질량 | 10 kg |
| 각 바퀴 충돌 원통 1 | 반지름 0.11894 m, 길이 0.07292 m |
| 각 바퀴 충돌 원통 2 | 반지름 0.091 m, 길이 0.16004 m |
| base_footprint → base_link | xyz `(0, 0, −0.127)` m, RPY `(0, 0, 0)` |

좌우 두 바퀴의 배치로부터 차동 구동형 이동 구조로 해석할 수 있으나, 이 파일에는 차동 구동 플러그인이나 제어기가 없다. 바퀴의 속도·토크 제한도 정의되어 있지 않으므로 최대 주행 속도는 계산할 수 없다.

파일 상단에는 Xacro 속성 `wheel_radius=0.265`, `wheel_length=0.031`, `wheel_separation=0.45`가 선언되어 있다. 그러나 `${...}` 형태의 참조는 없고 실제 바퀴 정의는 숫자를 직접 사용한다. 따라서 속성값을 실제 반지름·폭·윤거로 그대로 사용하면 충돌 형상 및 관절 위치와 불일치한다.

또한 `base_joint`의 Z 오프셋은 음수다. 관례적인 지면 프레임을 가정해 양수로 바꾸어 해석하지 말고, 시뮬레이션 배치 및 지면 접촉에서 원본 변환을 확인해야 한다.

## 7. 카메라와 IMU 좌표계

| 연결 | 부모 기준 xyz (m) | RPY (rad) |
|---|---|---|
| head_pitch → zed_cam_link | `(0.0723, 0, 0.1244)` | `(0, 0, 0)` |
| zed_cam_link → zed_camera_center | `(0, 0, 0.016)` | `(0, 0, 0)` |
| zed_camera_center → zed_left_camera_frame | `(−0.01, 0.06, 0)` | `(0, 0, 0)` |
| zed_camera_center → zed_right_camera_frame | `(−0.01, −0.06, 0)` | `(0, 0, 0)` |
| 좌/우 camera_frame → 각 optical_frame | `(0, 0, 0)` | `(−1.57, 0, −1.57)` |
| zed_camera_center → zed_imu_link | `(0, 0, 0)` | `(0, 0, 0)` |
| base_link → imu_link | `(−0.08, 0, 0)` | `(0, 0, 0)` |

좌우 카메라 프레임 간 간격은 0.12 m다. 파일에는 센서 프레임의 위치·자세만 있으며 해상도, 시야각, 영상 주기, 카메라 내부 파라미터, IMU 노이즈, ROS 토픽은 정의되어 있지 않다.

## 8. 질량과 형상

질량 합계는 `<inertial><mass>` 값의 단순 합이다. 질량이 생략된 프레임의 실제 질량을 추정하거나 추가하지 않았다.

| 부위 | 포함 범위 | 질량 (kg) |
|---|---|---:|
| 이동 베이스 | base_link | 57.1361 |
| 양쪽 바퀴 | left/right_wheel_link | 20.0000 |
| 몸통 | torso_link 및 상·하부 몸통 4링크 | 22.1759 |
| 왼팔 | left_arm_link_1~7 | 7.0041 |
| 오른팔 | right_arm_link_1~7 | 7.0039 |
| 왼손 | left_base_link 및 왼손 손가락 | 0.3900 |
| 오른손 | right_base_link 및 오른손 손가락 | 0.3900 |
| 머리 | head_yaw, head_pitch | 1.3414 |
| **전체 합계** | 질량이 선언된 모든 링크 | **115.4414** |

- 본체·팔·머리·바퀴 STL에는 `scale="0.001 0.001 0.001"`가 지정된다. 손 STL에는 scale이 생략되어 기본 배율 1이 적용된다. 메시 단위를 동일하게 가정하면 안 된다.
- 충돌 형상은 box 45개, cylinder 35개, sphere 5개, 총 85개다. 시각화용 상세 메시와 충돌 계산용 단순 형상이 분리되어 있다.
- 전역 재질은 black, white, red, blue, green, grey의 6종이다. 손의 일부 visual에는 `grey` 이름과 함께 흰색 RGBA가 직접 정의되어 있으므로 실제 색상은 로더 처리도 확인해야 한다.
- 모든 메시 참조 경로는 URDF 디렉터리 기준 실제 파일이 존재한다. 상대 경로이므로 로더가 해당 기준 경로를 올바르게 처리해야 한다.
- 각 링크의 질량중심과 관성 텐서는 원본 `<inertial>`에 정의된다. 관성 origin은 visual/collision origin과 별개다.

## 9. 사용 시 확인할 사항과 분석 한계

1. **모델 질량과 실제 중량 구분:** 115.4414 kg은 모델에 기록된 합계다. `zed_cam_link`와 `imu_link`는 질량·관성이 0이며, tip 및 여러 센서 프레임에는 inertial이 없다.
2. **고정 손 연결:** `l_hand`, `r_hand`에는 axis와 limit가 있지만 type은 fixed다. 독립 자유도나 구동 관절로 세지 않는다.
3. **mimic 지원:** 시뮬레이터·제어 도구가 mimic을 지원하는지 확인해야 한다. 미지원이면 몸통과 손가락의 종속 관계를 별도로 적용해야 한다.
4. **좌우 비대칭 보존:** 어깨·손목 roll 범위 외에도 손가락 origin/rpy에 좌우 차이가 있다. 예를 들어 `right_index_touch_joint`의 RPY는 `(1.7145, 0.4756, 0)`이다. 좌우 복사로 대체하면 원본 모델과 달라진다.
5. **센서·제어 설정 별도 필요:** `<gazebo>`, `<transmission>`, `<ros2_control>` 및 구동 플러그인은 없다. URDF만으로 센서 발행이나 실제 하드웨어 제어가 구성되지는 않는다.
6. **파일 형식:** `.urdf` 확장자지만 Xacro 네임스페이스와 property 태그가 포함되어 있다. 숫자 참조 표현은 없으며 사용하는 로더에 따라 Xacro 전처리가 필요할 수 있다.
7. **확정할 수 없는 사양:** 실물 전체 높이·외곽 크기, 가반하중, 정격 토크, 배터리·운용 시간, 최대 주행 속도 및 실제 센서 모델은 이 파일만으로 확정할 수 없다. 전체 메시 외곽이나 동역학 성능 검증은 수행하지 않았다.

## 부록 A. 전체 링크 및 선언 질량

`—`는 inertial/mass 생략이며, 명시적 0과 구분한다.

| 링크 | 질량 (kg) |
|---|---:|
| `torso_link` | 9.8258 |
| `left_arm_link_1` | 0.4954 |
| `left_arm_link_2` | 3.0479 |
| `left_arm_link_3` | 1.2352 |
| `left_arm_link_4` | 0.2903 |
| `left_arm_link_5` | 1.1843 |
| `left_arm_link_6` | 0.6233 |
| `left_arm_link_7` | 0.1277 |
| `left_base_link` | 0.238 |
| `left_thumb_metacarpal_link` | 0.01 |
| `left_thumb_proximal_link` | 0.05 |
| `left_thumb_distal_link` | 0.01 |
| `left_thumb_tip_link` | — |
| `left_thumb_touch_link` | 0.002 |
| `left_index_proximal_link` | 0.009 |
| `left_index_distal_link` | 0.009 |
| `left_index_tip_link` | — |
| `left_index_touch_link` | 0.002 |
| `left_middle_proximal_link` | 0.009 |
| `left_middle_distal_link` | 0.009 |
| `left_middle_tip_link` | — |
| `left_middle_touch_link` | 0.002 |
| `left_ring_proximal_link` | 0.009 |
| `left_ring_distal_link` | 0.009 |
| `left_ring_tip_link` | — |
| `left_ring_touch_link` | 0.002 |
| `left_pinky_proximal_link` | 0.009 |
| `left_pinky_distal_link` | 0.009 |
| `left_pinky_tip_link` | — |
| `left_pinky_touch_link` | 0.002 |
| `right_arm_link_1` | 0.4954 |
| `right_arm_link_2` | 3.0477 |
| `right_arm_link_3` | 1.2352 |
| `right_arm_link_4` | 0.2903 |
| `right_arm_link_5` | 1.1843 |
| `right_arm_link_6` | 0.6233 |
| `right_arm_link_7` | 0.1277 |
| `right_base_link` | 0.238 |
| `right_thumb_metacarpal_link` | 0.01 |
| `right_thumb_proximal_link` | 0.05 |
| `right_thumb_distal_link` | 0.01 |
| `right_thumb_tip_link` | — |
| `right_thumb_touch_link` | 0.002 |
| `right_index_proximal_link` | 0.009 |
| `right_index_distal_link` | 0.009 |
| `right_index_tip_link` | — |
| `right_index_touch_link` | 0.002 |
| `right_middle_proximal_link` | 0.009 |
| `right_middle_distal_link` | 0.009 |
| `right_middle_tip_link` | — |
| `right_middle_touch_link` | 0.002 |
| `right_ring_proximal_link` | 0.009 |
| `right_ring_distal_link` | 0.009 |
| `right_ring_tip_link` | — |
| `right_ring_touch_link` | 0.002 |
| `right_pinky_proximal_link` | 0.009 |
| `right_pinky_distal_link` | 0.009 |
| `right_pinky_tip_link` | — |
| `right_pinky_touch_link` | 0.002 |
| `head_yaw` | 0.4363 |
| `head_pitch` | 0.9051 |
| `zed_cam_link` | 0 |
| `zed_camera_center` | — |
| `zed_right_camera_frame` | — |
| `zed_left_camera_frame` | — |
| `zed_imu_link` | — |
| `zed_right_camera_optical_frame` | — |
| `zed_left_camera_optical_frame` | — |
| `upper_torso_link` | 4.7273 |
| `upper_torso_mimic_link` | 2.7917 |
| `lower_torso_mimic_link` | 2.1028 |
| `lower_torso_link` | 2.7283 |
| `base_link` | 57.1361 |
| `imu_link` | 0 |
| `left_wheel_link` | 10 |
| `right_wheel_link` | 10 |
| `base_footprint` | — |

## 부록 B. 전체 관절 연결 및 원점

모든 관절을 원본 등장 순서대로 나열한다. xyz는 m, rpy는 rad다. 축의 `—`는 생략을 뜻한다. 고정 관절의 axis 값은 회전 자유도를 만들지 않는다.

| 관절 | 타입 | 부모 → 자식 | xyz | rpy | axis | mimic |
|---|---|---|---|---|---|---|
| `l_sh_p` | revolute | `torso_link` → `left_arm_link_1` | `0 0.24 0` | `0 0 0` | `0 1 0` | — |
| `l_sh_r` | revolute | `left_arm_link_1` → `left_arm_link_2` | `0 0 0` | `0 0 0` | `1 0 0` | — |
| `l_sh_y` | revolute | `left_arm_link_2` → `left_arm_link_3` | `0 0 0` | `0 0 0` | `0 0 1` | — |
| `l_el_p` | revolute | `left_arm_link_3` → `left_arm_link_4` | `0 0 -0.26` | `0 0 0` | `0 1 0` | — |
| `l_wr_y` | revolute | `left_arm_link_4` → `left_arm_link_5` | `0 0 -0.255` | `0 0 0` | `0 0 1` | — |
| `l_wr_p` | revolute | `left_arm_link_5` → `left_arm_link_6` | `0 0 0` | `0 0 0` | `0 1 0` | — |
| `l_wr_r` | revolute | `left_arm_link_6` → `left_arm_link_7` | `0 0 0` | `0 0 0` | `1 0 0` | — |
| `l_hand` | fixed | `left_arm_link_7` → `left_base_link` | `0 0 -0.077` | `0 3.1416 1.5708` | `1 0 0` | — |
| `left_thumb_metacarpal_joint` | revolute | `left_base_link` → `left_thumb_metacarpal_link` | `0.0045 -0.019 0.027825` | `0 0 0.16787` | `0 0 1` | — |
| `left_thumb_proximal_joint` | revolute | `left_thumb_metacarpal_link` → `left_thumb_proximal_link` | `0 -0.014227 0` | `-0.17488 0.23736 -0.16787` | `-1 0 0` | — |
| `left_thumb_distal_joint` | revolute | `left_thumb_proximal_link` → `left_thumb_distal_link` | `0 -0.052 0` | `0 0 0` | `-1 0 0` | left_thumb_proximal_joint × 1 + 0 |
| `left_thumb_tip_joint` | fixed | `left_thumb_distal_link` → `left_thumb_tip_link` | `0 -0.026529 -0.0017734` | `0.06675 0 0` | `0 0 0` | — |
| `left_thumb_touch_joint` | fixed | `left_thumb_distal_link` → `left_thumb_touch_link` | `0 -0.013836 0.0076686` | `0.17488 0 0` | `0 0 0` | — |
| `left_index_proximal_joint` | revolute | `left_base_link` → `left_index_proximal_link` | `-0.0021349 -0.029578 0.080876` | `0.11819 0.14743 0.17422` | `0 1 0` | — |
| `left_index_distal_joint` | revolute | `left_index_proximal_link` → `left_index_distal_link` | `0 0 0.032` | `0 0 0` | `0 1 0` | left_index_proximal_joint × 1.155 + 0 |
| `left_index_tip_joint` | fixed | `left_index_distal_link` → `left_index_tip_link` | `0.011097 0 0.035739` | `0 0.48723 0` | `0 0 0` | — |
| `left_index_touch_joint` | fixed | `left_index_distal_link` → `left_index_touch_link` | `0.014497 0 0.023303` | `0 0.4756 0` | `0 0 0` | — |
| `left_middle_proximal_joint` | revolute | `left_base_link` → `left_middle_proximal_link` | `-0.0045935 -0.010061 0.084993` | `0.039468 0.1483 0.057975` | `0 1 0` | — |
| `left_middle_distal_joint` | revolute | `left_middle_proximal_link` → `left_middle_distal_link` | `0 0 0.037` | `0 0.12654 0` | `0 1 0` | left_middle_proximal_joint × 1.155 + 0 |
| `left_middle_tip_joint` | fixed | `left_middle_distal_link` → `left_middle_tip_link` | `0.0073534 0 0.041703` | `0 0.34907 0` | `0 0 0` | — |
| `left_middle_touch_joint` | fixed | `left_middle_distal_link` → `left_middle_touch_link` | `0.012493 0 0.028393` | `0 0.34907 0` | `0 0 0` | — |
| `left_ring_proximal_joint` | revolute | `left_base_link` → `left_ring_proximal_link` | `-0.0046877 0.010026 0.083982` | `-0.039462 0.14721 -0.057932` | `0 1 0` | — |
| `left_ring_distal_joint` | revolute | `left_ring_proximal_link` → `left_ring_distal_link` | `0 0 0.035` | `0 0.12654 0` | `0 1 0` | left_ring_proximal_joint × 1.155 + 0 |
| `left_ring_tip_joint` | fixed | `left_ring_distal_link` → `left_ring_tip_link` | `0.0071824 0 0.040734` | `0 0.34907 0` | `0 0 0` | — |
| `left_ring_touch_joint` | fixed | `left_ring_distal_link` → `left_ring_touch_link` | `0.012028 0 0.027796` | `0 0.34907 0` | `0 0 0` | — |
| `left_pinky_proximal_joint` | revolute | `left_base_link` → `left_pinky_proximal_link` | `-0.0023733 0.029356 0.078694` | `-0.11815 0.14491 -0.17392` | `0 1 0` | — |
| `left_pinky_distal_joint` | revolute | `left_pinky_proximal_link` → `left_pinky_distal_link` | `0 0 0.029` | `0 0 0` | `0 1 0` | left_pinky_proximal_joint × 1.155 + 0 |
| `left_pinky_tip_joint` | fixed | `left_pinky_distal_link` → `left_pinky_tip_link` | `0.010221 0 0.032918` | `0 0.4756 0` | `0 0 0` | — |
| `left_pinky_touch_joint` | fixed | `left_pinky_distal_link` → `left_pinky_touch_link` | `0.01401 0 0.021124` | `0 0.4756 0` | `0 0 0` | — |
| `r_sh_p` | revolute | `torso_link` → `right_arm_link_1` | `0 -0.240 0` | `0 0 0` | `0 1 0` | — |
| `r_sh_r` | revolute | `right_arm_link_1` → `right_arm_link_2` | `0 0 0` | `0 0 0` | `1 0 0` | — |
| `r_sh_y` | revolute | `right_arm_link_2` → `right_arm_link_3` | `0 0 0` | `0 0 0` | `0 0 1` | — |
| `r_el_p` | revolute | `right_arm_link_3` → `right_arm_link_4` | `0 0 -0.26` | `0 0 0` | `0 1 0` | — |
| `r_wr_y` | revolute | `right_arm_link_4` → `right_arm_link_5` | `0 0 -0.255` | `0 0 0` | `0 0 1` | — |
| `r_wr_p` | revolute | `right_arm_link_5` → `right_arm_link_6` | `0 0 0` | `0 0 0` | `0 1 0` | — |
| `r_wr_r` | revolute | `right_arm_link_6` → `right_arm_link_7` | `0 0 0` | `0 0 0` | `1 0 0` | — |
| `r_hand` | fixed | `right_arm_link_7` → `right_base_link` | `0 0 -0.077` | `0 3.1416 -1.5708` | `1 0 0` | — |
| `right_thumb_metacarpal_joint` | revolute | `right_base_link` → `right_thumb_metacarpal_link` | `0.005619 0.019867 0.027825` | `0 0 -0.16787` | `0 0 -1` | — |
| `right_thumb_proximal_joint` | revolute | `right_thumb_metacarpal_link` → `right_thumb_proximal_link` | `0 0.014227 0` | `0.17488 0.23736 0.16787` | `1 0 0` | — |
| `right_thumb_distal_joint` | revolute | `right_thumb_proximal_link` → `right_thumb_distal_link` | `0 0.052 0` | `0 0 0` | `1 0 0` | right_thumb_proximal_joint × 1 + 0 |
| `right_thumb_tip_joint` | fixed | `right_thumb_distal_link` → `right_thumb_tip_link` | `0 0.0265287811635831 -0.00177344281307929` | `-0.0667504507189062 0 0` | `0 0 0` | — |
| `right_thumb_touch_joint` | fixed | `right_thumb_distal_link` → `right_thumb_touch_link` | `0 0.0137927522748974 0.00767659940411662` | `-0.0667504507189062 0 0` | `0 0 0` | — |
| `right_index_proximal_joint` | revolute | `right_base_link` → `right_index_proximal_link` | `-0.0021181 0.029568 0.080876` | `-0.11819 0.14743 -0.17422` | `0 1 0` | — |
| `right_index_distal_joint` | revolute | `right_index_proximal_link` → `right_index_distal_link` | `0 0 0.032` | `0 0 0` | `0 1 0` | right_index_proximal_joint × 1.155 + 0 |
| `right_index_tip_joint` | fixed | `right_index_distal_link` → `right_index_tip_link` | `0.011097 0 0.035739` | `0 0.4756 0` | `0 0 0` | — |
| `right_index_touch_joint` | fixed | `right_index_distal_link` → `right_index_touch_link` | `0.014497 0 0.023303` | `1.7145 0.4756 0` | `0 0 0` | — |
| `right_middle_proximal_joint` | revolute | `right_base_link` → `right_middle_proximal_link` | `-0.0045767 0.010051 0.084993` | `-0.039468 0.1483 -0.057975` | `0 1 0` | — |
| `right_middle_distal_joint` | revolute | `right_middle_proximal_link` → `right_middle_distal_link` | `0 0 0.037` | `0 0 0` | `0 1 0` | right_middle_proximal_joint × 1.155 + 0 |
| `right_middle_tip_joint` | fixed | `right_middle_distal_link` → `right_middle_tip_link` | `0.012558 0 0.040442` | `0 0.4756 0` | `0 0 0` | — |
| `right_middle_touch_joint` | fixed | `right_middle_distal_link` → `right_middle_touch_link` | `0.015975871997898 0 0.0265894268903603` | `0 0.475602221168456 0` | `0 0 0` | — |
| `right_ring_proximal_joint` | revolute | `right_base_link` → `right_ring_proximal_link` | `-0.0046709 -0.010037 0.083982` | `0.039462 0.14721 0.057932` | `0 1 0` | — |
| `right_ring_distal_joint` | revolute | `right_ring_proximal_link` → `right_ring_distal_link` | `0 0 0.035` | `0 0 0` | `0 1 0` | right_ring_proximal_joint × 1.155 + 0 |
| `right_ring_tip_joint` | fixed | `right_ring_distal_link` → `right_ring_tip_link` | `0.012266 0 0.039501` | `0 0.4756 0` | `0 0 0` | — |
| `right_ring_touch_joint` | fixed | `right_ring_distal_link` → `right_ring_touch_link` | `0.0154392324697935 0 0.0260561041381566` | `0 0.475602221168648 0` | `0 0 0` | — |
| `right_pinky_proximal_joint` | revolute | `right_base_link` → `right_pinky_proximal_link` | `-0.0023566 -0.029366 0.078694` | `0.11815 0.14491 0.17392` | `0 1 0` | — |
| `right_pinky_distal_joint` | revolute | `right_pinky_proximal_link` → `right_pinky_distal_link` | `0 0 0.029` | `0 0 0` | `0 1 0` | right_pinky_proximal_joint × 1.155 + 0 |
| `right_pinky_tip_joint` | fixed | `right_pinky_distal_link` → `right_pinky_tip_link` | `0.010221 0 0.032918` | `0 0.4756 0` | `0 0 0` | — |
| `right_pinky_touch_joint` | fixed | `right_pinky_distal_link` → `right_pinky_touch_link` | `0.01401 0 0.021124` | `0 0.4756 0` | `0 0 0` | — |
| `head_y` | revolute | `torso_link` → `head_yaw` | `0 0 0.08` | `0 0 0` | `0 0 1` | — |
| `head_p` | revolute | `head_yaw` → `head_pitch` | `0 0 0` | `0 0 0` | `0 1 0` | — |
| `zed_cam` | fixed | `head_pitch` → `zed_cam_link` | `0.0723 0 0.1244` | `0 0 0` | `—` | — |
| `zed_center_camera_joint` | fixed | `zed_cam_link` → `zed_camera_center` | `0.0 0.0 0.016` | `0 0 0` | `—` | — |
| `zed_right_camera_joint` | fixed | `zed_camera_center` → `zed_right_camera_frame` | `-0.01 -0.06 0.0` | `0 0 0` | `—` | — |
| `zed_left_camera_joint` | fixed | `zed_camera_center` → `zed_left_camera_frame` | `-0.01 0.06 0.0` | `0 0 0` | `—` | — |
| `zed_imu_joint` | fixed | `zed_camera_center` → `zed_imu_link` | `0 0 0` | `0 0 0` | `—` | — |
| `zed_right_camera_optical_joint` | fixed | `zed_right_camera_frame` → `zed_right_camera_optical_frame` | `0 0 0` | `-1.57 0.0 -1.57` | `—` | — |
| `zed_left_camera_optical_joint` | fixed | `zed_left_camera_frame` → `zed_left_camera_optical_frame` | `0 0 0` | `-1.57 0.0 -1.57` | `—` | — |
| `waist_y` | revolute | `upper_torso_link` → `torso_link` | `0.035 0 0.38` | `0 0 0` | `0 0 1` | — |
| `waist_upper_pitch` | revolute | `upper_torso_mimic_link` → `upper_torso_link` | `0.0 0 0.28` | `0 0 0` | `0 1 0` | — |
| `upper_torso_mimic` | revolute | `lower_torso_mimic_link` → `upper_torso_mimic_link` | `0 0 0.2312` | `0 0 0` | `0 1 0` | waist_upper_pitch × -1.0 + 0.0 |
| `lower_torso_mimic` | revolute | `lower_torso_link` → `lower_torso_mimic_link` | `0 0 0.28` | `0 0 0` | `0 1 0` | waist_lower_pitch × -1.0 + 0.0 |
| `waist_lower_pitch` | revolute | `base_link` → `lower_torso_link` | `-0.15 0 0.21` | `0 0 0` | `0 1 0` | — |
| `imu_fixed` | fixed | `base_link` → `imu_link` | `-0.08 0.0 0.0` | `0 0 0` | `—` | — |
| `left_wheel_joint` | continuous | `base_link` → `left_wheel_link` | `0 0.222 0.0` | `0 0.0 0.0` | `0 1 0` | — |
| `right_wheel_joint` | continuous | `base_link` → `right_wheel_link` | `0 -0.222 0.0` | `0.0 0.0 0.0` | `0 1 0` | — |
| `base_joint` | fixed | `base_footprint` → `base_link` | `0.0 0.0 -0.127` | `0 0 0` | `—` | — |
