개요
---
1. 스터디 목적(오민식)
2. 강화학습 기본 이론(이연수)
3. DRL 설명 => DQN(최원강), TD3(오민식)
4.
5. 5. project 주제 설명=> Pick and Place(최원강)
6. requiremnets(최원강)
7. ref code process architecture(옥윤정)
8. ref code 설명(이연수)
9. 시행착오들(image2action, camera2world coordinate, etc...)(옥윤정)
10. 최종 결과(오민식)
---
스터디 목적
---
스터디 기간 : 24.03.15 ~ 24.12.23

본 스터디는 심층 신경망 강화학습(Deep Reinforcement Learning : DRL) 공부를 통해 구성한 스터디로, 기초부터 시작하여, 실제 로봇에 적용하는 것으로 목표로 하였다.

결론적으로는 실제 로봇 적용에는 실패하였지만, DRL에 대한 이론 및 동작 방식을 이해할 수 있었다.

   
TD3 (Twin Delayed Deep Deterministic Policy Gradient) 이론 및 구조 설명
---
로봇 팔 Pick and Place 구현을 위해 TD3 (Twin Delayed Deep Deterministic Policy Gradient) 방법을 사용하려고 하였으나, 최종적으로는 실패하였다.

하지만 공부한 내용을 기록하기 위해 이론 설명을 추가하였으며, 참고한 논문에 대한 리뷰는 "ppt" 폴더에 정리되어 있다.

TD3는 강화학습 알고리즘 중 하나로, **DDPG (Deep Deterministic Policy Gradient)**의 한계를 극복하기 위해 제안된 방법이다. 

주로 연속적인 액션 공간에서 작동하며, 정책의 안정성과 학습 성능을 개선하는 데 중점을 둔다. 

TD3는 두 개의 주요 네트워크인 Actor Network와 Critic Network를 사용하며, 이를 통해 정책(Actor)과 가치 함수(Critic)를 학습한다.(그림 추가)

⚙️ 2. 구조와 구성 요소
---
A. Replay Buffer
경험 (s, a, r, s')을 저장한다. 경험(experience)은 agent가 environment와 상호작용한 정보를 의미하며, DRL(Deep Reinforcement Learning)은 경험을 training data로 활용한다.

목적: 데이터의 상관관계를 줄이고 학습을 안정화하기 위해 에이전트가 환경에서 수집한 경험을 replay buffer에서 무작위로 샘플링한다.

B. Critic Network
기능: 상태-액션 쌍 (s, a)에 대한 Q-값을 추정한다.

구성: 두 개의 독립적인 Q-함수 critic1, critic2와 이들의 타겟 네트워크 target1, target2로 구성된다.

TD Error Update (Temporal Difference Error): 두 Q-네트워크는 TD 에러를 기반으로 업데이트된다.

Target Q 값 비교: 두 Critic 네트워크에서 Q 값을 예측한 후, 더 작은 Q 값을 선택하여 overestimation bias를 줄인다.

Critic Network 학습 단계:

1. critic1과 critic2는 각각의 Q-값을 예측한다.

2. 타겟 네트워크(target1, target2)는 목표 Q-값을 계산한다. 두 Q-값 중 더 작은 값을 선택하여 학습한다.

C. Actor Network

기능: 주어진 상태 s에 대해 최적의 행동 a를 생성한다.

구성: actor와 target 두 네트워크로 구성된다.

DPG Update (Deterministic Policy Gradient): Actor Network는 Critic Network에서 전달받은 Q-값을 최대화하도록 업데이트된다.

Actor Network 학습 단계:

1. Actor Network는 주어진 상태 s에 대해 행동 μ(s)를 생성한다.

2. 행동에 노이즈 N를 추가하여 탐색을 수행한다.

3. Critic Network로부터 피드백을 받아 정책을 업데이트한다.

D. Environment

Actor Network에서 생성된 행동 a는 환경에 전달되고, 환경은 해당 행동을 받아들여 다음 상태 s'와 보상 r을 반환한다.

반환된 (s, a, r, s') 정보 experience는 Replay Buffer에 저장된다.

🔑 3. TD3의 핵심
---
Double Q-Learning

두 Critic Network를 사용하여 Q-값의 과대평가를 방지한다.
Target Policy Smoothing

Actor Network에 노이즈를 추가하여 급격한 정책 변화와 과적합을 방지한다.
Delayed Policy Update

Critic Network가 매 업데이트마다 학습되는 반면, Actor Network는 일정 주기마다 업데이트된다.
이로 인해 Critic Network가 더 안정적인 Q-값을 예측할 수 있다.

🛠️ 4. TD3 알고리즘 동작 순서
---
1) 초기화 단계

1-1) Actor와 Critic 네트워크 초기화

1-2) Replay Buffer 초기화

2) 경험 수집

2-1) Actor 네트워크를 통해 행동 a_t 선택

2-2) 환경에 행동 전달 및 보상과 다음 상태 (r_t, s_{t+1}) 반환

2-3) 경험 (s_t, a_t, r_t, s_{t+1})을 Replay Buffer에 저장

3) 샘플링 및 학습

3-1) Replay Buffer에서 무작위로 경험 샘플링

3-2) Critic 네트워크 업데이트 (TD 에러 기반)

3-3) Actor 네트워크는 Critic 네트워크 업데이트 주기마다 학습

4) 타겟 네트워크 업데이트

4-1) Critic 타겟과 Actor 타겟 네트워크는 지연된 소프트 업데이트 수행

5) 반복

5-1) 위 과정을 반복하여 정책과 Q-함수를 최적화
---

## 7. ref code process architecture
---
1. 초기 설정 및 모델 로드

코드 실행 시 load_model() 함수가 호출되어, 사전 학습된 YOLO 객체 탐지 모델이 메모리에 로드. 로봇의 초기 상태는 reset_robot()을 통해 관절과 그리퍼를 초기 위치로 설정하며, 시뮬레이션 환경도 초기화.

2. 객체 탐지

카메라 이미지 또는 주어진 입력 이미지를 detect() 함수로 처리. YOLO 모델을 사용하여 이미지 속에서 객체의 위치와 클래스 정보를 탐지하며, 바운딩 박스와 탐지된 클래스 리스트를 반환

3. 중심 좌표 계산

탐지된 객체의 바운딩 박스 정보를 기반으로 calculate_centroid() 함수가 호출. 객체의 중심 좌표(이미지 상의 픽셀 위치)를 계산하여, 이후 로봇이 해당 위치를 목표로 삼을 수 있도록 함.

4. 로봇 이동

move_robot() 함수는 로봇의 관절 각도를 조절하여, 로봇의 끝단(End Effector)이 객체의 위치로 이동. 내부적으로 calculateIK()를 통해 목표 위치를 기반으로 역운동학(Inverse Kinematics)을 계산하고, 관절 제어 명령을 실행.

5. 그리퍼 동작

로봇이 객체 근처에 도착하면 closeGripper()를 호출하여, 물체를 집을 수 있도록 그리퍼를 닫음. 객체를 이동한 뒤, openGripper()로 그리퍼를 열어 객체를 놓음.

6. 객체 생성 및 배치

시뮬레이션 환경 내 객체는 create_parallelepiped(), create_l_object(), create_z_object()와 같은 함수로 생성. 다양한 형태의 객체를 특정 위치에 배치하며, 초기 상태를 설정.

8. 시뮬레이션 업데이트

로봇의 움직임과 환경의 동작은 p.stepSimulation()을 통해 지속적으로 갱신되며, 모든 상태 변화가 물리적으로 계산. simulate_movement()는 목표 위치까지 로봇이 효율적으로 이동하도록 반복적인 계산과 시뮬레이션 단계를 처리

10. 종료 및 반복

모든 작업이 완료된 후, 로봇과 객체의 상태는 초기화되거나 다음 작업 흐름을 위해 새로운 명령을 대기.


## 9. 시행착오
---
1. 
