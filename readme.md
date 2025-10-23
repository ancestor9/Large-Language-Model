### 🔍 LLM 및 딥러닝 vs scikit-learn 모델 비교표


| 구분                            | **Deep Learning (예: LLM, CNN, Transformer)**                                                                                 | **scikit-learn (예: SVM, RandomForest, LinearRegression)**                                                                                                                                 |
| ----------------------------- | ---------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Architecture**              | 복잡한 **신경망 구조(Neural Architecture)**<br>- Transformer, CNN, RNN, Attention 등<br>- 수백~수천 개 레이어와 파라미터로 구성                       | 고정된 **수학적/통계적 모델 구조**<br>- 선형결합, 결정트리, 서포트벡터, 클러스터 중심 등<br>- 구조가 단순하고 해석 가능                                                                                                               |
| **Training Algorithm / Loss** | - **Gradient Descent 기반 최적화** (SGD, Adam, AdamW)<br>- 손실함수: Cross-Entropy, MSE, Contrastive, RLHF 등<br>- GPU/TPU 기반 병렬 학습 필수 | - 다양한 **최적화 절차 또는 해석적 해법**<br>  • LinearRegression → Normal Equation, GD<br>  • SVM → Quadratic Programming<br>  • DecisionTree → Gini/Entropy 기준 분할<br>- 손실: MSE, Hinge Loss, Log-Loss 등 |
| **Data**                      | - 수십억~수조 토큰 규모의 **비정형 대규모 데이터** (텍스트, 이미지, 오디오 등)<br>- 데이터 품질, 다양성, 균형성이 성능에 결정적                                             | - 상대적으로 **작은 구조화 데이터셋**<br>- CSV, 수치형 feature 중심<br>- 전처리(Scaling, Encoding)가 중요                                                                                                          |
| **Evaluation**                | - BLEU, ROUGE, Accuracy, Perplexity, F1 등<br>- Human evaluation (RLHF, preference-based)<br>- 다양한 downstream task로 검증        | - 정확도(Accuracy), F1-score, ROC-AUC, RMSE 등<br>- k-fold cross-validation, confusion matrix 등 정량 평가                                                                                         |
| **Systems**                   | - **대규모 분산 학습 시스템** 필요 (GPU/TPU 클러스터, DeepSpeed, Megatron, ZeRO)<br>- Mixed Precision, Checkpoint, Pipeline 병렬화              | - **단일 CPU 또는 소형 서버**에서도 충분<br>- 라이브러리 수준 병렬화 지원 (n_jobs, joblib 등)                                                                                                                       |
| **주요 특징**                     | - 고용량, 고복잡도, 범용 표현 학습<br>- “End-to-End” 학습 및 Fine-tuning 가능<br>- 연산 자원, 데이터 품질에 성능 의존                                        | - 가볍고 빠름, 해석 용이<br>- 데이터 크기 및 특징 수에 제한<br>- 파라미터 적고 과적합 제어 용이                                                                                                                             |
| **대표 구현체**                    | PyTorch, TensorFlow, JAX                                                                                                     | scikit-learn, XGBoost, LightGBM                                                                                                                                                           |

### 🧩 요약적 관점
| 관점         | Deep Learning                        | scikit-learn                     |
| ---------- | ------------------------------------ | -------------------------------- |
| **모델 중심성** | Architecture와 Training Algorithm이 핵심 | 데이터 전처리와 Feature Engineering이 핵심 |
| **연산 자원**  | GPU/TPU 필수                           | CPU 기반으로 충분                      |
| **학습 방식**  | End-to-End 자동 학습                     | 수동 피처 설계 + 경량 학습                 |
| **확장성**    | 대규모 분산/병렬 필수                         | 단일 머신에서도 효율적                     |
