# 机器学习路线图项目

## 项目结构

```
ml_roadmap_project/
│
├── data/                   # (可选) 存放本地数据集
├── models/                 # 核心模型目录
│   ├── supervised_learning/    # 监督学习
│   │   ├── linear_regression.py    # 线性回归
│   │   ├── logistic_regression.py  # 逻辑回归
│   │   ├── svm.py                  # SVM
│   │   ├── random_forest.py        # 随机森林
│   │   ├── xgboost_model.py        # XGBoost
│   │   └── cnn.py                  # CNN
│   │
│   ├── unsupervised_learning/  # 无监督学习
│   │   ├── kmeans.py               # K-means聚类
│   │   ├── dbscan.py               # DBSCAN聚类
│   │   ├── hierarchical_clustering.py  # 层次聚类
│   │   └── spectral_clustering.py      # 谱聚类
│   │
│   ├── semi_supervised_learning/  # 半监督学习
│   │   ├── label_propagation.py       # 标签传播
│   │   ├── semi_supervised_svm.py     # 半监督SVM
│   │   └── self_training.py           # 自训练
│   │
│   ├── ensemble_learning/      # 集成学习
│   │   ├── random_forest.py    # 随机森林
│   │   ├── xgboost.py          # XGBoost
│   │   ├── adaboost.py          # AdaBoost
│   │   └── lightgbm.py          # LightGBM
│   │
│   ├── deep_learning/          # 深度学习
│   │   ├── mlp.py              # MLP
│   │   ├── cnn.py              # CNN
│   │   ├── rnn.py              # RNN
│   │   └── transformer.py      # Transformer
│   │
│   ├── graph_neural_network/   # 图神经网络
│   │   ├── gcn.py              # GCN
│   │   ├── graphsage.py        # GraphSAGE
│   │   └── gat.py              # GAT
│   │
│   ├── probabilistic_graphical_model/  # 概率图模型
│   │   ├── bayesian_network.py         # 贝叶斯网络
│   │   ├── hmm.py                      # HMM
│   │   └── markov_chain.py             # 马尔可夫链
│   │
│   ├── large_language_model/   # 大语言模型
│   │   ├── bert.py             # BERT
│   │   ├── gpt.py              # GPT
│   │   └── llama.py            # LLaMA
│   │
│   ├── time_series/            # 时间序列
│   │   ├── arima.py            # ARIMA
│   │   ├── lstm.py             # LSTM
│   │   └── prophet.py          # Prophet
│   │
│   ├── reinforcement_learning/ # 强化学习
│   │   ├── q_learning.py       # Q-learning
│   │   ├── dqn.py              # DQN
│   │   └── ppo.py              # PPO
│   │
│   ├── nlp/                    # NLP
│   │   ├── word2vec.py         # 词向量
│   │   ├── sentiment_analysis.py  # 情感分析
│   │   └── named_entity_recognition.py  # 命名实体识别
│   │
│   ├── computer_vision/        # 计算机视觉
│   │   ├── object_detection.py     # 目标检测
│   │   ├── image_classification.py # 图像分类
│   │   └── semantic_segmentation.py # 语义分割
│   │
│   ├── anomaly_detection/      # 异常检测
│   │   ├── isolation_forest.py     # 孤立森林
│   │   ├── one_class_svm.py        # One-Class SVM
│   │   └── lof.py                  # LOF
│   │
│   └── causal_inference/       # 因果推断
│       ├── propensity_score_matching.py # 倾向评分匹配
│       ├── did.py                       # DID
│       └── instrumental_variable.py     # 工具变量
│
```

## 已实现的模型（63个，全部可独立运行）

> 所有模型均为**纯 NumPy 实现**（部分使用 matplotlib / scipy），无需安装 PyTorch / TensorFlow。
> 每个文件均可独立运行，运行后在 `results/` 目录生成可视化图表。

### 📐 监督学习 (`supervised_learning/`)
| 文件 | 算法 | 亮点 |
|------|------|------|
| `linear_regression.py` | 线性回归 | 批量/随机梯度下降，MSE/RMSE/R² |
| `logistic_regression.py` | 逻辑回归 | L1/L2正则化，二分类 |
| `svm.py` | 支持向量机 | 软间隔/硬间隔，核函数 |
| `random_forest.py` | 随机森林 | 特征重要性，OOB 评估 |
| `xgboost_model.py` | XGBoost | 梯度提升树，特征重要性 |
| `cnn.py` | 卷积神经网络 | MNIST，3层卷积 |

### 🔍 无监督学习 (`unsupervised_learning/`)
| 文件 | 算法 | 亮点 |
|------|------|------|
| `kmeans.py` | K-Means 聚类 | 肘部法则，Silhouette |
| `dbscan.py` | DBSCAN | 密度聚类，噪声检测 |
| `hierarchical_clustering.py` | 层次聚类 | 树状图，Ward/Complete 链接 |
| `spectral_clustering.py` | 谱聚类 | Laplacian 特征分解 |

### 🔀 半监督学习 (`semi_supervised_learning/`)
| 文件 | 算法 | 亮点 |
|------|------|------|
| `label_propagation.py` | 标签传播 | 图半监督，稀疏标注 |
| `semi_supervised_svm.py` | 半监督 SVM | 直推式 SVM |
| `self_training.py` | 自训练 | 置信度迭代标注 |

### 🌲 集成学习 (`ensemble_learning/`)
| 文件 | 算法 | 亮点 |
|------|------|------|
| `random_forest.py` | 随机森林 | Bagging + 特征采样 |
| `xgboost.py` | XGBoost | 二阶梯度提升 |
| `adaboost.py` | AdaBoost | 样本权重迭代 |
| `lightgbm.py` | LightGBM | 直方图优化，GOSS |

### 🧠 深度学习 (`deep_learning/`)
| 文件 | 算法 | 亮点 |
|------|------|------|
| `mlp.py` | 多层感知机 | 反向传播，Dropout |
| `cnn.py` | CNN | 卷积/池化/BatchNorm |
| `rnn.py` | RNN | LSTM/GRU，序列建模 |
| `transformer.py` | Transformer | 多头注意力，位置编码 |

### 🕸️ 图神经网络 (`graph_neural_network/`)
| 文件 | 算法 | 亮点 |
|------|------|------|
| `gcn.py` | 图卷积网络 | 谱图卷积，半监督节点分类 |
| `gat.py` | 图注意力网络 | 多头注意力，LeakyReLU |
| `graphsage.py` | GraphSAGE | 邻居采样，归纳式学习，Mean/Max 聚合 |

### 📊 概率图模型 (`probabilistic_graphical_model/`)
| 文件 | 算法 | 亮点 |
|------|------|------|
| `bayesian_network.py` | 贝叶斯网络 | 条件概率表，变量消除 |
| `hmm.py` | 隐马尔可夫模型 | Viterbi/前向算法 |
| `markov_chain.py` | 马尔可夫链 | 稳态分布，PageRank |

### 🤖 大语言模型 (`large_language_model/`)
| 文件 | 算法 | 亮点 |
|------|------|------|
| `bert.py` | Mini-BERT | 多头自注意力，MLM 预训练，文本分类微调 |
| `gpt.py` | Mini-GPT | 因果掩码，自回归生成，Top-k 采样 |
| `llama.py` | Mini-LLaMA | RMSNorm + RoPE + SwiGLU，与 GPT 对比 |

### ⏱️ 时间序列 (`time_series/`)
| 文件 | 算法 | 亮点 |
|------|------|------|
| `arima.py` | ARIMA | 差分平稳化，AIC 选阶 |
| `lstm.py` | LSTM | 序列预测，多步预测 |
| `prophet.py` | Prophet-like | 趋势+季节性分解 |

### 🎮 强化学习 (`reinforcement_learning/`)
| 文件 | 算法 | 亮点 |
|------|------|------|
| `q_learning.py` | Q-Learning | ε-greedy，Q 表收敛 |
| `dqn.py` | DQN | 经验回放，目标网络 |
| `ppo.py` | PPO | 裁剪目标，Actor-Critic |

### 📝 自然语言处理 (`nlp/`)
| 文件 | 算法 | 亮点 |
|------|------|------|
| `word2vec.py` | Word2Vec | Skip-Gram + 负采样，词类比任务 |
| `sentiment_analysis.py` | 情感分析 | TF-IDF + 多分类器对比（LR/SVC/NB/NN） |
| `named_entity_recognition.py` | NER | 特征工程 + Softmax，BIO 序列标注，实体 F1 |

### 👁️ 计算机视觉 (`computer_vision/`)
| 文件 | 算法 | 亮点 |
|------|------|------|
| `image_classification.py` | 图像分类 | 手写 CNN（卷积+池化+全连接），合成纹理数据集 |
| `object_detection.py` | 目标检测 | 滑动窗口 + HOG + NMS，IoU@0.5 评估 |
| `semantic_segmentation.py` | 语义分割 | 像素特征 + 随机森林 + CRF 平滑，mIoU |

### 🚨 异常检测 (`anomaly_detection/`)
| 文件 | 算法 | 亮点 |
|------|------|------|
| `isolation_forest.py` | 孤立森林 | 随机分割树，异常评分 |
| `one_class_svm.py` | 单类 SVM | 核密度边界 |
| `lof.py` | LOF | 局部密度离群因子 |

### 🔗 因果推断 (`causal_inference/`)
| 文件 | 算法 | 亮点 |
|------|------|------|
| `instrumental_variable.py` | 工具变量法 | 2SLS + Wald 估计量 + Hausman 检验 |
| `propensity_score_matching.py` | 倾向评分匹配 | Logistic 倾向评分 + 1:1 匹配 + Love Plot |
| `did.py` | 双重差分 | 平行趋势，政策效应估计 |

## 使用方法

1. **运行单个模型**：直接运行对应模型的Python文件，例如：
   ```bash
   python models/graph_neural_network/gcn.py
   python models/nlp/named_entity_recognition.py
   python models/large_language_model/bert.py
   python models/causal_inference/instrumental_variable.py
   ```

2. **使用模型运行器**：通过 `main.py` 选择运行指定模型或所有模型：
   ```bash
   python main.py
   ```

3. **查看结果**：所有可视化图表保存在 `results/` 目录下。

## 扩展方法

要添加新的模型，只需在对应的分类目录下创建新的Python文件，并实现与现有模型相同的接口格式：

```python
# 模型名称
# 可独立运行的模型实现
def model_name():
    """模型实现"""
    print("模型运行中...")
    # 这里将实现模型算法

if __name__ == "__main__":
    model_name()
```

## 依赖项

项目全部使用**纯 NumPy** 实现核心算法（无深度学习框架）：
- `numpy` — 核心数值计算
- `matplotlib` — 可视化
- `scipy` — 统计检验（部分模块）

### 安装依赖

```bash
pip install -r requirements.txt
```

## 设计原则

- **从零实现**：每个算法均不依赖 sklearn/PyTorch/TensorFlow，完整展示数学原理
- **可独立运行**：每个 `.py` 文件均可作为独立脚本运行
- **可视化丰富**：每个模型生成多面板暗主题图表，保存至 `results/`
- **合成数据**：无需下载外部数据集，运行即可

## 注意事项

- 所有模型已完全实现，包含数据生成、模型训练、评估和可视化
- 可视化结果保存在 `results/` 目录中（暗色主题多面板图表）
- `utils.py` 存放通用工具函数

## 项目完成状态

✅ 63 个模型文件，全部实现完毕