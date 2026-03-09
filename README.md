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

## 已经实现的模型

1. **线性回归**
   - 实现了基本的线性回归模型
   - 支持批量梯度下降和随机梯度下降优化
   - 提供了模型评估指标，如均方误差（MSE）、均方根误差（RMSE）、R2分数等

2. **逻辑回归**
   - 实现了逻辑回归模型，用于二分类问题
   - 支持L1正则化和L2正则化
   - 提供了模型评估指标，如准确率、精确率、召回率、F1分数等

3. **支持向量机（SVM）**
   - 实现了线性SVM和核SVM模型
   - 支持软间隔和硬间隔分类
   - 提供了模型评估指标，如准确率、精确率、召回率、F1分数等

4. **随机森林** (`random_forest.py`)
   - 使用100棵树的随机森林分类器
   - 分析并可视化特征重要性
   - 输出详细的分类评估指标

5. **XGBoost** (`xgboost_model.py`)
   - 使用梯度提升树分类器
   - 分析并可视化特征重要性
   - 输出模型评估指标

6. **CNN** (`cnn.py`)
   - 使用MNIST手写数字数据集
   - 构建包含3个卷积层的CNN模型
   - 可视化训练过程和预测结果
   - 输出模型准确率和分类报告

## 使用方法

1. **运行单个模型**：直接运行对应模型的Python文件，例如：
   ```bash
   python models/supervised_learning/linear_regression.py
   ```

2. **使用模型运行器**：通过main.py选择运行指定模型或所有模型：
   ```bash
   python main.py
   ```
   然后按照提示输入模型编号即可。

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

项目需要以下依赖项：
- numpy
- matplotlib
- scikit-learn
- xgboost (用于XGBoost模型)
- tensorflow (用于CNN模型)

### 安装依赖

```bash
pip install -r requirements.txt
```

## 注意事项

- 监督学习模型已完全实现，包含数据准备、模型训练、评估和可视化功能
- 其他模型类别目前只提供了框架结构，具体实现需要根据实际需求进行填充
- 所有模型的可视化结果会保存在results目录中
- 可以根据需要在data目录下存放本地数据集
- utils.py文件可用于存放通用工具函数，如数据处理、绘图等

## 未来计划

- 完善其他模型类别的具体实现
- 添加模型评估和比较功能
- 增加更多的模型类型和变种
- 提供示例数据集和使用案例
- 优化模型性能和代码结构