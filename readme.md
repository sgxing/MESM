# MESM: 融合特征嵌入与时序因果建模的信贷风控专家混合跨期预测系统

> **Master's Thesis Project**: *Mixture-of-Experts Credit Scoring Model Integrating Feature Embedding and Temporal Modeling (MESM)*  
> 基于 JAX / Flax NNX 现代化生态与 Apple Silicon Metal 硬件加速构建。

---

## 1. 核心架构与模型设计

本项目针对信贷征信风控中静态高维特征、长期信贷时序轨迹、多机构查询记录以及跨期违约（MOB3 与 MOB6）单调性约束难题。

```
                              ┌──────────────────────────────────────────────────────────────┐
                              │                    信贷征信多模态输入特征                    │
                              └──────┬────────────────────┬───────────────────┬──────────────┘
                                     │                    │                   │
                        ┌────────────▼──────────┐ ┌───────▼────────┐ ┌────────▼──────────┐
                        │   Dense 静态连续特征   │ │  类别特征 Embedding │ │   时序因果网络    │
                        │ (15维基础+137维贷款聚合│ │ (DeterministicEmbed │ │ (月度池化扩张因果 │
                        │  +57维查询聚合=209维) │ │  自定义VJP消除分叉) │ │   TCN 编码器)    │
                        └────────────┬──────────┘ └───────┬────────┘ └────────┬──────────┘
                                     │                    │                   │
                                     └────────────────────┼───────────────────┘
                                                          ▼
                                            ┌───────────────────────────┐
                                            │      高维多模态融合隐空间 │
                                            └─────────────┬─────────────┘
                                                          ▼
                        ┌─────────────────────────────────────────────────────────────────┐
                        │        DeepSeek-V3 风格稀疏专家混合 (SparseMoE Backbone)        │
                        │  • 1 个始终激活的共享专家 (Shared Expert, 捕获全局基线风险)     │
                        │  • E 个细分稀疏专家 (Top-K 动态选路, 自适应分化借款人群体)      │
                        │  • DeepSeek-V3 动态偏置自均衡 (Aux-loss-free, 梯度零扭曲)       │
                        └───────────────────────────────┬─────────────────────────────────┘
                                                        │
                                      ┌─────────────────┴─────────────────┐
                                      ▼                                   ▼
                       ┌──────────────────────────────┐    ┌──────────────────────────────┐
                       │     短期逾期塔 (MOB3 Tower)  │    │     增量逾期塔 (Cond Delta)  │
                       │    独立门控 + 专用风险输出头 │    │    独立门控 + 专用对数增量头 │
                       └──────────────┬───────────────┘    └──────────────┬───────────────┘
                                      │                                   │
                                 logits_mob3                     logits_delta (对数增量)
                                      │                                   │
                                      │                        ┌──────────▼──────────┐
                                      │                        │   softplus 激活     │
                                      │                        │ (非负单调增量保证)  │
                                      │                        └──────────┬──────────┘
                                      │                                   │
                                      │                          z_delta (>= 0)
                                      │                                   │
                                      ├───────────────────────────────────┘
                                      ▼
             ┌──────────────────────────────────────────────────┐
             │ 严格硬单调长期逾期: logits_mob6 = logits_mob3 + z_delta │
             │ 跨期倒挂违规率 (Violation Rate) 恒为 0.00% 成立   │
             └──────────────────────────────────────────────────┘
```

---

## 2. 环境准备与安装

```bash
# 安装基础核心依赖 (推荐 Python >= 3.10, 本项目已针对 Python 3.12 深度调优)
pip install -r requirements.txt

# (macOS Apple Silicon 芯片用户若使用 Metal 加速，推荐安装)
pip install jax-mps
```

---

## 3. 命令行运行指南

### 3.1 核心提出模型评测 (MESM: ESMM + 动态偏置 MoE + 时序 TCN)
```bash
# 默认运行 MESM 模型并输出论文汇报指标
python3.12 main.py --model esmm

# 指定多个随机种子以评估均值与标准差 (Mean ± Std)
python3.12 main.py --model esmm --seeds 777,42,100 --epochs 50
```

### 3.2 基准与消融对比实验
```bash
# 评测单任务 MoE 预测 MOB6
python3.12 main.py --model moe_mob6

# 评测单任务 MoE 预测 MOB3
python3.12 main.py --model moe_mob3

# 评测消融基线 (完全置空贷款与查询时序特征，验证因果时序卷积的贡献)
python3.12 main.py --model ablation
```

### 3.3 一键全量 Benchmark 大比武 (生成学位论文全模型横向对比总表)
```bash
python3.12 main.py --model all --seeds 777,42,100
```

### 3.4 MoE 专家分化与门控路由深度诊断
```bash
# 打印专家输入权重 (W1) 余弦相似度、Top-K 选择率分布以及隐层激活 CKA 对齐矩阵
python3.12 main.py --model esmm --diag
```

### 3.5 两级可解释性归因与 SHAP 分析
```bash
# 计算时序加法 Shapley 贡献分解并生成 SHAP 特征重要度榜单与图像 (ranking & waterfall)
python3.12 main.py --model moe_mob6 --explain
```

---

## 4. 目录与工程结构

```
MESM/
├── common.py           # 公共基础模块: 评测指标 (AUC, KS, VR, LogLoss), 统计汇总与损失函数
├── read_and_feature.py # 数据工程流水线: JSON 序列解析、时序聚合特征衍生、标准化与秒级缓存
├── load_data.py        # 数据加载快捷导出入口 (向后兼容)
├── model.py            # 核心网络定义: DeterministicEmbed, LoanTCN, QueryEncoder, SparseMoE, ESMM
├── train.py            # 训练与评估引擎: JIT 编译步、早停与权重回滚、多随机种子循环与 Benchmark
├── main.py             # CLI 命令行入口: 支持单模型实验、大比武、MoE 诊断与 SHAP 可解释性
├── sample.csv          # 征信建模真实样例数据集
├── requirements.txt    # 项目环境依赖清册
└── README.md           # 项目详细说明文档与实验指南
```

---

## 5. 学术成果与核心创新点

1. **确定性反向传播与芯片级加速**：
   引入自定义 VJP 的 `DeterministicEmbed`，在 Apple Silicon MPS / GPU 上将稀疏乱序写入转换为矩阵乘法归约，消除不可复现梯度分叉。
2. **时序月度池化与因果卷积 (TCN)**：
   将借款人分散且非等间隔的信贷与查询明细按月聚合，结合扩张因果卷积与时间衰减注意力池化，兼顾长期依赖与近期风险敏感度。
3. **DeepSeek-V3 启发式动态偏置 (Aux-loss-free MoE)**：
   摆脱传统辅助损失破坏主任务梯度的弊端，通过在线偏置偏差调节实现专家负载均衡，保障专家网络功能极致分化。
4. **数学级硬单调性保证 (Hard Monotonicity)**：
   在 logit 空间通过 Softplus 增量桥接，证明并实证了 $\text{VR} \equiv 0.00\%$，彻底根除信贷跨期风险倒挂。
