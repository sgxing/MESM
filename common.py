import os
os.environ['JAX_PLATFORM_NAME'] = 'mps'
os.environ['JAX_MPS_ASYNC_DISPATCH'] = '0'
import time
from typing import Dict, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
from flax import nnx
import optax
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, log_loss


def compute_metrics_cpu(probs: Union[np.ndarray, jax.Array], labels: Union[np.ndarray, jax.Array]) -> Tuple[float, float]:
    """
    在 CPU 上计算二分类的 ROC-AUC 和 KS (Kolmogorov-Smirnov) 统计量。
    """
    p = np.asarray(probs, dtype=np.float32).reshape(-1)
    y = np.asarray(labels, dtype=np.float32).reshape(-1)
    
    # 避免全零或全一导致 AUC 报错
    if len(np.unique(y)) < 2:
        return 0.5, 0.0

    auc = float(roc_auc_score(y, p))
    desc = np.argsort(p)[::-1]
    y_sorted = y[desc]
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    tpr = np.cumsum(y_sorted) / max(n_pos, 1.0)
    fpr = np.cumsum(1.0 - y_sorted) / max(n_neg, 1.0)
    ks = float(np.max(np.abs(tpr - fpr)))
    return auc, ks


def compute_standard_eval_metrics(
    p_mob3: Union[np.ndarray, jax.Array],
    p_mob6: Union[np.ndarray, jax.Array],
    y_mob3: Union[np.ndarray, jax.Array],
    y_mob6: Union[np.ndarray, jax.Array],
    compute_rectified: bool = False,
    eps_vr: float = 0.0,
) -> Dict[str, float]:
    """
    统一计算跨期风控多任务 (MOB3 与 MOB6) 的核心评估指标:
    - 短期 (MOB3): AUC, KS, LogLoss
    - 长期 (MOB6): AUC, KS, LogLoss
    - 跨期单调违规率 VR: P(MOB6) < P(MOB3) - eps_vr
    - 可选后验规则修正: P(MOB6_rect) = max(P(MOB3), P(MOB6))
    """
    y3 = np.asarray(y_mob3, dtype=np.float32).reshape(-1)
    y6 = np.asarray(y_mob6, dtype=np.float32).reshape(-1)
    p3 = np.asarray(p_mob3, dtype=np.float32).reshape(-1)
    p6 = np.asarray(p_mob6, dtype=np.float32).reshape(-1)

    auc3, ks3 = compute_metrics_cpu(p3, y3)
    auc6, ks6 = compute_metrics_cpu(p6, y6)

    p3_clip = np.clip(p3, 1e-15, 1.0 - 1e-15)
    p6_clip = np.clip(p6, 1e-15, 1.0 - 1e-15)
    loss3 = float(log_loss(y3, p3_clip))
    loss6 = float(log_loss(y6, p6_clip))

    # 跨期单调倒挂违规统计: 业务逻辑上 3月内逾期是 6月内逾期的子集, 故 P(MOB6) >= P(MOB3) 恒成立
    is_violation = (p6 < p3 - eps_vr)
    n_violations = int(np.sum(is_violation))
    vr_pct = float(np.mean(is_violation)) * 100.0

    metrics: Dict[str, float] = {
        "auc_mob3": float(auc3),
        "ks_mob3": float(ks3),
        "loss_mob3": float(loss3),
        "auc_mob6": float(auc6),
        "ks_mob6": float(ks6),
        "loss_mob6": float(loss6),
        "violation_rate": float(vr_pct / 100.0),
        "vr_pct": float(vr_pct),
        "violations_count": float(n_violations),
        "total_samples": float(len(p6)),
    }

    if compute_rectified:
        p6_rect = np.maximum(p3, p6)
        auc6_rect, ks6_rect = compute_metrics_cpu(p6_rect, y6)
        p6_rect_clip = np.clip(p6_rect, 1e-15, 1.0 - 1e-15)
        loss6_rect = float(log_loss(y6, p6_rect_clip))
        metrics.update({
            "auc_mob6_rect": float(auc6_rect),
            "ks_mob6_rect": float(ks6_rect),
            "loss_mob6_rect": float(loss6_rect),
            "vr_pct_rect": 0.0,
        })

    return metrics


def focal_loss(
    logits: jax.Array,
    labels: jax.Array,
    gamma: float = 2.0,
    alpha: float = 0.75,
) -> jax.Array:
    """
    Focal Loss (针对信贷极端正负不平衡场景, 如违约率 1:7 甚至 1:20):
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """
    p = jax.nn.sigmoid(logits)
    p_t = labels * p + (1.0 - labels) * (1.0 - p)
    alpha_t = labels * alpha + (1.0 - labels) * (1.0 - alpha)
    bce = optax.sigmoid_binary_cross_entropy(logits, labels)
    loss = alpha_t * ((1.0 - p_t) ** gamma) * bce
    return jnp.mean(loss)


def linear_cka(h1: np.ndarray, h2: np.ndarray) -> float:
    """
    中心化核对齐 (Linear CKA, Centered Kernel Alignment, Kornblith et al. 2019):
    衡量两个专家表征矩阵之间的相似度 (越接近 0 分化越彻底, 越接近 1 越同质化)。
    """
    h1 = np.asarray(h1, dtype=np.float32)
    h2 = np.asarray(h2, dtype=np.float32)
    h1_c = h1 - h1.mean(axis=0, keepdims=True)
    h2_c = h2 - h2.mean(axis=0, keepdims=True)
    k1 = h1_c @ h1_c.T
    k2 = h2_c @ h2_c.T
    hsic = (k1 * k2).sum()
    norm1 = np.sqrt((k1 * k1).sum())
    norm2 = np.sqrt((k2 * k2).sum())
    return float(hsic / (norm1 * norm2 + 1e-8))


def compute_stats_summary(
    df_results: pd.DataFrame,
    has_rectified: bool = False,
) -> Dict[str, Dict[str, float]]:
    """
    计算多随机种子评测的均值与标准差 (Mean ± Std)。
    """
    cols = ["auc_mob3", "ks_mob3", "loss_mob3", "auc_mob6", "ks_mob6", "loss_mob6", "vr_pct"]
    if has_rectified and "auc_mob6_rect" in df_results.columns:
        cols += ["auc_mob6_rect", "ks_mob6_rect", "loss_mob6_rect", "vr_pct_rect"]

    stats: Dict[str, Dict[str, float]] = {}
    for col in cols:
        if col in df_results.columns:
            m = float(df_results[col].mean())
            s = float(df_results[col].std(ddof=1)) if len(df_results) > 1 else 0.0
            stats[col] = {"mean": m, "std": s}
    return stats


def print_experiment_summary(
    model_name: str,
    df_results: pd.DataFrame,
    stats_summary: Dict[str, Dict[str, float]],
    has_rectified: bool = False,
):
    """
    格式化打印多随机种子在测试集 (OOT) 上的评估报告及论文标准表格。
    """
    print("\n" + "=" * 96)
    print(f"📊 {model_name} 多随机种子测试集评测总榜 (Test Set / OOT)")
    print("=" * 96)
    cols_show = ["seed", "best_epoch", "auc_mob3", "ks_mob3", "loss_mob3", "auc_mob6", "ks_mob6", "loss_mob6", "vr_pct"]
    cols_exist = [c for c in cols_show if c in df_results.columns]
    print(df_results[cols_exist].to_string(index=False))

    print("\n" + "-" * 96)
    print(f"🏆 【硕士学位论文标准汇报格式 (Mean ± Std, {len(df_results)} 独立随机种子)】: {model_name}")
    print("-" * 96)
    
    if "auc_mob3" in stats_summary:
        m3, s3 = stats_summary["auc_mob3"]["mean"], stats_summary["auc_mob3"]["std"]
        k3, sk3 = stats_summary["ks_mob3"]["mean"], stats_summary["ks_mob3"]["std"]
        l3, sl3 = stats_summary["loss_mob3"]["mean"], stats_summary["loss_mob3"]["std"]
        print(f"  • 短期 (MOB3) : AUC = {m3:.4f} ± {s3:.4f} | KS = {k3:.4f} ± {sk3:.4f} | Loss = {l3:.4f} ± {sl3:.4f}")

    if "auc_mob6" in stats_summary:
        m6, s6 = stats_summary["auc_mob6"]["mean"], stats_summary["auc_mob6"]["std"]
        k6, sk6 = stats_summary["ks_mob6"]["mean"], stats_summary["ks_mob6"]["std"]
        l6, sl6 = stats_summary["loss_mob6"]["mean"], stats_summary["loss_mob6"]["std"]
        print(f"  • 长期 (MOB6) : AUC = {m6:.4f} ± {s6:.4f} | KS = {k6:.4f} ± {sk6:.4f} | Loss = {l6:.4f} ± {sl6:.4f}")

    if "vr_pct" in stats_summary:
        vr_m, vr_s = stats_summary["vr_pct"]["mean"], stats_summary["vr_pct"]["std"]
        print(f"  • 跨期倒挂违规率 VR: {vr_m:.2f}% ± {vr_s:.2f}%  (P(MOB6) < P(MOB3) 样本占比)")

    if has_rectified and "auc_mob6_rect" in stats_summary:
        m6_r, s6_r = stats_summary["auc_mob6_rect"]["mean"], stats_summary["auc_mob6_rect"]["std"]
        k6_r, sk6_r = stats_summary["ks_mob6_rect"]["mean"], stats_summary["ks_mob6_rect"]["std"]
        print(f"  • 规则后验修正 (MOB6_rect): AUC = {m6_r:.4f} ± {s6_r:.4f} | KS = {k6_r:.4f} ± {sk6_r:.4f} | VR = 0.00%")
    print("=" * 96 + "\n")
