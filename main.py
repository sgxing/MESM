import os
os.environ['JAX_PLATFORM_NAME'] = 'mps'
os.environ['JAX_MPS_ASYNC_DISPATCH'] = '0'
import argparse
import time
from typing import Any, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
from flax import nnx
import optax
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap

import model
import common
from read_and_feature import load_data
from train import (
    train_single_seed_esmm,
    run_10_seeds_esmm,
    train_single_seed_moe,
    run_10_seeds_moe,
    train_single_seed_ablation,
    run_10_seeds_ablation,
    run_model_experiment,
    run_benchmark,
    MODEL_DISPATCHER,
)


# ====================================================================
# 1. 专家分化与门控路由诊断 (MoE Diagnostics)
#    - W1 专家首层权重余弦相似度矩阵 (同质化检测)
#    - 专家激活隐层表征 CKA (Centered Kernel Alignment)
#    - 动态门控路由选择率与 DeepSeek-V3 偏置监测
# ====================================================================

def run_moe_diagnostics(
    net: Any,
    dataset: Dict[str, Any],
    n_sample: int = 500,
):
    """
    执行 MoE 专家分化与路由行为全面诊断 (兼容 ESMM 双塔模型与单任务 MoE 模型)
    """
    print("\n" + "=" * 70)
    print("🔬 正在执行 MoE 专家分化与门控路由行为诊断 (MoE Diagnostics)")
    print("=" * 70)

    moe = getattr(net, "esmm", getattr(net, "moe", None))
    if moe is None:
        print("⚠️ 当前模型未包含 MoE 专家模块，跳过专家诊断。")
        return

    n_exp = moe.num_experts

    # 1. W1 余弦相似度矩阵
    print("\n[1/3] 专家输入层权重 (W1) 余弦相似度矩阵 (越小 = 专家特征偏好分化越好):")
    W1s = np.stack([
        np.asarray(moe.expert_w1[e].kernel[...]).reshape(-1)
        for e in range(n_exp)
    ])
    W1s = W1s / (np.linalg.norm(W1s, axis=1, keepdims=True) + 1e-8)
    cos_mat = W1s @ W1s.T

    header = "      " + "".join([f"{'E'+str(i):>8}" for i in range(n_exp)])
    print(header)
    for i in range(n_exp):
        row = f"E{i:<4} " + "".join([f"{cos_mat[i, j]:>8.4f}" for j in range(n_exp)])
        print(row)

    mask = 1.0 - np.eye(n_exp)
    mean_cos = float(np.abs(cos_mat * mask).sum() / (n_exp * (n_exp - 1)))
    print(f"\n  • 非对角元素平均 |余弦相似度|: {mean_cos:.4f}")
    if mean_cos < 0.5:
        print("  • 诊断结论: ✓ 专家权重矩阵已显著分化，无同质化冗余！")
    else:
        print("  • 诊断结论: ℹ️ 专家呈现中度相关，具备部分通用共享表征。")

    # 2. 提取验证集特征表征与门控路由统计
    X_static = dataset['X_valid_static']
    X_cat = dataset['X_valid_cat']
    batch_loan = dataset['batch_loan_valid']
    batch_query = dataset['batch_query_valid']

    actual_n = min(n_sample, len(X_static))
    xs = jnp.asarray(X_static[:actual_n])
    xc = jnp.asarray(X_cat[:actual_n])
    bl = {k: jnp.asarray(v[:actual_n]) for k, v in batch_loan.items()}
    bq = {k: jnp.asarray(v[:actual_n]) for k, v in batch_query.items()}

    d = net._shared_branch(xs, xc, training=False)
    if hasattr(net, 'loan_tcn_encoder') and hasattr(net, 'query_encoder'):
        t = net.loan_tcn_encoder(
            bl['product_ids'], bl['amounts'], bl['recencies'], bl['time_deltas'],
            bl['seq_lengths'], bl['num_products_per_month'], training=False
        )
        q = net.query_encoder(
            bq['org_ids'], bq['reason_ids'], bq['event_counts'], bq['recencies'],
            bq['time_deltas'], bq['seq_lengths'], bq['num_per_month'], training=False
        )
        fused = jnp.concatenate([d, t, q], axis=-1)
    else:
        fused = d

    # 门控 Logits 与 Top-K 索引
    if hasattr(moe, 'gate_mob3_w1'):
        # ESMM 双门控 (MOB3 塔 vs 增量 Cond 塔)
        gate_m3 = nnx.relu(moe.gate_mob3_w1(fused))
        logits_m3 = moe.gate_mob3_w2(gate_m3)
        probs_m3 = jax.nn.sigmoid(logits_m3)
        topk_m3 = np.asarray(jax.lax.top_k(probs_m3 + moe.expert_bias_mob3[None, :], moe.top_k)[1])

        gate_cond = nnx.relu(moe.gate_cond_w1(fused))
        logits_cond = moe.gate_cond_w2(gate_cond)
        probs_cond = jax.nn.sigmoid(logits_cond)
        topk_cond = np.asarray(jax.lax.top_k(probs_cond + moe.expert_bias_cond[None, :], moe.top_k)[1])

        print(f"\n[2/3] ESMM 双门控 (MOB3 塔 vs 增量 Cond 塔) 专家选中率分布 (Top-{moe.top_k}):")
        counts_m3 = np.bincount(topk_m3.reshape(-1), minlength=n_exp)
        counts_cond = np.bincount(topk_cond.reshape(-1), minlength=n_exp)
        total_slots = actual_n * moe.top_k

        print(f"{'Expert':<8}{'MOB3 选中数':>12}{'MOB3 选中率':>14}{'Cond 选中数':>14}{'Cond 选中率':>14}")
        print("-" * 64)
        for e in range(n_exp):
            r_m3 = counts_m3[e] / total_slots * 100.0
            r_cond = counts_cond[e] / total_slots * 100.0
            print(f"E{e:<7}{counts_m3[e]:>12,}{r_m3:>13.2f}%{counts_cond[e]:>14,}{r_cond:>13.2f}%")
    else:
        # 单任务 MoE 门控
        gate_h = nnx.relu(moe.gate_w1(fused))
        logits_g = moe.gate_w2(gate_h)
        probs_g = jax.nn.sigmoid(logits_g)
        topk_g = np.asarray(jax.lax.top_k(probs_g + moe.expert_bias[None, :], moe.top_k)[1])

        print(f"\n[2/3] 单任务 MoE 门控专家选中率分布 (Top-{moe.top_k}):")
        counts_g = np.bincount(topk_g.reshape(-1), minlength=n_exp)
        total_slots = actual_n * moe.top_k

        print(f"{'Expert':<8}{'选中数':>12}{'选中率':>14}")
        print("-" * 36)
        for e in range(n_exp):
            r_g = counts_g[e] / total_slots * 100.0
            print(f"E{e:<7}{counts_g[e]:>12,}{r_g:>12.2f}%")

    # 3. 专家特征表征的 CKA 矩阵
    print("\n[3/3] 激活样本上的专家隐层输出表征 Linear CKA 矩阵:")
    expert_hiddens = []
    for e in range(n_exp):
        h = nnx.relu(moe.expert_w1[e](fused))
        h = nnx.relu(moe.expert_w2[e](h))
        expert_hiddens.append(np.asarray(h))

    cka_mat = np.zeros((n_exp, n_exp))
    for i in range(n_exp):
        for j in range(n_exp):
            cka_mat[i, j] = common.linear_cka(expert_hiddens[i], expert_hiddens[j])

    header_cka = "      " + "".join([f"{'E'+str(i):>8}" for i in range(n_exp)])
    print(header_cka)
    for i in range(n_exp):
        row = f"E{i:<4} " + "".join([f"{cka_mat[i, j]:>8.4f}" for j in range(n_exp)])
        print(row)
    print("=" * 70 + "\n")


# ====================================================================
# 2. 跨期序列可加分解与 SHAP 可解释性归因 (Explainability)
# ====================================================================

def run_explainability_analysis(
    net: Any,
    dataset: Dict[str, Any],
    n_background: int = 100,
    n_explain: int = 200,
    save_prefix: str = "mesm",
):
    """
    执行两级可解释性归因:
    1. 宏观序列加法分解 (Log-odds 空间的 Shapley 分解: Base, Loan Seq, Query Seq)
    2. 微观静态特征归因 (SHAP KernelExplainer) + 序列贡献拼接
    3. 输出特征排名表并保存图片
    """
    print("\n" + "=" * 70)
    print("🔍 正在执行两级可解释性归因与 SHAP 特征重要度分析 (Explainability)")
    print("=" * 70)

    X_static = dataset['X_valid_static']
    X_cat = dataset['X_valid_cat']
    batch_loan = dataset['batch_loan_valid']
    batch_query = dataset['batch_query_valid']
    N_val = len(X_static)

    # 1. 预计算空序列基线
    max_months = batch_loan['time_deltas'].shape[1]
    max_prods = batch_loan['product_ids'].shape[2]
    max_q_months = batch_query['time_deltas'].shape[1]
    max_q_per_month = batch_query['org_ids'].shape[2]

    el_empty = {
        'product_ids': np.zeros((N_val, max_months, max_prods), dtype=np.int32),
        'amounts': np.zeros((N_val, max_months, max_prods), dtype=np.float32),
        'recencies': np.zeros((N_val, max_months, max_prods), dtype=np.float32),
        'time_deltas': np.zeros((N_val, max_months), dtype=np.float32),
        'seq_lengths': np.zeros((N_val,), dtype=np.int32),
        'num_products_per_month': np.zeros((N_val, max_months), dtype=np.int32),
    }

    eq_empty = {
        'org_ids': np.zeros((N_val, max_q_months, max_q_per_month), dtype=np.int32),
        'reason_ids': np.zeros((N_val, max_q_months, max_q_per_month), dtype=np.int32),
        'event_counts': np.zeros((N_val, max_q_months, max_q_per_month), dtype=np.float32),
        'recencies': np.zeros((N_val, max_q_months, max_q_per_month), dtype=np.float32),
        'time_deltas': np.zeros((N_val, max_q_months), dtype=np.float32),
        'seq_lengths': np.zeros((N_val,), dtype=np.int32),
        'num_per_month': np.zeros((N_val, max_q_months), dtype=np.int32),
    }

    def compute_logits(xs, xc, bl, bq):
        bs_inner = 256
        N = len(xs)
        all_logits = []
        for s in range(0, N, bs_inner):
            e = min(s + bs_inner, N)
            out = net(
                jnp.asarray(xs[s:e]), jnp.asarray(xc[s:e]),
                jnp.asarray(bl['product_ids'][s:e]), jnp.asarray(bl['amounts'][s:e]),
                jnp.asarray(bl['recencies'][s:e]), jnp.asarray(bl['time_deltas'][s:e]),
                jnp.asarray(bl['seq_lengths'][s:e]), jnp.asarray(bl['num_products_per_month'][s:e]),
                jnp.asarray(bq['org_ids'][s:e]), jnp.asarray(bq['reason_ids'][s:e]),
                jnp.asarray(bq['event_counts'][s:e]), jnp.asarray(bq['recencies'][s:e]),
                jnp.asarray(bq['time_deltas'][s:e]), jnp.asarray(bq['seq_lengths'][s:e]),
                jnp.asarray(bq['num_per_month'][s:e]), training=False
            )
            if isinstance(out, tuple):
                # ESMM / Ablation: (logits_mob3, logits_mob6, aux_mob3, aux_delta)
                # MoE: (logits, aux_loss)
                l = out[1] if len(out) == 4 else out[0]
            else:
                l = out
            all_logits.append(np.asarray(l.squeeze(-1), dtype=np.float32))
        return np.concatenate(all_logits, axis=0) if len(all_logits) > 0 else np.zeros((0,), dtype=np.float32)

    print("\n[1/3] 计算序列在 Log-odds 空间的加法贡献 (Shapley 分解)...")
    base_logits = compute_logits(X_static, X_cat, el_empty, eq_empty)
    loan_logits = compute_logits(X_static, X_cat, batch_loan, eq_empty)
    query_logits = compute_logits(X_static, X_cat, el_empty, batch_query)
    full_logits = compute_logits(X_static, X_cat, batch_loan, batch_query)

    loan_contrib = 0.5 * ((loan_logits - base_logits) + (full_logits - query_logits))
    query_contrib = 0.5 * ((query_logits - base_logits) + (full_logits - loan_logits))

    print(f"  • Base Logits (空序列基线)  : Mean = {base_logits.mean():+.4f} (Std = {base_logits.std():.4f})")
    print(f"  • 贷款时序贡献 (Loan Seq)    : Mean = {loan_contrib.mean():+.4f} (Std = {loan_contrib.std():.4f})")
    print(f"  • 查询时序贡献 (Query Seq)   : Mean = {query_contrib.mean():+.4f} (Std = {query_contrib.std():.4f})")

    # 2. SHAP 归因
    print(f"\n[2/3] 构建 SHAP KernelExplainer (背景样本 {n_background}, 待解释样本 {n_explain})...")
    X_num = X_static.astype(np.float64)
    X_cat_f = X_cat.astype(np.float64)
    X_all = np.concatenate([X_num, X_cat_f], axis=1)

    D_num = X_num.shape[1]
    D_cat = X_cat_f.shape[1]

    rng = np.random.RandomState(42)
    bg_idx = rng.choice(N_val, size=min(n_background, N_val), replace=False)
    exp_idx = rng.choice(N_val, size=min(n_explain, N_val), replace=False)

    X_bg = X_all[bg_idx]
    X_exp = X_all[exp_idx]

    def shap_predict_fn(x_flat):
        x_flat = np.atleast_2d(x_flat)
        N = len(x_flat)
        xs_sub = x_flat[:, :D_num].astype(np.float32)
        xc_sub = x_flat[:, D_num:D_num + D_cat].astype(np.int64)

        el_sub = {k: np.zeros((N, *v.shape[1:]), dtype=v.dtype) for k, v in el_empty.items()}
        eq_sub = {k: np.zeros((N, *v.shape[1:]), dtype=v.dtype) for k, v in eq_empty.items()}
        logits = compute_logits(xs_sub, xc_sub, el_sub, eq_sub)
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -15.0, 15.0)))
        return np.clip(probs, 1e-7, 1.0 - 1e-7)

    explainer = shap.KernelExplainer(shap_predict_fn, X_bg, link="logit")
    shap_vals_raw = explainer.shap_values(X_exp, nsamples=min(100, n_explain))
    if isinstance(shap_vals_raw, list):
        shap_vals_raw = shap_vals_raw[0]

    loan_exp_contrib = loan_contrib[exp_idx].reshape(-1, 1)
    query_exp_contrib = query_contrib[exp_idx].reshape(-1, 1)
    final_shap_values = np.concatenate([shap_vals_raw, loan_exp_contrib, query_exp_contrib], axis=1)

    all_feat_names = list(dataset['feature_names'])

    # 3. 输出 Top 特征排名表
    print("\n[3/3] 全特征 |SHAP| 均值绝对贡献 Top 20 榜单:")
    mean_abs_shap = np.abs(final_shap_values).mean(axis=0)
    top_indices = np.argsort(mean_abs_shap)[::-1][:20]

    print(f"{'排名':<6}{'特征名称':<32}{'|SHAP| 均值':>14}")
    print("-" * 54)
    for rank, idx in enumerate(top_indices, 1):
        fn = all_feat_names[idx] if idx < len(all_feat_names) else f"Feature_{idx}"
        print(f"{rank:<6}{fn:<32}{mean_abs_shap[idx]:>14.4f}")

    # 保存特征重要度柱状图
    fig, ax = plt.subplots(figsize=(10, 8))
    top_rev = top_indices[::-1]
    top_names = [all_feat_names[i] if i < len(all_feat_names) else f"Feat_{i}" for i in top_rev]
    ax.barh(top_names, mean_abs_shap[top_rev], color="#1f77b4")
    ax.set_xlabel("Mean |SHAP Value| (Log-odds Impact)", fontsize=12)
    ax.set_title("MESM Feature Attribution Ranking (Top 15)", fontsize=14)
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    bar_path = f"{save_prefix}_shap_ranking.png"
    plt.savefig(bar_path, dpi=200)
    plt.close()
    print(f"  • ✅ 特征排名图已保存至: {bar_path}")

    # 保存单样本 Waterfall 瀑布图
    sample_idx = 0
    base_val = float(np.asarray(explainer.expected_value).ravel()[0])
    waterfall_path = f"{save_prefix}_shap_waterfall.png"
    plt.figure(figsize=(10, 6))
    shap.waterfall_plot(
        shap.Explanation(
            values=final_shap_values[sample_idx],
            base_values=base_val,
            data=np.concatenate([X_exp[sample_idx], [loan_exp_contrib[sample_idx, 0], query_exp_contrib[sample_idx, 0]]]),
            feature_names=all_feat_names,
        ),
        max_display=12,
        show=False,
    )
    plt.tight_layout()
    plt.savefig(waterfall_path, dpi=200)
    plt.close()
    print(f"  • ✅ 样本个体预测瀑布图已保存至: {waterfall_path}")
    print("=" * 70 + "\n")


# ====================================================================
# 3. CLI 命令行参数解析与运行主入口
# ====================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MESM: 融合特征嵌入与时序因果建模的信贷风控专家混合跨期预测系统",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="esmm",
        choices=["esmm", "moe_mob6", "moe_mob3", "ablation", "all"],
        help="待评测模型选择: esmm (本文核心提议), moe_mob6 (单任务MOB6), moe_mob3 (单任务MOB3), ablation (无时序消融), all (全模型基准大比武)",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="777",
        help="评测随机种子列表 (逗号分隔, 例如: '777' 或 '777,42,100,2024,2025')",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="模型训练最大轮数",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="训练批次大小 (Batch Size)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-4,
        help="AdamW 优化器初始学习率",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="早停容忍轮数 (Early Stopping Patience)",
    )
    parser.add_argument(
        "--force-recompute",
        action="store_true",
        help="强制重新执行全量特征工程并刷新缓存",
    )
    parser.add_argument(
        "--diag",
        action="store_true",
        help="在训练完成后对 MoE 专家分化 (W1 相似度、CKA) 和门控路由统计进行全面诊断",
    )
    parser.add_argument(
        "--explain",
        action="store_true",
        help="执行序列贡献分解与 SHAP 特征重要度可解释性分析并保存归因可视化图像",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. 打印运行平台与设备信息
    print("=" * 70)
    print("🌟 启动 MESM: 信贷风控混合专家跨期预测系统")
    print(f"   JAX 版本: {jax.__version__} | 后端设备: {jax.devices()}")
    print("=" * 70)

    # 2. 加载数据集
    dataset = load_data(force_recompute=args.force_recompute)

    # 3. 解析随机种子
    seed_list = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    # 4. 执行模型实验
    if args.model == "all":
        run_benchmark(
            dataset=dataset,
            seeds=seed_list,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            patience=args.patience,
        )
    else:
        run_model_experiment(
            model_name=args.model,
            dataset=dataset,
            seeds=seed_list,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            patience=args.patience,
        )


    # 5. 若启用诊断模式或可解释性模式，训练一个用于分析的完整模型
    if args.diag or args.explain:
        first_seed = seed_list[0]
        model_name = args.model if args.model != "all" else "esmm"
        print(f"\n⚙️ 正在初始化用于诊断与可解释性分析的 {model_name} 模型 (Seed {first_seed})...")
        if model_name == "esmm":
            trained_net, _ = train_single_seed_esmm(
                dataset=dataset,
                seed=first_seed,
                epochs=min(args.epochs, 30),
                lr=args.lr,
                batch_size=args.batch_size,
                patience=args.patience,
            )
        elif model_name in ["moe_mob6", "moe_mob3"]:
            task = 'mob6' if model_name == 'moe_mob6' else 'mob3'
            trained_net, _ = train_single_seed_moe(
                dataset=dataset,
                task=task,
                seed=first_seed,
                epochs=min(args.epochs, 30),
                lr=args.lr,
                batch_size=args.batch_size,
                patience=args.patience,
            )
        elif model_name == "ablation":
            trained_net, _ = train_single_seed_ablation(
                dataset=dataset,
                seed=first_seed,
                epochs=min(args.epochs, 30),
                lr=args.lr,
                batch_size=args.batch_size,
                patience=args.patience,
            )
        else:
            raise ValueError(f"未知模型类型: {model_name}")

        if args.diag:
            run_moe_diagnostics(trained_net, dataset)

        if args.explain:
            run_explainability_analysis(trained_net, dataset, save_prefix=f"{model_name}")

    print("\n🎉 全部任务执行完毕！")


if __name__ == "__main__":
    main()
