import os
os.environ['JAX_PLATFORM_NAME'] = 'mps'
os.environ['JAX_MPS_ASYNC_DISPATCH'] = '0'
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
from flax import nnx
import optax
import numpy as np
import pandas as pd

import model
import common
from read_and_feature import load_data


# ====================================================================
# 1. 显存/设备张量驻留预加载辅助
# ====================================================================

def to_device_batch(batch_dict: Dict[str, np.ndarray]) -> Dict[str, jax.Array]:
    """将时序字典内所有 numpy ndarray 转换为 jax DeviceArray"""
    return {k: jnp.asarray(v) for k, v in batch_dict.items()}


# ====================================================================
# 2. ESMM 双塔硬单调模型训练与评估 (核心模型)
# ====================================================================

@nnx.jit
def train_step_esmm(
    model_net: model.HybridCreditModel_ESMM,
    optimizer: nnx.Optimizer,
    batch_xs: jax.Array,
    batch_xc: jax.Array,
    batch_loan: Dict[str, jax.Array],
    batch_query: Dict[str, jax.Array],
    y6: jax.Array,
    y3: jax.Array,
) -> jax.Array:
    """
    全空间多任务 ESMM 损失训练单步:
    L = BCE(p_mob3, y3) + BCE(p_mob6, y6) + (aux_mob3 + aux_cond)
    """
    def loss_fn(net):
        logits_mob3, logits_mob6, aux_mob3, aux_delta = net(
            batch_xs, batch_xc,
            batch_loan['product_ids'], batch_loan['amounts'],
            batch_loan['recencies'], batch_loan['time_deltas'],
            batch_loan['seq_lengths'], batch_loan['num_products_per_month'],
            batch_query['org_ids'], batch_query['reason_ids'],
            batch_query['event_counts'], batch_query['recencies'],
            batch_query['time_deltas'], batch_query['seq_lengths'],
            batch_query['num_per_month'], training=True
        )
        l6 = jnp.mean(optax.sigmoid_binary_cross_entropy(logits_mob6.squeeze(-1), y6))
        l3 = jnp.mean(optax.sigmoid_binary_cross_entropy(logits_mob3.squeeze(-1), y3))
        total = l6 + l3 + aux_mob3 + aux_delta
        return total

    loss, grads = nnx.value_and_grad(loss_fn)(model_net)
    optimizer.update(model_net, grads)
    return loss


@nnx.jit
def eval_forward_esmm(
    model_net: model.HybridCreditModel_ESMM,
    batch_xs: jax.Array,
    batch_xc: jax.Array,
    batch_loan: Dict[str, jax.Array],
    batch_query: Dict[str, jax.Array],
    y6: jax.Array,
    y3: jax.Array,
) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """返回 (total_loss, p_mob6, p_mob3, loss6, loss3)"""
    logits_mob3, logits_mob6, aux_mob3, aux_delta = model_net(
        batch_xs, batch_xc,
        batch_loan['product_ids'], batch_loan['amounts'],
        batch_loan['recencies'], batch_loan['time_deltas'],
        batch_loan['seq_lengths'], batch_loan['num_products_per_month'],
        batch_query['org_ids'], batch_query['reason_ids'],
        batch_query['event_counts'], batch_query['recencies'],
        batch_query['time_deltas'], batch_query['seq_lengths'],
        batch_query['num_per_month'], training=False
    )
    p_mob6 = jax.nn.sigmoid(logits_mob6.squeeze(-1))
    p_mob3 = jax.nn.sigmoid(logits_mob3.squeeze(-1))
    loss6 = jnp.mean(optax.sigmoid_binary_cross_entropy(logits_mob6.squeeze(-1), y6))
    loss3 = jnp.mean(optax.sigmoid_binary_cross_entropy(logits_mob3.squeeze(-1), y3))
    total_loss = loss6 + loss3
    return total_loss, p_mob6, p_mob3, loss6, loss3


def evaluate_full_esmm(
    model_net: model.HybridCreditModel_ESMM,
    X_static: np.ndarray,
    X_cat: np.ndarray,
    batch_loan: Dict[str, np.ndarray],
    batch_query: Dict[str, np.ndarray],
    y6: np.ndarray,
    y3: np.ndarray,
    eval_bs: int = 1024,
) -> Tuple[float, Dict[str, float]]:
    """分批次对完整数据集进行全面指标评估"""
    N = len(X_static)
    all_p6, all_p3 = [], []
    total_losses, l6_list, l3_list = [], [], []

    for i in range(0, N, eval_bs):
        end = min(i + eval_bs, N)
        xs = jnp.asarray(X_static[i:end])
        xc = jnp.asarray(X_cat[i:end])
        bl = {k: jnp.asarray(v[i:end]) for k, v in batch_loan.items()}
        bq = {k: jnp.asarray(v[i:end]) for k, v in batch_query.items()}
        b_y6 = jnp.asarray(y6[i:end], dtype=jnp.float32)
        b_y3 = jnp.asarray(y3[i:end], dtype=jnp.float32)

        t_loss, p6, p3, l6, l3 = eval_forward_esmm(model_net, xs, xc, bl, bq, b_y6, b_y3)
        bs_actual = end - i
        total_losses.append(float(t_loss) * bs_actual)
        l6_list.append(float(l6) * bs_actual)
        l3_list.append(float(l3) * bs_actual)
        all_p6.append(np.asarray(p6, dtype=np.float32))
        all_p3.append(np.asarray(p3, dtype=np.float32))

    p6_arr = np.concatenate(all_p6)
    p3_arr = np.concatenate(all_p3)
    mean_total_loss = float(sum(total_losses) / N)

    metrics = common.compute_standard_eval_metrics(
        p_mob3=p3_arr, p_mob6=p6_arr, y_mob3=y3, y_mob6=y6, compute_rectified=False
    )
    metrics["loss_total"] = mean_total_loss
    return mean_total_loss, metrics


def train_single_seed_esmm(
    dataset: Dict[str, Any],
    seed: int = 777,
    epochs: int = 100,
    lr: float = 5e-4,
    batch_size: int = 512,
    patience: int = 10,
    weight_decay: float = 0.1,
    verbose: bool = True,
) -> Tuple[model.HybridCreditModel_ESMM, Dict[str, float]]:
    """单随机种子完整训练流程: HybridCreditModel_ESMM"""
    if verbose:
        print(f"\n{'='*30} 🚀 开始训练 MESM-ESMM [Seed {seed}] {'='*30}")

    np.random.seed(seed)
    rngs = nnx.Rngs(seed)
    net = model.HybridCreditModel_ESMM(
        rngs=rngs,
        num_static_features=dataset['num_static_features'],
        cat_vocab_sizes=dataset['cat_vocab_sizes'],
        num_product_types=len(dataset['loan_vocab']),
        num_query_org_types=len(dataset['query_vocab_org']),
        num_query_reason_types=len(dataset['query_vocab_reason']),
        dropout_rate=0.5,
    )

    optimizer = nnx.Optimizer(net, optax.adamw(lr, weight_decay=weight_decay), wrt=nnx.Param)

    X_train_static = dataset['X_train_static']
    X_train_cat = dataset['X_train_cat']
    batch_loan_train = dataset['batch_loan_train']
    batch_query_train = dataset['batch_query_train']
    y_train_mob6 = dataset['y_train_mob6']
    y_train_mob3 = dataset['y_train_mob3']

    X_valid_static = dataset['X_valid_static']
    X_valid_cat = dataset['X_valid_cat']
    batch_loan_valid = dataset['batch_loan_valid']
    batch_query_valid = dataset['batch_query_valid']
    y_valid_mob6 = dataset['y_valid_mob6']
    y_valid_mob3 = dataset['y_valid_mob3']

    N_train = len(X_train_static)

    best_val_loss = float('inf')
    best_epoch = 0
    best_params = None
    best_bias_mob3 = None
    best_bias_cond = None
    wait = 0

    t_start = time.time()
    for ep in range(1, epochs + 1):
        perm = np.random.permutation(N_train)
        for i, s in enumerate(range(0, N_train, batch_size)):
            e = min(s + batch_size, N_train)
            idx = perm[s:e]
            xs = jnp.asarray(X_train_static[idx])
            xc = jnp.asarray(X_train_cat[idx])
            bl = {k: jnp.asarray(v[idx]) for k, v in batch_loan_train.items()}
            bq = {k: jnp.asarray(v[idx]) for k, v in batch_query_train.items()}
            y6_b = jnp.asarray(y_train_mob6[idx], dtype=jnp.float32)
            y3_b = jnp.asarray(y_train_mob3[idx], dtype=jnp.float32)

            train_step_esmm(net, optimizer, xs, xc, bl, bq, y6_b, y3_b)

        val_loss, val_m = evaluate_full_esmm(
            net, X_valid_static, X_valid_cat, batch_loan_valid, batch_query_valid, y_valid_mob6, y_valid_mob3
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = ep
            best_params = jax.tree_util.tree_map(lambda x: x.copy(), nnx.state(net, nnx.Param))
            best_bias_mob3 = net.esmm.expert_bias_mob3[...].copy()
            best_bias_cond = net.esmm.expert_bias_cond[...].copy()
            wait = 0
            star = " ★ (Best)"
        else:
            wait += 1
            star = ""

        if verbose and (ep % 5 == 0 or ep == 1 or wait == 0):
            print(f"Epoch {ep:3d}/{epochs} | Val Total Loss={val_loss:.4f} | MOB6 AUC={val_m['auc_mob6']:.4f} KS={val_m['ks_mob6']:.4f} | MOB3 AUC={val_m['auc_mob3']:.4f} KS={val_m['ks_mob3']:.4f} | VR={val_m['vr_pct']:.2f}%{star}")

        if wait >= patience:
            if verbose:
                print(f"早停触发！已在 Epoch {best_epoch} 达到最佳 Val Loss: {best_val_loss:.4f}")
            break

    # 恢复最佳验证集权重
    if best_params is not None:
        nnx.update(net, nnx.state(net, nnx.Param), best_params)
        net.esmm.expert_bias_mob3[...] = best_bias_mob3
        net.esmm.expert_bias_cond[...] = best_bias_cond

    t_cost = time.time() - t_start

    # 在 OOT 测试集上评估
    _, test_m = evaluate_full_esmm(
        net, dataset['X_test_static'], dataset['X_test_cat'],
        dataset['batch_loan_test'], dataset['batch_query_test'],
        dataset['y_test_mob6'], dataset['y_test_mob3']
    )
    test_m['best_epoch'] = best_epoch
    test_m['train_time_sec'] = t_cost
    test_m['seed'] = seed

    if verbose:
        print(f"✅ MESM-ESMM [Seed {seed}] 测试集 (OOT) 最终评测: MOB6 AUC={test_m['auc_mob6']:.4f} | MOB6 KS={test_m['ks_mob6']:.4f} | VR={test_m['vr_pct']:.2f}% | 耗时 {t_cost:.1f}s")
    return net, test_m


def run_10_seeds_esmm(
    dataset: Dict[str, Any],
    seeds: Optional[List[int]] = None,
    epochs: int = 100,
    lr: float = 5e-4,
    batch_size: int = 512,
    patience: int = 10,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    """运行多随机种子 ESMM 评测并汇总论文指标"""
    if seeds is None:
        seeds = [777, 42, 100, 2024, 2025, 123, 456, 789, 888, 999]

    results = []
    for s in seeds:
        _, m = train_single_seed_esmm(
            dataset=dataset, seed=s, epochs=epochs, lr=lr, batch_size=batch_size, patience=patience, verbose=True
        )
        results.append(m)

    df_res = pd.DataFrame(results)
    summary = common.compute_stats_summary(df_res, has_rectified=False)
    common.print_experiment_summary("MESM (Proposed Hybrid MoE + ESMM Hard-Monotonic)", df_res, summary, has_rectified=False)
    return df_res, summary


# ====================================================================
# 3. 单任务基线模型 (HybridCreditModel_MoE) 训练与评测
# ====================================================================

@nnx.jit
def train_step_moe(
    model_net: model.HybridCreditModel_MoE,
    optimizer: nnx.Optimizer,
    batch_xs: jax.Array,
    batch_xc: jax.Array,
    batch_loan: Dict[str, jax.Array],
    batch_query: Dict[str, jax.Array],
    y: jax.Array,
) -> jax.Array:
    """单任务训练单步"""
    def loss_fn(net):
        logits, aux_loss = net(
            batch_xs, batch_xc,
            batch_loan['product_ids'], batch_loan['amounts'],
            batch_loan['recencies'], batch_loan['time_deltas'],
            batch_loan['seq_lengths'], batch_loan['num_products_per_month'],
            batch_query['org_ids'], batch_query['reason_ids'],
            batch_query['event_counts'], batch_query['recencies'],
            batch_query['time_deltas'], batch_query['seq_lengths'],
            batch_query['num_per_month'], training=True
        )
        loss = jnp.mean(optax.sigmoid_binary_cross_entropy(logits.squeeze(-1), y))
        return loss + aux_loss

    loss, grads = nnx.value_and_grad(loss_fn)(model_net)
    optimizer.update(model_net, grads)
    return loss


@nnx.jit
def eval_forward_moe(
    model_net: model.HybridCreditModel_MoE,
    batch_xs: jax.Array,
    batch_xc: jax.Array,
    batch_loan: Dict[str, jax.Array],
    batch_query: Dict[str, jax.Array],
    y: jax.Array,
) -> Tuple[jax.Array, jax.Array]:
    logits, _ = model_net(
        batch_xs, batch_xc,
        batch_loan['product_ids'], batch_loan['amounts'],
        batch_loan['recencies'], batch_loan['time_deltas'],
        batch_loan['seq_lengths'], batch_loan['num_products_per_month'],
        batch_query['org_ids'], batch_query['reason_ids'],
        batch_query['event_counts'], batch_query['recencies'],
        batch_query['time_deltas'], batch_query['seq_lengths'],
        batch_query['num_per_month'], training=False
    )
    p = jax.nn.sigmoid(logits.squeeze(-1))
    loss = jnp.mean(optax.sigmoid_binary_cross_entropy(logits.squeeze(-1), y))
    return loss, p


def predict_proba_moe(
    model_net: model.HybridCreditModel_MoE,
    X_static: np.ndarray,
    X_cat: np.ndarray,
    batch_loan: Dict[str, np.ndarray],
    batch_query: Dict[str, np.ndarray],
    y: np.ndarray,
    eval_bs: int = 1024,
) -> Tuple[float, np.ndarray]:
    N = len(X_static)
    all_p = []
    total_losses = []
    for i in range(0, N, eval_bs):
        end = min(i + eval_bs, N)
        xs = jnp.asarray(X_static[i:end])
        xc = jnp.asarray(X_cat[i:end])
        bl = {k: jnp.asarray(v[i:end]) for k, v in batch_loan.items()}
        bq = {k: jnp.asarray(v[i:end]) for k, v in batch_query.items()}
        b_y = jnp.asarray(y[i:end], dtype=jnp.float32)

        l, p = eval_forward_moe(model_net, xs, xc, bl, bq, b_y)
        total_losses.append(float(l) * (end - i))
        all_p.append(np.asarray(p, dtype=np.float32))

    p_arr = np.concatenate(all_p)
    mean_loss = float(sum(total_losses) / N)
    return mean_loss, p_arr


def train_single_seed_moe(
    dataset: Dict[str, Any],
    task: str = "mob6",
    seed: int = 777,
    epochs: int = 100,
    lr: float = 5e-4,
    batch_size: int = 512,
    patience: int = 10,
    weight_decay: float = 0.1,
    verbose: bool = True,
) -> Tuple[model.HybridCreditModel_MoE, Dict[str, float]]:
    """单任务 MoE 训练: 可指定 'mob6' 或 'mob3' 目标"""
    if verbose:
        print(f"\n{'='*30} 🚀 开始训练单任务 MoE [{task.upper()}, Seed {seed}] {'='*30}")

    np.random.seed(seed)
    rngs = nnx.Rngs(seed)
    net = model.HybridCreditModel_MoE(
        rngs=rngs,
        num_static_features=dataset['num_static_features'],
        cat_vocab_sizes=dataset['cat_vocab_sizes'],
        num_product_types=len(dataset['loan_vocab']),
        num_query_org_types=len(dataset['query_vocab_org']),
        num_query_reason_types=len(dataset['query_vocab_reason']),
        dropout_rate=0.5,
    )
    optimizer = nnx.Optimizer(net, optax.adamw(lr, weight_decay=weight_decay), wrt=nnx.Param)

    y_train = dataset[f'y_train_{task}']
    y_valid = dataset[f'y_valid_{task}']
    y_test = dataset[f'y_test_{task}']

    N_train = len(dataset['X_train_static'])

    best_val_loss = float('inf')
    best_epoch = 0
    best_params = None
    best_bias = None
    wait = 0

    t_start = time.time()
    for ep in range(1, epochs + 1):
        perm = np.random.permutation(N_train)
        for i, s in enumerate(range(0, N_train, batch_size)):
            e = min(s + batch_size, N_train)
            idx = perm[s:e]
            xs = jnp.asarray(dataset['X_train_static'][idx])
            xc = jnp.asarray(dataset['X_train_cat'][idx])
            bl = {k: jnp.asarray(v[idx]) for k, v in dataset['batch_loan_train'].items()}
            bq = {k: jnp.asarray(v[idx]) for k, v in dataset['batch_query_train'].items()}
            y_b = jnp.asarray(y_train[idx], dtype=jnp.float32)

            train_step_moe(net, optimizer, xs, xc, bl, bq, y_b)

        val_loss, val_p = predict_proba_moe(
            net, dataset['X_valid_static'], dataset['X_valid_cat'],
            dataset['batch_loan_valid'], dataset['batch_query_valid'], y_valid
        )
        val_auc, val_ks = common.compute_metrics_cpu(val_p, y_valid)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = ep
            best_params = jax.tree_util.tree_map(lambda x: x.copy(), nnx.state(net, nnx.Param))
            best_bias = net.moe.expert_bias[...].copy()
            wait = 0
            star = " ★ (Best)"
        else:
            wait += 1
            star = ""

        if verbose and (ep % 5 == 0 or ep == 1 or wait == 0):
            print(f"Epoch {ep:3d}/{epochs} | Val Loss={val_loss:.4f} | AUC={val_auc:.4f} KS={val_ks:.4f}{star}")

        if wait >= patience:
            if verbose:
                print(f"早停触发！最佳 Val Loss: {best_val_loss:.4f} @ Epoch {best_epoch}")
            break

    if best_params is not None:
        nnx.update(net, nnx.state(net, nnx.Param), best_params)
        net.moe.expert_bias[...] = best_bias

    t_cost = time.time() - t_start
    test_loss, test_p = predict_proba_moe(
        net, dataset['X_test_static'], dataset['X_test_cat'],
        dataset['batch_loan_test'], dataset['batch_query_test'], y_test
    )
    test_auc, test_ks = common.compute_metrics_cpu(test_p, y_test)

    m = {
        f"auc_{task}": float(test_auc),
        f"ks_{task}": float(test_ks),
        f"loss_{task}": float(test_loss),
        "best_epoch": best_epoch,
        "seed": seed,
        "train_time_sec": t_cost,
    }
    if verbose:
        print(f"✅ 单任务 MoE [{task.upper()}, Seed {seed}] 测试集: AUC={test_auc:.4f} | KS={test_ks:.4f} | 耗时 {t_cost:.1f}s")
    return net, m


def run_10_seeds_moe(
    dataset: Dict[str, Any],
    task: str = "mob6",
    seeds: Optional[List[int]] = None,
    epochs: int = 100,
    lr: float = 5e-4,
    batch_size: int = 512,
    patience: int = 10,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    """多随机种子单任务 MoE 评测"""
    if seeds is None:
        seeds = [777, 42, 100, 2024, 2025, 123, 456, 789, 888, 999]

    results = []
    for s in seeds:
        _, m = train_single_seed_moe(
            dataset=dataset, task=task, seed=s, epochs=epochs, lr=lr, batch_size=batch_size, patience=patience, verbose=True
        )
        results.append(m)

    df_res = pd.DataFrame(results)
    summary = common.compute_stats_summary(df_res, has_rectified=False)
    common.print_experiment_summary(f"Single-Task MoE ({task.upper()})", df_res, summary, has_rectified=False)
    return df_res, summary


# ====================================================================
# 4. 消融实验基准 (HybridCreditModel_Ablation: 无时序特征)
# ====================================================================

@nnx.jit
def train_step_ablation(
    model_net: model.HybridCreditModel_Ablation,
    optimizer: nnx.Optimizer,
    batch_xs: jax.Array,
    batch_xc: jax.Array,
    y6: jax.Array,
    y3: jax.Array,
) -> jax.Array:
    def loss_fn(net):
        l3, l6, a3, ac = net(batch_xs, batch_xc, training=True)
        loss6 = jnp.mean(optax.sigmoid_binary_cross_entropy(l6.squeeze(-1), y6))
        loss3 = jnp.mean(optax.sigmoid_binary_cross_entropy(l3.squeeze(-1), y3))
        return loss6 + loss3 + a3 + ac

    loss, grads = nnx.value_and_grad(loss_fn)(model_net)
    optimizer.update(model_net, grads)
    return loss


@nnx.jit
def eval_forward_ablation(
    model_net: model.HybridCreditModel_Ablation,
    batch_xs: jax.Array,
    batch_xc: jax.Array,
    y6: jax.Array,
    y3: jax.Array,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    l3, l6, _, _ = model_net(batch_xs, batch_xc, training=False)
    p6 = jax.nn.sigmoid(l6.squeeze(-1))
    p3 = jax.nn.sigmoid(l3.squeeze(-1))
    loss6 = jnp.mean(optax.sigmoid_binary_cross_entropy(l6.squeeze(-1), y6))
    loss3 = jnp.mean(optax.sigmoid_binary_cross_entropy(l3.squeeze(-1), y3))
    return loss6 + loss3, p6, p3


def evaluate_full_ablation(
    model_net: model.HybridCreditModel_Ablation,
    X_static: np.ndarray,
    X_cat: np.ndarray,
    y6: np.ndarray,
    y3: np.ndarray,
    eval_bs: int = 1024,
) -> Tuple[float, Dict[str, float]]:
    N = len(X_static)
    all_p6, all_p3 = [], []
    total_losses = []

    for i in range(0, N, eval_bs):
        end = min(i + eval_bs, N)
        xs = jnp.asarray(X_static[i:end])
        xc = jnp.asarray(X_cat[i:end])
        b_y6 = jnp.asarray(y6[i:end], dtype=jnp.float32)
        b_y3 = jnp.asarray(y3[i:end], dtype=jnp.float32)

        t_loss, p6, p3 = eval_forward_ablation(model_net, xs, xc, b_y6, b_y3)
        total_losses.append(float(t_loss) * (end - i))
        all_p6.append(np.asarray(p6, dtype=np.float32))
        all_p3.append(np.asarray(p3, dtype=np.float32))

    p6_arr = np.concatenate(all_p6)
    p3_arr = np.concatenate(all_p3)
    mean_loss = float(sum(total_losses) / N)
    metrics = common.compute_standard_eval_metrics(p3_arr, p6_arr, y3, y6)
    metrics['loss_total'] = mean_loss
    return mean_loss, metrics


def train_single_seed_ablation(
    dataset: Dict[str, Any],
    seed: int = 777,
    epochs: int = 100,
    lr: float = 5e-4,
    batch_size: int = 512,
    patience: int = 10,
    weight_decay: float = 0.1,
    verbose: bool = True,
) -> Tuple[model.HybridCreditModel_Ablation, Dict[str, float]]:
    """训练序列置空消融基线模型"""
    if verbose:
        print(f"\n{'='*30} 🚀 开始训练消融基准 (Ablation, 无时序) [Seed {seed}] {'='*30}")

    rngs = nnx.Rngs(seed)
    net = model.HybridCreditModel_Ablation(
        rngs=rngs,
        num_static_features=dataset['num_static_features'],
        cat_vocab_sizes=dataset['cat_vocab_sizes'],
    )
    optimizer = nnx.Optimizer(net, optax.adamw(lr, weight_decay=weight_decay), wrt=nnx.Param)

    X_train_static = dataset['X_train_static']
    X_train_cat = dataset['X_train_cat']
    y_train_mob6 = dataset['y_train_mob6']
    y_train_mob3 = dataset['y_train_mob3']

    N_train = len(X_train_static)

    best_val_loss = float('inf')
    best_epoch = 0
    best_params = None
    wait = 0

    t_start = time.time()
    for ep in range(1, epochs + 1):
        perm = np.random.RandomState(seed + ep).permutation(N_train)

        for i, s in enumerate(range(0, N_train, batch_size)):
            e = min(s + batch_size, N_train)
            idx = perm[s:e]
            xs = jnp.asarray(X_train_static[idx])
            xc = jnp.asarray(X_train_cat[idx])
            y6_b = jnp.asarray(y_train_mob6[idx], dtype=jnp.float32)
            y3_b = jnp.asarray(y_train_mob3[idx], dtype=jnp.float32)
            train_step_ablation(net, optimizer, xs, xc, y6_b, y3_b)

        val_loss, val_m = evaluate_full_ablation(
            net, dataset['X_valid_static'], dataset['X_valid_cat'],
            dataset['y_valid_mob6'], dataset['y_valid_mob3']
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = ep
            best_params = jax.tree_util.tree_map(lambda x: x.copy(), nnx.state(net, nnx.Param))
            wait = 0
            star = " ★ (Best)"
        else:
            wait += 1
            star = ""

        if verbose and (ep % 5 == 0 or ep == 1 or wait == 0):
            print(f"Epoch {ep:3d}/{epochs} | Val Loss={val_loss:.4f} | MOB6 AUC={val_m['auc_mob6']:.4f} | MOB3 AUC={val_m['auc_mob3']:.4f}{star}")

        if wait >= patience:
            if verbose:
                print(f"早停触发！最佳 Val Loss: {best_val_loss:.4f} @ Epoch {best_epoch}")
            break

    if best_params is not None:
        nnx.update(net, nnx.state(net, nnx.Param), best_params)

    t_cost = time.time() - t_start
    _, test_m = evaluate_full_ablation(
        net, dataset['X_test_static'], dataset['X_test_cat'],
        dataset['y_test_mob6'], dataset['y_test_mob3']
    )
    test_m['best_epoch'] = best_epoch
    test_m['train_time_sec'] = t_cost
    test_m['seed'] = seed

    if verbose:
        print(f"✅ 消融基线 (无时序) [Seed {seed}] 测试集: MOB6 AUC={test_m['auc_mob6']:.4f} | KS={test_m['ks_mob6']:.4f} | 耗时 {t_cost:.1f}s")
    return net, test_m


def run_10_seeds_ablation(
    dataset: Dict[str, Any],
    seeds: Optional[List[int]] = None,
    epochs: int = 100,
    lr: float = 5e-4,
    batch_size: int = 512,
    patience: int = 10,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    """多随机种子消融实验"""
    if seeds is None:
        seeds = [777, 42, 100, 2024, 2025, 123, 456, 789, 888, 999]

    results = []
    for s in seeds:
        _, m = train_single_seed_ablation(
            dataset=dataset, seed=s, epochs=epochs, lr=lr, batch_size=batch_size, patience=patience, verbose=True
        )
        results.append(m)

    df_res = pd.DataFrame(results)
    summary = common.compute_stats_summary(df_res, has_rectified=False)
    common.print_experiment_summary("Ablation Model (No Sequences, Dense + Embed Only)", df_res, summary, has_rectified=False)
    return df_res, summary


# ====================================================================
# 5. 模型分发与基准横向大比武 (Benchmark)
# ====================================================================

MODEL_DISPATCHER = {
    "esmm": run_10_seeds_esmm,
    "moe_mob6": lambda dataset, seeds=None, **kw: run_10_seeds_moe(dataset=dataset, task="mob6", seeds=seeds, **kw),
    "moe_mob3": lambda dataset, seeds=None, **kw: run_10_seeds_moe(dataset=dataset, task="mob3", seeds=seeds, **kw),
    "ablation": run_10_seeds_ablation,
}


def run_model_experiment(
    model_name: str,
    dataset: Dict[str, Any],
    seeds: Optional[List[int]] = None,
    epochs: int = 100,
    lr: float = 5e-4,
    batch_size: int = 512,
    patience: int = 10,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    """运行指定名称的模型实验"""
    if model_name not in MODEL_DISPATCHER:
        raise ValueError(f"未知模型名称: {model_name}. 可选: {list(MODEL_DISPATCHER.keys())}")
    return MODEL_DISPATCHER[model_name](
        dataset=dataset, seeds=seeds, epochs=epochs, lr=lr, batch_size=batch_size, patience=patience
    )


def run_benchmark(
    dataset: Dict[str, Any],
    seeds: Optional[List[int]] = None,
    epochs: int = 100,
    lr: float = 5e-4,
    batch_size: int = 512,
    patience: int = 10,
) -> pd.DataFrame:
    """
    一键运行全量基准大比武 (Benchmark):
    横向对比 4 种方案:
    1. MESM (Proposed ESMM + TCN + DeepSeek MoE + Hard Monotonic)
    2. Single-Task MoE (MOB6)
    3. Single-Task MoE (MOB3)
    4. Ablation (No Sequences)
    生成符合硕士学位论文标准的总汇报表。
    """
    if seeds is None:
        seeds = [777, 42, 100]

    bench_records = []
    print("\n" + "=" * 96)
    print("🏁 启动 MESM 全模型学术基准大比武 (Comprehensive Benchmark Suite)")
    print(f"   随机种子列表: {seeds} | 轮数: {epochs} | 批大小: {batch_size}")
    print("=" * 96)

    # 1. MESM
    print("\n>>> [1/4] 正在评测 MESM (本文提出模型)...")
    df_esmm, sum_esmm = run_10_seeds_esmm(dataset, seeds=seeds, epochs=epochs, lr=lr, batch_size=batch_size, patience=patience)
    bench_records.append({
        "Model": "MESM (Proposed Full)",
        "MOB6 AUC": f"{sum_esmm['auc_mob6']['mean']:.4f} ± {sum_esmm['auc_mob6']['std']:.4f}",
        "MOB6 KS": f"{sum_esmm['ks_mob6']['mean']:.4f} ± {sum_esmm['ks_mob6']['std']:.4f}",
        "MOB3 AUC": f"{sum_esmm['auc_mob3']['mean']:.4f} ± {sum_esmm['auc_mob3']['std']:.4f}",
        "MOB3 KS": f"{sum_esmm['ks_mob3']['mean']:.4f} ± {sum_esmm['ks_mob3']['std']:.4f}",
        "VR (%)": f"{sum_esmm['vr_pct']['mean']:.2f}%",
        "Monotonic": "硬单调 (100% 成立)",
    })

    # 2. Single Task MOB6
    print("\n>>> [2/4] 正在评测 Single-Task MoE (MOB6)...")
    df_m6, sum_m6 = run_10_seeds_moe(dataset, task="mob6", seeds=seeds, epochs=epochs, lr=lr, batch_size=batch_size, patience=patience)
    bench_records.append({
        "Model": "Single-Task MoE (MOB6)",
        "MOB6 AUC": f"{sum_m6['auc_mob6']['mean']:.4f} ± {sum_m6['auc_mob6']['std']:.4f}",
        "MOB6 KS": f"{sum_m6['ks_mob6']['mean']:.4f} ± {sum_m6['ks_mob6']['std']:.4f}",
        "MOB3 AUC": "—",
        "MOB3 KS": "—",
        "VR (%)": "—",
        "Monotonic": "无约束",
    })

    # 3. Single Task MOB3
    print("\n>>> [3/4] 正在评测 Single-Task MoE (MOB3)...")
    df_m3, sum_m3 = run_10_seeds_moe(dataset, task="mob3", seeds=seeds, epochs=epochs, lr=lr, batch_size=batch_size, patience=patience)
    bench_records.append({
        "Model": "Single-Task MoE (MOB3)",
        "MOB6 AUC": "—",
        "MOB6 KS": "—",
        "MOB3 AUC": f"{sum_m3['auc_mob3']['mean']:.4f} ± {sum_m3['auc_mob3']['std']:.4f}",
        "MOB3 KS": f"{sum_m3['ks_mob3']['mean']:.4f} ± {sum_m3['ks_mob3']['std']:.4f}",
        "VR (%)": "—",
        "Monotonic": "无约束",
    })

    # 4. Ablation (No Sequences)
    print("\n>>> [4/4] 正在评测 Ablation Model (无时序)...")
    df_abl, sum_abl = run_10_seeds_ablation(dataset, seeds=seeds, epochs=epochs, lr=lr, batch_size=batch_size, patience=patience)
    bench_records.append({
        "Model": "Ablation (Dense+Embed Only)",
        "MOB6 AUC": f"{sum_abl['auc_mob6']['mean']:.4f} ± {sum_abl['auc_mob6']['std']:.4f}",
        "MOB6 KS": f"{sum_abl['ks_mob6']['mean']:.4f} ± {sum_abl['ks_mob6']['std']:.4f}",
        "MOB3 AUC": f"{sum_abl['auc_mob3']['mean']:.4f} ± {sum_abl['auc_mob3']['std']:.4f}",
        "MOB3 KS": f"{sum_abl['ks_mob3']['mean']:.4f} ± {sum_abl['ks_mob3']['std']:.4f}",
        "VR (%)": f"{sum_abl['vr_pct']['mean']:.2f}%",
        "Monotonic": "硬单调 (100% 成立)",
    })

    df_bench = pd.DataFrame(bench_records)
    print("\n" + "=" * 96)
    print("🏆 【硕士学位论文总览对比表 (Master's Thesis Benchmark Summary Table)】")
    print("=" * 96)
    print(df_bench.to_string(index=False))
    print("=" * 96 + "\n")
    return df_bench
