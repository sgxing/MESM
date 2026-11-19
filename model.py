import os
os.environ['JAX_PLATFORM_NAME'] = 'mps'
os.environ['JAX_MPS_ASYNC_DISPATCH'] = '0'
from typing import Dict, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
from flax import nnx
from flax.nnx.nn.linear import default_embed_init
import optax
import numpy as np


# ====================================================================
# 1. 确定性 Embedding 层 (MPS / Metal 硬件加速无分叉保证)
# ====================================================================

@jax.custom_vjp
def safe_embedding_lookup(emb_matrix: jax.Array, indices: jax.Array) -> jax.Array:
    """前向 = 纯 gather (零额外开销); 反向 = matmul 归约 (Apple Silicon MPS 上梯度完全确定)"""
    return emb_matrix[indices]


def _fwd_lookup(emb_matrix: jax.Array, indices: jax.Array):
    return emb_matrix[indices], (emb_matrix.shape, indices)


def _bwd_lookup(aux, g):
    emb_shape, indices = aux
    V, D = emb_shape
    oh = jax.nn.one_hot(indices, V)               # (..., V)
    N = g.size // D
    d_emb = oh.reshape(N, V).T @ g.reshape(N, D)  # (V, D) 纯 matmul 替代原子 scatter-add
    return d_emb, None


safe_embedding_lookup.defvjp(_fwd_lookup, _bwd_lookup)


class DeterministicEmbed(nnx.Module):
    """
    可复现确定版 Embedding:
    使用自定义 VJP 规避 GPU / MPS 上 scatter_add 浮点乱序累加导致的不可复现问题。
    """
    def __init__(
        self,
        num_embeddings: int,
        features: int,
        *,
        rngs: nnx.Rngs,
        param_dtype: jnp.dtype = jnp.float32,
        embedding_init=default_embed_init,
    ):
        super().__init__()
        self.embedding = nnx.Param(
            embedding_init(rngs.params(), (num_embeddings, features), param_dtype)
        )
        self.num_embeddings = num_embeddings
        self.features = features

    def __call__(self, indices: jax.Array) -> jax.Array:
        return safe_embedding_lookup(self.embedding, indices.astype(jnp.int32))


# ====================================================================
# 2. 征信贷款序列因果卷积编码器 (Loan TCN Encoder)
# ====================================================================

class LoanTCNEncoder(nnx.Module):
    """
    贷款历史序列编码器 (月度池化 + 扩张因果时序卷积网络 TCN + 时间衰减注意力池化)
    输入: (B, T, P) 月度池化贷款序列，每月最多 P 笔贷款
    结构: Product Embed + Amount(log1p) + Recency(log1p) + Time Delta -> Proj -> 月内池化 -> 多层因果卷积残差块 -> 时间衰减加权池化
    """
    def __init__(
        self,
        rngs: nnx.Rngs,
        num_product_types: int,
        tcn_hidden: int = 64,
        tcn_num_layers: int = 4,
        kernel_size: int = 3,
        base_dilation: int = 2,
        dropout_rate: float = 0.5,
        pool_mode: str = 'max',
        use_bf16: bool = False,
    ):
        super().__init__()
        self.use_bf16 = use_bf16
        self.compute_dtype = jnp.bfloat16 if use_bf16 else jnp.float32
        self.pool_mode = pool_mode

        self.product_embed = DeterministicEmbed(num_product_types, max(4, tcn_hidden // 4), rngs=rngs)
        self.input_proj = nnx.Linear(max(4, tcn_hidden // 4) + 3, tcn_hidden, rngs=rngs)

        self.kernel_size = kernel_size
        self.base_dilation = base_dilation
        convs = []
        for i in range(tcn_num_layers):
            convs.append(
                nnx.Conv(
                    tcn_hidden,
                    tcn_hidden,
                    (kernel_size,),
                    feature_group_count=1,
                    padding='VALID',
                    kernel_dilation=base_dilation ** i,
                    rngs=rngs,
                )
            )
        self.convs = nnx.List(convs)
        self.empty_embed = nnx.Param(jnp.zeros(tcn_hidden))
        self.dropout = nnx.Dropout(dropout_rate, rngs=rngs)

    def _bf16(self, x: jax.Array, training: bool) -> jax.Array:
        if self.use_bf16 and training:
            return x.astype(self.compute_dtype)
        return x

    def _causal_pad(self, x: jax.Array, k: int, d: int) -> jax.Array:
        """严格因果填充: 仅在序列左侧补零, 杜绝向未来时间步泄露"""
        return jnp.pad(x, ((0, 0), ((k - 1) * d, 0), (0, 0)))

    def __call__(
        self,
        pids: jax.Array,
        amts: jax.Array,
        recs: jax.Array,
        deltas: jax.Array,
        lens: jax.Array,
        nprods: jax.Array,
        training: bool = False,
    ) -> jax.Array:
        B, T, P = pids.shape
        pe = self.product_embed(pids)
        feat = jnp.concatenate([
            pe,
            jnp.log1p(amts[..., None]),
            jnp.log1p(recs[..., None]),
            jnp.repeat(deltas[:, :, None, None], P, 2)
        ], axis=-1)
        feat = self._bf16(feat, training)
        feat = nnx.relu(self.input_proj(feat))

        # 月内池化 (Pool over P items within each month)
        pmask = (jnp.arange(P)[None, None, :] < nprods[:, :, None])
        if self.pool_mode == 'max':
            feat = jnp.where(pmask[..., None], feat, -jnp.inf)
            x = feat.max(2)
            x = jnp.where(jnp.isinf(x), 0.0, x)
        else:
            feat = feat * pmask.astype(jnp.float32)[..., None]
            x = feat.sum(2) / jnp.maximum(nprods[:, :, None].astype(jnp.float32), 1.0)

        # 序列时序 Mask
        smask = (jnp.arange(T)[None, :] < lens[:, None]).astype(jnp.float32)[:, :, None]
        x = x * smask

        # 逐层因果扩张卷积 + 残差连接
        for i, conv in enumerate(self.convs):
            r = x
            d = self.base_dilation ** i
            xp = self._causal_pad(x, self.kernel_size, d)
            x = conv(self._bf16(xp, training))
            x = x.astype(jnp.float32)
            x = nnx.relu(x)
            x = self.dropout(x, deterministic=not training)
            if r.shape == x.shape:
                x = x + r
            x = x * smask

        # 时间衰减注意力池化 (越近期的记录权重越高)
        month_recs = recs[:, :, 0]
        tau = 12.0
        attn_weights = jnp.exp(-month_recs / tau)
        attn_weights = attn_weights * smask.squeeze(-1)
        attn_weights = attn_weights / (attn_weights.sum(1, keepdims=True) + 1e-8)
        gap = (x * attn_weights[:, :, None]).sum(1)

        # 对空序列借款人使用可学习占位符 empty_embed
        empty = (smask.sum(1).squeeze(-1) == 0).astype(jnp.float32)[:, None]
        gap = (1.0 - empty) * gap + empty * self.empty_embed[...]
        return self.dropout(gap, deterministic=not training)


# ====================================================================
# 3. 征信查询序列编码器 (Query Encoder)
# ====================================================================

class QueryEncoder(nnx.Module):
    """
    征信查询记录时序编码器 (机构类型嵌入 + 查询原因嵌入 + 事件数 + 距今月数 + 扩张因果卷积)
    """
    def __init__(
        self,
        rngs: nnx.Rngs,
        num_org_types: int,
        num_reason_types: int,
        embed_dim: int = 8,
        hidden: int = 32,
        num_layers: int = 4,
        kernel_size: int = 3,
        base_dilation: int = 2,
        dropout_rate: float = 0.3,
        pool_mode: str = 'max',
    ):
        super().__init__()
        self.pool_mode = pool_mode
        self.org_embed = DeterministicEmbed(num_org_types, embed_dim, rngs=rngs)
        self.reason_embed = DeterministicEmbed(num_reason_types, embed_dim, rngs=rngs)
        self.input_proj = nnx.Linear(embed_dim * 2 + 3, hidden, rngs=rngs)

        self.kernel_size = kernel_size
        self.base_dilation = base_dilation
        convs = []
        for i in range(num_layers):
            convs.append(
                nnx.Conv(
                    hidden,
                    hidden,
                    (kernel_size,),
                    feature_group_count=1,
                    padding='VALID',
                    kernel_dilation=base_dilation ** i,
                    rngs=rngs,
                )
            )
        self.convs = nnx.List(convs)
        self.empty_embed = nnx.Param(jnp.zeros(hidden))
        self.dropout = nnx.Dropout(dropout_rate, rngs=rngs)

    def _causal_pad(self, x: jax.Array, k: int, d: int) -> jax.Array:
        return jnp.pad(x, ((0, 0), ((k - 1) * d, 0), (0, 0)))

    def __call__(
        self,
        org_ids: jax.Array,
        reason_ids: jax.Array,
        event_counts: jax.Array,
        recs: jax.Array,
        deltas: jax.Array,
        lens: jax.Array,
        npermonth: jax.Array,
        training: bool = False,
    ) -> jax.Array:
        B, T, P = org_ids.shape

        oe = self.org_embed(jnp.clip(org_ids.astype(jnp.int32), 0, None))
        re = self.reason_embed(jnp.clip(reason_ids.astype(jnp.int32), 0, None))
        feat = jnp.concatenate([
            oe,
            re,
            jnp.log1p(event_counts[..., None]),
            jnp.log1p(recs[..., None]),
            jnp.repeat(deltas[:, :, None, None], P, 2)
        ], axis=-1)
        x = nnx.relu(self.input_proj(feat))

        pmask = (jnp.arange(P)[None, None, :] < npermonth[:, :, None])
        if self.pool_mode == 'max':
            x = jnp.where(pmask[..., None], x, -jnp.inf)
            x = x.max(2)
            x = jnp.where(jnp.isinf(x), 0.0, x)
        else:
            x = x * pmask.astype(jnp.float32)[..., None]
            x = x.sum(2) / jnp.maximum(npermonth[:, :, None].astype(jnp.float32), 1.0)

        smask = (jnp.arange(T)[None, :] < lens[:, None]).astype(jnp.float32)[:, :, None]
        x = x * smask

        for i, conv in enumerate(self.convs):
            r = x
            d = self.base_dilation ** i
            xp = self._causal_pad(x, self.kernel_size, d)
            x = nnx.relu(conv(xp))
            x = self.dropout(x, deterministic=not training)
            if r.shape == x.shape:
                x = x + r
            x = x * smask

        month_recs = recs[:, :, 0]
        tau = 12.0
        attn_weights = jnp.exp(-month_recs / tau)
        attn_weights = attn_weights * smask.squeeze(-1)
        attn_weights = attn_weights / (attn_weights.sum(1, keepdims=True) + 1e-8)
        gap = (x * attn_weights[:, :, None]).sum(1)

        empty = (smask.sum(1).squeeze(-1) == 0).astype(jnp.float32)[:, None]
        gap = (1.0 - empty) * gap + empty * self.empty_embed[...]
        return self.dropout(gap, deterministic=not training)


# ====================================================================
# 4. DeepSeek-V3 风格单任务稀疏专家混合 (SparseMoE)
# ====================================================================

class SparseMoE(nnx.Module):
    """
    单任务稀疏 MoE 模块:
    1. 共享专家 (Shared Expert): 始终激活, 提取信贷宏观总体基线特征；
    2. 稀疏专家 (Sparse Experts): Top-K 动态路由, 分群细化风险；
    3. DeepSeek-V3 动态偏置 (Aux-loss-free):
       偏置仅参与 Top-K 选路决策, 不扭曲可微门控权重的反向传播梯度。
    """
    def __init__(
        self,
        rngs: nnx.Rngs,
        input_dim: int,
        expert_hidden: int,
        num_experts: int = 6,
        top_k: int = 2,
        dropout_rate: float = 0.3,
        lb_bias_alpha: float = 0.001,
        lb_bias_gamma: float = 0.0,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.lb_bias_alpha = lb_bias_alpha
        self.lb_bias_gamma = lb_bias_gamma
        self.expert_bias = nnx.Variable(jnp.zeros(num_experts))

        # Shared Expert (始终激活)
        self.shared_w1 = nnx.Linear(input_dim, expert_hidden, rngs=rngs)
        self.shared_w2 = nnx.Linear(expert_hidden, expert_hidden // 2, rngs=rngs)
        self.shared_out = nnx.Linear(expert_hidden // 2, 1, rngs=rngs)

        # 2 层非线性 Gating Network
        gate_hidden = max(32, num_experts * 4)
        self.gate_w1 = nnx.Linear(input_dim, gate_hidden, rngs=rngs)
        self.gate_w2 = nnx.Linear(gate_hidden, num_experts, rngs=rngs, use_bias=False)

        # Sparse Experts
        w1_list, w2_list, out_list = [], [], []
        for _ in range(num_experts):
            w1_list.append(nnx.Linear(input_dim, expert_hidden, rngs=rngs))
            w2_list.append(nnx.Linear(expert_hidden, expert_hidden // 2, rngs=rngs))
            out_list.append(nnx.Linear(expert_hidden // 2, 1, rngs=rngs))
        self.expert_w1 = nnx.List(w1_list)
        self.expert_w2 = nnx.List(w2_list)
        self.expert_out = nnx.List(out_list)

        self.dropout = nnx.Dropout(dropout_rate, rngs=rngs)

    def _shared_forward(self, x: jax.Array, training: bool) -> jax.Array:
        h = nnx.relu(self.shared_w1(x))
        h = self.dropout(h, deterministic=not training)
        h = nnx.relu(self.shared_w2(h))
        h = self.dropout(h, deterministic=not training)
        return self.shared_out(h)

    def _expert_forward(self, e_idx: int, x: jax.Array, training: bool) -> jax.Array:
        h = nnx.relu(self.expert_w1[e_idx](x))
        h = self.dropout(h, deterministic=not training)
        h = nnx.relu(self.expert_w2[e_idx](h))
        h = self.dropout(h, deterministic=not training)
        return self.expert_out[e_idx](h)

    def __call__(self, x: jax.Array, training: bool = False) -> Tuple[jax.Array, jax.Array]:
        B = x.shape[0]

        # ── 2 层非线性门控 ──
        gate_h = nnx.relu(self.gate_w1(x))
        gate_logits = self.gate_w2(gate_h)  # (B, num_experts)

        # ── DeepSeek-V3 风格 Sigmoid Top-K 选路与偏置 ──
        gate_probs = jax.nn.sigmoid(gate_logits)
        _, topk_idx = jax.lax.top_k(gate_probs + self.expert_bias[...], self.top_k)
        topk_vals = jnp.take_along_axis(gate_probs, topk_idx, axis=-1)
        topk_vals = topk_vals / (topk_vals.sum(axis=-1, keepdims=True) + 1e-8)

        # ── Shared Expert (始终激活) ──
        shared_out = self._shared_forward(x, training)  # (B, 1)

        # ── Sparse Experts 加权聚合 ──
        expert_weights_all = jnp.zeros((B, self.num_experts), dtype=x.dtype)
        expert_weights_all = expert_weights_all.at[
            jnp.arange(B)[:, None], topk_idx
        ].set(topk_vals)

        sparse_outputs = []
        for e_idx in range(self.num_experts):
            w = expert_weights_all[:, e_idx]
            out = self._expert_forward(e_idx, x, training)
            sparse_outputs.append(out * w[:, None])

        sparse_out = sum(sparse_outputs)  # (B, 1)
        output = shared_out + sparse_out  # (B, 1)

        # ── 训练时更新 DeepSeek-V3 动态偏置 (aux-loss-free 负载均衡) ──
        if training and self.lb_bias_alpha > 0.0:
            load = jnp.zeros(self.num_experts, dtype=jnp.float32)
            for e in range(self.num_experts):
                load = load.at[e].set((topk_idx == e).sum().astype(jnp.float32))
            mean_load = jnp.mean(load)
            target = mean_load * (1.0 + self.lb_bias_gamma)
            delta = self.lb_bias_alpha * jnp.sign(target - load)
            self.expert_bias[...] += delta

        aux_loss = jnp.array(0.0, dtype=jnp.float32)
        return output, aux_loss


# ====================================================================
# 5. ESMM 双塔硬单调稀疏专家混合 (SparseMoE_ESMM)
# ====================================================================

class SparseMoE_ESMM(nnx.Module):
    """
    ESMM 双塔稀疏 MoE 架构 (Ma et al., SIGIR 2018 跨期信贷迁移):
    - 共享基础骨干与专家池: 1 个共享专家 + E 个稀疏专家
    - 双塔分流决策:
      塔 1 (MOB3): 短期违约概率预测, 独立门控 gate_mob3 + 专属输出头
      塔 2 (Delta): 增量逾期对数几率预测, 独立门控 gate_cond + 专属输出头
    - 结合 softplus 机制, 构造硬单调递增性: logits_mob6 = logits_mob3 + softplus(logits_delta)
    """
    def __init__(
        self,
        rngs: nnx.Rngs,
        input_dim: int,
        expert_hidden: int,
        num_experts: int = 6,
        top_k: int = 2,
        dropout_rate: float = 0.3,
        lb_bias_alpha: float = 0.001,
        lb_bias_gamma: float = 0.0,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.lb_bias_alpha = lb_bias_alpha
        self.lb_bias_gamma = lb_bias_gamma

        # 双塔各自维护独立的负载均衡偏置 (Variable 状态, 不参与梯度更新)
        self.expert_bias_mob3 = nnx.Variable(jnp.zeros(num_experts))
        self.expert_bias_cond = nnx.Variable(jnp.zeros(num_experts))

        # 共用的 Shared Expert 特征抽取
        self.shared_w1 = nnx.Linear(input_dim, expert_hidden, rngs=rngs)
        self.shared_w2 = nnx.Linear(expert_hidden, expert_hidden // 2, rngs=rngs)

        # 两个塔专属的 Shared Expert 输出头
        self.shared_out_mob3 = nnx.Linear(expert_hidden // 2, 1, rngs=rngs)
        self.shared_out_cond = nnx.Linear(expert_hidden // 2, 1, rngs=rngs)

        # 共享的 Sparse Experts
        w1_list, w2_list, out_list = [], [], []
        for _ in range(num_experts):
            w1_list.append(nnx.Linear(input_dim, expert_hidden, rngs=rngs))
            w2_list.append(nnx.Linear(expert_hidden, expert_hidden // 2, rngs=rngs))
            out_list.append(nnx.Linear(expert_hidden // 2, 1, rngs=rngs))
        self.expert_w1 = nnx.List(w1_list)
        self.expert_w2 = nnx.List(w2_list)
        self.expert_out = nnx.List(out_list)

        # 两个独立的门控网络
        gate_hidden = max(32, num_experts * 4)
        self.gate_mob3_w1 = nnx.Linear(input_dim, gate_hidden, rngs=rngs)
        self.gate_mob3_w2 = nnx.Linear(gate_hidden, num_experts, rngs=rngs, use_bias=False)
        self.gate_cond_w1 = nnx.Linear(input_dim, gate_hidden, rngs=rngs)
        self.gate_cond_w2 = nnx.Linear(gate_hidden, num_experts, rngs=rngs, use_bias=False)

        self.dropout = nnx.Dropout(dropout_rate, rngs=rngs)

    def _shared_features(self, x: jax.Array, training: bool) -> jax.Array:
        h = nnx.relu(self.shared_w1(x))
        h = self.dropout(h, deterministic=not training)
        h = nnx.relu(self.shared_w2(h))
        h = self.dropout(h, deterministic=not training)
        return h

    def _expert_forward(self, e_idx: int, x: jax.Array, training: bool) -> jax.Array:
        h = nnx.relu(self.expert_w1[e_idx](x))
        h = self.dropout(h, deterministic=not training)
        h = nnx.relu(self.expert_w2[e_idx](h))
        h = self.dropout(h, deterministic=not training)
        return self.expert_out[e_idx](h)

    def _gate_forward(
        self,
        x: jax.Array,
        gate_w1: nnx.Linear,
        gate_w2: nnx.Linear,
    ) -> jax.Array:
        h = nnx.relu(gate_w1(x))
        return gate_w2(h)

    def _tower_forward(
        self,
        x: jax.Array,
        gate_w1: nnx.Linear,
        gate_w2: nnx.Linear,
        shared_out_layer: nnx.Linear,
        expert_bias: nnx.Variable,
        training: bool,
    ) -> Tuple[jax.Array, jax.Array]:
        """单个塔的完整前向:
        门控 → Top-K 路由 (含 DeepSeek-V3 动态偏置) →
        Shared Expert 输出 + Sparse Experts 加权聚合"""
        B = x.shape[0]
        gate_logits = self._gate_forward(x, gate_w1, gate_w2)  # (B, num_experts)

        gate_probs = jax.nn.sigmoid(gate_logits)
        _, topk_idx = jax.lax.top_k(gate_probs + expert_bias[...], self.top_k)
        topk_vals = jnp.take_along_axis(gate_probs, topk_idx, axis=-1)
        topk_vals = topk_vals / (topk_vals.sum(axis=-1, keepdims=True) + 1e-8)

        shared_feat = self._shared_features(x, training)   # (B, expert_hidden//2)
        shared_out = shared_out_layer(shared_feat)         # (B, 1)

        expert_weights_all = jnp.zeros((B, self.num_experts), dtype=x.dtype)
        expert_weights_all = expert_weights_all.at[
            jnp.arange(B)[:, None], topk_idx].set(topk_vals)
        sparse_outputs = []
        for e_idx in range(self.num_experts):
            w = expert_weights_all[:, e_idx]
            out = self._expert_forward(e_idx, x, training)
            sparse_outputs.append(out * w[:, None])
        sparse_out = sum(sparse_outputs)

        output = shared_out + sparse_out  # (B, 1)

        # ── DeepSeek-V3 动态偏置更新 (aux-loss-free 负载均衡) ──
        #    b_i ← b_i − α·sign(l_i − l̄)   (l̄ = mean_load·(1+γ), γ=0 即论文原式)
        #    l_i: 当前 batch 路由到专家 i 的样本数 (top-k 计数)
        if training and self.lb_bias_alpha > 0.0:
            load = jnp.zeros(self.num_experts, dtype=jnp.float32)
            for e in range(self.num_experts):
                load = load.at[e].set((topk_idx == e).sum().astype(jnp.float32))
            mean_load = jnp.mean(load)
            target = mean_load * (1.0 + self.lb_bias_gamma)
            delta = self.lb_bias_alpha * jnp.sign(target - load)
            expert_bias[...] += delta

        aux_loss = jnp.array(0.0, dtype=jnp.float32)
        return output, aux_loss

    def __call__(
        self,
        x: jax.Array,
        training: bool = False,
    ) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        out_mob3, aux_mob3 = self._tower_forward(
            x, self.gate_mob3_w1, self.gate_mob3_w2, self.shared_out_mob3,
            self.expert_bias_mob3, training
        )
        out_cond, aux_cond = self._tower_forward(
            x, self.gate_cond_w1, self.gate_cond_w2, self.shared_out_cond,
            self.expert_bias_cond, training
        )
        return out_mob3, aux_mob3, out_cond, aux_cond


# ====================================================================
# 6. 单任务模型整合 (HybridCreditModel_MoE)
# ====================================================================

class HybridCreditModel_MoE(nnx.Module):
    """
    单任务混合征信风控网络:
    特征结构: [Dense 静态数值 (含手工序列聚合特征) + 字符类别 Embedding + 贷款 TCN 序列隐表征 + 查询月池化隐表征]
    融合策略: 拼接进入 DeepSeek-V3 稀疏 MoE 模块进行单任务打分。
    """
    def __init__(
        self,
        rngs: nnx.Rngs,
        num_static_features: int,
        cat_vocab_sizes: Dict[str, int],
        num_product_types: int,
        num_query_org_types: int,
        num_query_reason_types: int,
        static_hidden: int = 64,
        embed_dim: int = 16,
        embed_hidden: int = 64,
        tcn_hidden: int = 64,
        tcn_num_layers: int = 4,
        tcn_kernel_size: int = 3,
        tcn_base_dilation: int = 2,
        query_hidden: int = 32,
        query_num_layers: int = 2,
        num_experts: int = 6,
        top_k: int = 2,
        expert_hidden: int = 128,
        dropout_rate: float = 0.5,
        pool_mode: str = 'max',
        use_bf16: bool = False,
        moe_lb_bias_alpha: float = 0.001,
        moe_lb_bias_gamma: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        self.use_bf16 = use_bf16
        self.cat_embeddings = nnx.List([
            DeterministicEmbed(vocab_size, embed_dim, rngs=rngs)
            for vocab_size in cat_vocab_sizes.values()
        ])
        total_fields = len(cat_vocab_sizes)
        self.input_dim = total_fields * embed_dim + num_static_features
        self.cat_in = nnx.Linear(self.input_dim, embed_hidden, rngs=rngs)
        self.dropout = nnx.Dropout(dropout_rate, rngs=rngs)
        self.cat_out = nnx.Linear(embed_hidden, embed_hidden // 2, rngs=rngs)

        # 贷款 TCN 时序分支
        self.loan_tcn_encoder = LoanTCNEncoder(
            rngs=rngs,
            num_product_types=num_product_types,
            tcn_hidden=tcn_hidden,
            tcn_num_layers=tcn_num_layers,
            kernel_size=tcn_kernel_size,
            base_dilation=tcn_base_dilation,
            dropout_rate=dropout_rate,
            pool_mode=pool_mode,
            use_bf16=use_bf16,
        )

        # 查询时序分支
        self.query_encoder = QueryEncoder(
            rngs=rngs,
            num_org_types=num_query_org_types,
            num_reason_types=num_query_reason_types,
            hidden=query_hidden,
            num_layers=query_num_layers,
            dropout_rate=dropout_rate,
            pool_mode=pool_mode,
        )

        # MoE 融合主干
        fusion_in = embed_hidden // 2 + tcn_hidden + query_hidden
        self.moe = SparseMoE(
            rngs=rngs,
            input_dim=fusion_in,
            expert_hidden=expert_hidden,
            num_experts=num_experts,
            top_k=top_k,
            dropout_rate=dropout_rate,
            lb_bias_alpha=moe_lb_bias_alpha,
            lb_bias_gamma=moe_lb_bias_gamma,
        )

    def _shared_branch(self, x_static: jax.Array, x_cat: jax.Array, training: bool) -> jax.Array:
        cat_emb = jnp.concatenate([
            emb(x_cat[:, i].astype(jnp.int32)) for i, emb in enumerate(self.cat_embeddings)
        ], axis=-1)
        common_in = jnp.concatenate([x_static, cat_emb], axis=-1)
        h = nnx.relu(self.cat_in(common_in))
        h = self.dropout(h, deterministic=not training)
        h = nnx.relu(self.cat_out(h))
        return h

    def __call__(
        self,
        x_static: jax.Array,
        x_cat: jax.Array,
        pids: jax.Array,
        amts: jax.Array,
        recs: jax.Array,
        deltas: jax.Array,
        lens: jax.Array,
        nprods: jax.Array,
        q_org_ids: jax.Array,
        q_rsn_ids: jax.Array,
        q_counts: jax.Array,
        q_recs: jax.Array,
        q_deltas: jax.Array,
        q_lens: jax.Array,
        q_npermonth: jax.Array,
        training: bool = False,
    ) -> Tuple[jax.Array, jax.Array]:
        d = self._shared_branch(x_static, x_cat, training)
        t = self.loan_tcn_encoder(pids, amts, recs, deltas, lens, nprods, training)
        q = self.query_encoder(q_org_ids, q_rsn_ids, q_counts, q_recs, q_deltas, q_lens, q_npermonth, training)
        fused = jnp.concatenate([d, t, q], axis=-1)
        output, aux_loss = self.moe(fused, training=training)
        return output, aux_loss


# ====================================================================
# 7. 硬单调多任务模型整合 (HybridCreditModel_ESMM)
# ====================================================================

class HybridCreditModel_ESMM(nnx.Module):
    """
    硕士论文核心模型: MESM (Mixture-of-Experts Entire Space Multi-Task Net)
    融合特征嵌入与时序因果建模，并在对数几率 (logit) 空间施加硬单调增量约束:
      logits_mob6 = logits_mob3 + softplus(logits_delta)
    数学上恒满足:
      logits_mob6 >= logits_mob3
      P(MOB6=1) >= P(MOB3=1)
    彻底杜绝跨期倒挂 (Monotonic Violation Rate 恒等于 0.00%)。
    """
    def __init__(
        self,
        rngs: nnx.Rngs,
        num_static_features: int,
        cat_vocab_sizes: Dict[str, int],
        num_product_types: int,
        num_query_org_types: int,
        num_query_reason_types: int,
        static_hidden: int = 64,
        embed_dim: int = 16,
        embed_hidden: int = 64,
        tcn_hidden: int = 64,
        tcn_num_layers: int = 4,
        tcn_kernel_size: int = 3,
        tcn_base_dilation: int = 2,
        query_hidden: int = 32,
        query_num_layers: int = 2,
        num_experts: int = 6,
        top_k: int = 2,
        expert_hidden: int = 128,
        dropout_rate: float = 0.5,
        pool_mode: str = 'max',
        use_bf16: bool = False,
        moe_lb_bias_alpha: float = 0.001,
        moe_lb_bias_gamma: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        self.use_bf16 = use_bf16
        self.cat_embeddings = nnx.List([
            DeterministicEmbed(vocab_size, embed_dim, rngs=rngs)
            for vocab_size in cat_vocab_sizes.values()
        ])
        total_fields = len(cat_vocab_sizes)
        self.input_dim = total_fields * embed_dim + num_static_features
        self.cat_in = nnx.Linear(self.input_dim, embed_hidden, rngs=rngs)
        self.dropout = nnx.Dropout(dropout_rate, rngs=rngs)
        self.cat_out = nnx.Linear(embed_hidden, embed_hidden // 2, rngs=rngs)

        # 贷款 TCN
        self.loan_tcn_encoder = LoanTCNEncoder(
            rngs=rngs,
            num_product_types=num_product_types,
            tcn_hidden=tcn_hidden,
            tcn_num_layers=tcn_num_layers,
            kernel_size=tcn_kernel_size,
            base_dilation=tcn_base_dilation,
            dropout_rate=dropout_rate,
            pool_mode=pool_mode,
            use_bf16=use_bf16,
        )

        # 查询月池化 TCN
        self.query_encoder = QueryEncoder(
            rngs=rngs,
            num_org_types=num_query_org_types,
            num_reason_types=num_query_reason_types,
            hidden=query_hidden,
            num_layers=query_num_layers,
            dropout_rate=dropout_rate,
            pool_mode=pool_mode,
        )

        # ESMM 双塔 MoE
        fusion_in = embed_hidden // 2 + tcn_hidden + query_hidden
        self.esmm = SparseMoE_ESMM(
            rngs=rngs,
            input_dim=fusion_in,
            expert_hidden=expert_hidden,
            num_experts=num_experts,
            top_k=top_k,
            dropout_rate=dropout_rate,
            lb_bias_alpha=moe_lb_bias_alpha,
            lb_bias_gamma=moe_lb_bias_gamma,
        )

    def _shared_branch(self, x_static: jax.Array, x_cat: jax.Array, training: bool) -> jax.Array:
        cat_emb = jnp.concatenate([
            emb(x_cat[:, i].astype(jnp.int32)) for i, emb in enumerate(self.cat_embeddings)
        ], axis=-1)
        common_in = jnp.concatenate([x_static, cat_emb], axis=-1)
        h = nnx.relu(self.cat_in(common_in))
        h = self.dropout(h, deterministic=not training)
        h = nnx.relu(self.cat_out(h))
        return h

    def __call__(
        self,
        x_static: jax.Array,
        x_cat: jax.Array,
        pids: jax.Array,
        amts: jax.Array,
        recs: jax.Array,
        deltas: jax.Array,
        lens: jax.Array,
        nprods: jax.Array,
        q_org_ids: jax.Array,
        q_rsn_ids: jax.Array,
        q_counts: jax.Array,
        q_recs: jax.Array,
        q_deltas: jax.Array,
        q_lens: jax.Array,
        q_npermonth: jax.Array,
        training: bool = False,
    ) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        d = self._shared_branch(x_static, x_cat, training)
        t = self.loan_tcn_encoder(pids, amts, recs, deltas, lens, nprods, training)
        q = self.query_encoder(q_org_ids, q_rsn_ids, q_counts, q_recs, q_deltas, q_lens, q_npermonth, training)
        fused = jnp.concatenate([d, t, q], axis=-1)

        logits_mob3, aux_mob3, logits_delta, aux_delta = self.esmm(fused, training=training)
        # 硬单调保证: softplus(logits_delta) 恒非负
        z_delta = nnx.softplus(logits_delta)
        logits_mob6 = logits_mob3 + z_delta

        return logits_mob3, logits_mob6, aux_mob3, aux_delta


# ====================================================================
# 8. 消融实验基准模型 (HybridCreditModel_Ablation)
# ====================================================================

class HybridCreditModel_Ablation(nnx.Module):
    """
    消融基线模型: 完全移除贷款时序与查询时序 (仅保留 Dense 静态特征与类别特征 Embedding),
    用于量化评估因果时序卷积模块所带来的增量业务收益与 AUC 提升。
    """
    def __init__(
        self,
        rngs: nnx.Rngs,
        num_static_features: int,
        cat_vocab_sizes: Dict[str, int],
        embed_dim: int = 16,
        embed_hidden: int = 64,
        num_experts: int = 6,
        top_k: int = 2,
        expert_hidden: int = 128,
        dropout_rate: float = 0.5,
        **kwargs,
    ):
        super().__init__()
        self.cat_embeddings = nnx.List([
            DeterministicEmbed(vocab_size, embed_dim, rngs=rngs)
            for vocab_size in cat_vocab_sizes.values()
        ])
        total_fields = len(cat_vocab_sizes)
        self.input_dim = total_fields * embed_dim + num_static_features
        self.cat_in = nnx.Linear(self.input_dim, embed_hidden, rngs=rngs)
        self.dropout = nnx.Dropout(dropout_rate, rngs=rngs)
        self.cat_out = nnx.Linear(embed_hidden, embed_hidden // 2, rngs=rngs)

        fusion_in = embed_hidden // 2
        self.esmm = SparseMoE_ESMM(
            rngs=rngs,
            input_dim=fusion_in,
            expert_hidden=expert_hidden,
            num_experts=num_experts,
            top_k=top_k,
            dropout_rate=dropout_rate,
        )

    def _shared_branch(self, x_static: jax.Array, x_cat: jax.Array, training: bool) -> jax.Array:
        cat_emb = jnp.concatenate([
            emb(x_cat[:, i].astype(jnp.int32)) for i, emb in enumerate(self.cat_embeddings)
        ], axis=-1)
        common_in = jnp.concatenate([x_static, cat_emb], axis=-1)
        h = nnx.relu(self.cat_in(common_in))
        h = self.dropout(h, deterministic=not training)
        h = nnx.relu(self.cat_out(h))
        return h

    def __call__(
        self,
        x_static: jax.Array,
        x_cat: jax.Array,
        *args,
        training: bool = False,
        **kwargs,
    ) -> Tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
        fused = self._shared_branch(x_static, x_cat, training)
        logits_mob3, aux_mob3, logits_delta, aux_delta = self.esmm(fused, training=training)
        z_delta = nnx.softplus(logits_delta)
        logits_mob6 = logits_mob3 + z_delta
        return logits_mob3, logits_mob6, aux_mob3, aux_delta


__all__ = [
    "DeterministicEmbed",
    "LoanTCNEncoder",
    "QueryEncoder",
    "SparseMoE",
    "SparseMoE_ESMM",
    "HybridCreditModel_MoE",
    "HybridCreditModel_ESMM",
    "HybridCreditModel_Ablation",
]
