import os
os.environ['JAX_PLATFORM_NAME'] = 'mps'
os.environ['JAX_MPS_ASYNC_DISPATCH'] = '0'
import json
import pickle
import glob
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RAW_CSV_PATH = os.path.join(CURRENT_DIR, "sample.csv")
DEFAULT_CACHE_PATH = os.path.join(CURRENT_DIR, "mesm_dataset_cache.pkl")


# ====================================================================
# 1. 业务机构与产品类型映射配置 (基于征信领域知识与数据定义)
# ====================================================================

PRODUCT_TO_INST_CATEGORY = {
    "商业银行": "bank",
    "消费金融公司": "consumer_fin",
    "小额贷款公司": "micro_loan",
    "融资担保公司": "guarantee",
}
INST_CATEGORIES = ['bank', 'consumer_fin', 'micro_loan', 'guarantee']
INST_CATEGORY_CN = {
    'bank': '商业银行',
    'consumer_fin': '消费金融公司',
    'micro_loan': '小额贷款公司',
    'guarantee': '融资担保公司',
}

PRODUCT_TO_PROD_CATEGORY = {
    "贷记卡": "credit_card",
    "其他个人消费贷款": "other_consumer",
    "个人住房商业贷款": "housing_commercial",
}
PROD_CATEGORIES = ['credit_card', 'other_consumer', 'housing_commercial']
PROD_CATEGORY_CN = {
    'credit_card': '贷记卡',
    'other_consumer': '其他个人消费贷款',
    'housing_commercial': '个人住房商业贷款',
}

NUMERIC_COLS = [
    'age',
    'phone_number_num',
    'phone_update_diff',
    'phone_update_last_zj',
    'phone_update_first_zj',
    'live_province_num',
    'live_city_num',
    'line_address_num',
    'live_update_last_zj',
    'live_update_first_zj',
    'job_cmpy_num',
    'job_update_last_zj',
    'job_update_first_zj',
    'job_cmpy_avg_dur',
    'gjj_amt',
]

STRING_COLS = [
    'gender',
    'education',
    'province',
    'city',
    'district',
    'marital_status',
]


def _get_inst_cat(loan_type: str) -> str:
    return PRODUCT_TO_INST_CATEGORY.get(loan_type, 'other')


def _get_prod_cat(loan_type: str) -> str:
    return PRODUCT_TO_PROD_CATEGORY.get(loan_type, 'other')


# ====================================================================
# 2. 序列解析与词汇表构建
# ====================================================================

def parse_loan_seq(json_str: Union[str, float]) -> List[Tuple[str, str, float]]:
    """解析 loan_5y_before_6m_list JSON → [(month_str, loan_type, yx_amt), ...]"""
    if not isinstance(json_str, str) or json_str == '[]' or json_str == '':
        return []
    try:
        return json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        return []


def parse_query_seq(json_str: Union[str, float]) -> List[List[Union[str, int]]]:
    """解析 query_req_before_6m_list JSON → [[date, org_type, reason], ...]"""
    if not isinstance(json_str, str) or json_str == '[]' or json_str == '':
        return []
    try:
        return json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        return []


def parse_report_time(val: str) -> datetime:
    """解析 report_time → datetime (往前推 6 个月作为实际特征截止时间，严防未来信息泄露)"""
    dt = datetime.fromisoformat(val)
    dt = dt.replace(day=1)
    m = dt.month - 6
    y = dt.year
    if m <= 0:
        m += 12
        y -= 1
    return dt.replace(year=y, month=m)


def build_vocab(all_seqs: List[List], key_idx: int) -> Dict[str, int]:
    """从序列列表中收集指定索引的词汇, 建立 id 映射 (0=<UNK>)"""
    all_vals = set()
    for seq in all_seqs:
        for item in seq:
            if len(item) > key_idx:
                all_vals.add(item[key_idx])
    vocab = {v: i + 1 for i, v in enumerate(sorted(all_vals))}
    vocab['<UNK>'] = 0
    return vocab


def build_product_vocab(all_seqs: List[List]) -> Dict[str, int]:
    """收集所有贷款类型, 建立 id 映射 (0=<UNK>)"""
    all_types = set()
    for seq in all_seqs:
        for _, loan_type, _ in seq:
            all_types.add(loan_type)
    vocab = {p: i + 1 for i, p in enumerate(sorted(all_types))}
    vocab['<UNK>'] = 0
    return vocab


# ====================================================================
# 3. 序列聚合统计特征提取 (Handcrafted Statistical Features)
# ====================================================================

def extract_loan_agg_features(
    loan_seqs: List[List[Tuple[str, str, float]]],
    report_times: List[datetime],
) -> np.ndarray:
    """
    从贷款时序中提取 137 维手工统计聚合特征:
    - 25 维基础特征 (笔数、金额、距今月数、近1/3/6/12月变化率、多样性等)
    - 64 维机构类别特征 (4 类机构 × 16 维)
    - 48 维产品类别特征 (3 类产品 × 16 维)
    """
    feats = []
    for seq, report_dt in zip(loan_seqs, report_times):
        report_ym = report_dt.year * 12 + report_dt.month
        if not seq:
            # 25 基础 + 64 机构 + 48 产品 = 137
            feats.append([0.0] * 137)
            continue

        n_loans = len(seq)
        amts, recencies, months, prods = [], [], set(), set()
        recent_1 = recent_3 = recent_6 = recent_12 = 0
        recent_amt_1m = recent_amt_3m = recent_amt_6m = recent_amt_12m = 0.0

        inst_amt = defaultdict(float)
        inst_cnt = defaultdict(int)
        inst_amt_12m, inst_cnt_12m = defaultdict(float), defaultdict(int)
        inst_amt_6m, inst_cnt_6m = defaultdict(float), defaultdict(int)
        inst_amt_3m, inst_cnt_3m = defaultdict(float), defaultdict(int)
        inst_amt_1m, inst_cnt_1m = defaultdict(float), defaultdict(int)

        prod_amt = defaultdict(float)
        prod_cnt = defaultdict(int)
        prod_amt_12m, prod_cnt_12m = defaultdict(float), defaultdict(int)
        prod_amt_6m, prod_cnt_6m = defaultdict(float), defaultdict(int)
        prod_amt_3m, prod_cnt_3m = defaultdict(float), defaultdict(int)
        prod_amt_1m, prod_cnt_1m = defaultdict(float), defaultdict(int)

        for date_str, loan_type, amt in seq:
            year = int(date_str[:4])
            month = int(date_str[5:7])
            recency = report_ym - (year * 12 + month)
            amt_val = float(amt)

            amts.append(amt_val)
            recencies.append(recency)
            months.add((year, month))
            prods.add(loan_type)

            icat = _get_inst_cat(loan_type)
            pcat = _get_prod_cat(loan_type)

            inst_amt[icat] += amt_val
            inst_cnt[icat] += 1
            prod_amt[pcat] += amt_val
            prod_cnt[pcat] += 1

            if recency <= 1:
                recent_1 += 1
                recent_amt_1m += amt_val
                inst_amt_1m[icat] += amt_val
                inst_cnt_1m[icat] += 1
                prod_amt_1m[pcat] += amt_val
                prod_cnt_1m[pcat] += 1
            if recency <= 3:
                recent_3 += 1
                recent_amt_3m += amt_val
                inst_amt_3m[icat] += amt_val
                inst_cnt_3m[icat] += 1
                prod_amt_3m[pcat] += amt_val
                prod_cnt_3m[pcat] += 1
            if recency <= 6:
                recent_6 += 1
                recent_amt_6m += amt_val
                inst_amt_6m[icat] += amt_val
                inst_cnt_6m[icat] += 1
                prod_amt_6m[pcat] += amt_val
                prod_cnt_6m[pcat] += 1
            if recency <= 12:
                recent_12 += 1
                recent_amt_12m += amt_val
                inst_amt_12m[icat] += amt_val
                inst_cnt_12m[icat] += 1
                prod_amt_12m[pcat] += amt_val
                prod_cnt_12m[pcat] += 1

        n_months = len(months)
        n_prods = len(prods)
        avg_amt = float(np.mean(amts))
        max_amt = float(np.max(amts))
        total_amt = float(np.sum(amts))
        avg_recency = float(np.mean(recencies))
        first_recency = float(np.max(recencies))
        last_recency = float(np.min(recencies))
        prod_diversity = n_prods / max(n_loans, 1)
        loans_per_month = n_loans / max(n_months, 1)

        recent_1_3_rate = (recent_1 / recent_3) if recent_3 > 0 else 0.0
        recent_3_6_rate = (recent_3 / recent_6) if recent_6 > 0 else 0.0
        recent_3_12_rate = (recent_3 / recent_12) if recent_12 > 0 else 0.0

        recent_amt_1_3_rate = (recent_amt_1m / recent_amt_3m) if recent_amt_3m > 0 else 0.0
        recent_amt_3_6_rate = (recent_amt_3m / recent_amt_6m) if recent_amt_6m > 0 else 0.0
        recent_amt_3_12_rate = (recent_amt_3m / recent_amt_12m) if recent_amt_12m > 0 else 0.0

        # 机构类别特征 (4 类 × 16 维)
        inst_feats = []
        for cat in INST_CATEGORIES:
            amt_r = inst_amt[cat] / total_amt if total_amt > 0 else 0.0
            cnt_r = inst_cnt[cat] / n_loans if n_loans > 0 else 0.0
            a12, c12 = inst_amt_12m[cat], inst_cnt_12m[cat]
            a6, c6 = inst_amt_6m[cat], inst_cnt_6m[cat]
            a3, c3 = inst_amt_3m[cat], inst_cnt_3m[cat]
            a1, c1 = inst_amt_1m[cat], inst_cnt_1m[cat]

            amt_12_r = a12 / recent_amt_12m if recent_amt_12m > 0 else 0.0
            cnt_12_r = c12 / recent_12 if recent_12 > 0 else 0.0
            amt_6_r = a6 / recent_amt_6m if recent_amt_6m > 0 else 0.0
            cnt_6_r = c6 / recent_6 if recent_6 > 0 else 0.0
            amt_3_r = a3 / recent_amt_3m if recent_amt_3m > 0 else 0.0
            cnt_3_r = c3 / recent_3 if recent_3 > 0 else 0.0
            amt_1_r = a1 / recent_amt_1m if recent_amt_1m > 0 else 0.0
            cnt_1_r = c1 / recent_1 if recent_1 > 0 else 0.0

            amt_3_12_r = amt_3_r / amt_12_r if amt_12_r > 0 else 0.0
            cnt_3_12_r = cnt_3_r / cnt_12_r if cnt_12_r > 0 else 0.0
            amt_1_3_r = amt_1_r / amt_3_r if amt_3_r > 0 else 0.0
            cnt_1_3_r = cnt_1_r / cnt_3_r if cnt_3_r > 0 else 0.0
            amt_3_6_r = amt_3_r / amt_6_r if amt_6_r > 0 else 0.0
            cnt_3_6_r = cnt_3_r / cnt_6_r if cnt_6_r > 0 else 0.0
            inst_feats.extend([
                amt_r, cnt_r, amt_12_r, cnt_12_r, amt_6_r, cnt_6_r, amt_3_r, cnt_3_r,
                amt_1_r, cnt_1_r, amt_3_12_r, cnt_3_12_r, amt_1_3_r, cnt_1_3_r, amt_3_6_r, cnt_3_6_r
            ])

        # 产品类别特征 (3 类 × 16 维)
        prod_feats = []
        for cat in PROD_CATEGORIES:
            amt_r = prod_amt[cat] / total_amt if total_amt > 0 else 0.0
            cnt_r = prod_cnt[cat] / n_loans if n_loans > 0 else 0.0
            a12, c12 = prod_amt_12m[cat], prod_cnt_12m[cat]
            a6, c6 = prod_amt_6m[cat], prod_cnt_6m[cat]
            a3, c3 = prod_amt_3m[cat], prod_cnt_3m[cat]
            a1, c1 = prod_amt_1m[cat], prod_cnt_1m[cat]

            amt_12_r = a12 / recent_amt_12m if recent_amt_12m > 0 else 0.0
            cnt_12_r = c12 / recent_12 if recent_12 > 0 else 0.0
            amt_6_r = a6 / recent_amt_6m if recent_amt_6m > 0 else 0.0
            cnt_6_r = c6 / recent_6 if recent_6 > 0 else 0.0
            amt_3_r = a3 / recent_amt_3m if recent_amt_3m > 0 else 0.0
            cnt_3_r = c3 / recent_3 if recent_3 > 0 else 0.0
            amt_1_r = a1 / recent_amt_1m if recent_amt_1m > 0 else 0.0
            cnt_1_r = c1 / recent_1 if recent_1 > 0 else 0.0

            amt_3_12_r = amt_3_r / amt_12_r if amt_12_r > 0 else 0.0
            cnt_3_12_r = cnt_3_r / cnt_12_r if cnt_12_r > 0 else 0.0
            amt_1_3_r = amt_1_r / amt_3_r if amt_3_r > 0 else 0.0
            cnt_1_3_r = cnt_1_r / cnt_3_r if cnt_3_r > 0 else 0.0
            amt_3_6_r = amt_3_r / amt_6_r if amt_6_r > 0 else 0.0
            cnt_3_6_r = cnt_3_r / cnt_6_r if cnt_6_r > 0 else 0.0
            prod_feats.extend([
                amt_r, cnt_r, amt_12_r, cnt_12_r, amt_6_r, cnt_6_r, amt_3_r, cnt_3_r,
                amt_1_r, cnt_1_r, amt_3_12_r, cnt_3_12_r, amt_1_3_r, cnt_1_3_r, amt_3_6_r, cnt_3_6_r
            ])

        base_feats = [
            n_loans, n_months, n_prods, avg_amt, max_amt, total_amt,
            avg_recency, first_recency, last_recency, recent_1, recent_3, recent_6, recent_12,
            recent_amt_1m, recent_amt_3m, recent_amt_6m, recent_amt_12m,
            recent_1_3_rate, recent_3_6_rate, recent_3_12_rate,
            recent_amt_1_3_rate, recent_amt_3_6_rate, recent_amt_3_12_rate,
            prod_diversity, loans_per_month
        ]
        feats.append(base_feats + inst_feats + prod_feats)

    return np.array(feats, dtype=np.float32)


def extract_query_agg_features(
    query_seqs: List[List[List[Union[str, int]]]],
    report_times: List[datetime],
) -> np.ndarray:
    """
    从查询时序提取聚合特征:
    - 13 维基础特征 (总查询数、机构数、原因数、平均距今、近3/6/12月查询等)
    - 各查询机构时间维度特征 (每机构 × 11 维)
    """
    query_org_types = list(PRODUCT_TO_INST_CATEGORY.keys())
    query_feat_dim = 13 + len(query_org_types) * 11
    feats = []

    for seq, report_dt in zip(query_seqs, report_times):
        report_ym = report_dt.year * 12 + report_dt.month
        if not seq:
            feats.append([0.0] * query_feat_dim)
            continue

        n_queries = len(seq)
        orgs, reasons, recencies = set(), set(), []
        recent_1 = recent_3 = recent_6 = recent_12 = 0
        org_cnt_1m = defaultdict(int)
        org_cnt_3m = defaultdict(int)
        org_cnt_6m = defaultdict(int)
        org_cnt_12m = defaultdict(int)

        for item in seq:
            date_str = item[0]
            org_type = item[1] if len(item) > 1 else ""
            reason = item[2] if len(item) > 2 else ""

            year = int(date_str[:4])
            month = int(date_str[5:7])
            recency = report_ym - (year * 12 + month)

            orgs.add(org_type)
            reasons.add(reason)
            recencies.append(recency)

            if recency <= 1:
                recent_1 += 1
                org_cnt_1m[org_type] += 1
            if recency <= 3:
                recent_3 += 1
                org_cnt_3m[org_type] += 1
            if recency <= 6:
                recent_6 += 1
                org_cnt_6m[org_type] += 1
            if recency <= 12:
                recent_12 += 1
                org_cnt_12m[org_type] += 1

        n_orgs = len(orgs)
        n_reasons = len(reasons)
        avg_recency = float(np.mean(recencies))
        queries_per_org = n_queries / max(n_orgs, 1)
        first_recency = float(np.max(recencies))
        last_recency = float(np.min(recencies))

        recent_1_3_rate = (recent_1 / recent_3) if recent_3 > 0 else 0.0
        recent_3_6_rate = (recent_3 / recent_6) if recent_6 > 0 else 0.0
        recent_3_12_rate = (recent_3 / recent_12) if recent_12 > 0 else 0.0

        org_feats = []
        for ot in query_org_types:
            c1 = org_cnt_1m[ot]
            c3 = org_cnt_3m[ot]
            c6 = org_cnt_6m[ot]
            c12 = org_cnt_12m[ot]
            c12_r = (c12 / recent_12) if recent_12 > 0 else 0.0
            c6_r = (c6 / recent_6) if recent_6 > 0 else 0.0
            c3_r = (c3 / recent_3) if recent_3 > 0 else 0.0
            c1_r = (c1 / recent_1) if recent_1 > 0 else 0.0
            cnt_3_12_r = (c3 / c12) if c12 > 0 else 0.0
            cnt_1_3_r = (c1 / c3) if c3 > 0 else 0.0
            cnt_3_6_r = (c3 / c6) if c6 > 0 else 0.0
            org_feats.extend([c1, c3, c6, c12, c1_r, c3_r, c6_r, c12_r, cnt_1_3_r, cnt_3_6_r, cnt_3_12_r])

        base_feats = [
            n_queries, n_orgs, n_reasons, avg_recency, first_recency, last_recency,
            recent_3, recent_6, recent_12, queries_per_org, recent_1_3_rate, recent_3_6_rate, recent_3_12_rate
        ]
        feats.append(base_feats + org_feats)

    return np.array(feats, dtype=np.float32)


# ====================================================================
# 4. 时序序列张量化与月度池化批处理准备
# ====================================================================

def prepare_loan_batch(
    loan_seqs: List[List[Tuple[str, str, float]]],
    report_times: List[datetime],
    product_to_id: Dict[str, int],
    fixed_max_months: Optional[int] = None,
    fixed_max_prods: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """
    将原始贷款时序转换为月池化 TCN 模型输入张量:
    - product_ids: (B, T, P)
    - amounts: (B, T, P)
    - recencies: (B, T, P)
    - time_deltas: (B, T)
    - seq_lengths: (B,)
    - num_products_per_month: (B, T)
    """
    batch_pids, batch_amts, batch_recencies = [], [], []
    batch_deltas, batch_seq_lens, batch_nprods = [], [], []

    for seq, report_dt in zip(loan_seqs, report_times):
        report_ym = report_dt.year * 12 + report_dt.month
        month_groups = defaultdict(list)

        for date_str, loan_type, amt in seq:
            year = int(date_str[:4])
            month = int(date_str[5:7])
            loan_ym = year * 12 + month
            recency = report_ym - loan_ym
            pid = product_to_id.get(loan_type, 0)
            month_groups[(year, month)].append((pid, amt, recency))

        sorted_yms = sorted(month_groups.keys())
        pids_per_month, amts_per_month, recs_per_month = [], [], []
        deltas = []
        prev_ym = None

        for ym in sorted_yms:
            if prev_ym is None:
                delta = 0
            else:
                delta = (ym[0] - prev_ym[0]) * 12 + (ym[1] - prev_ym[1])
            deltas.append(delta)
            prev_ym = ym

            pid_sum_amt = {}
            pid_first_rec = {}
            for pid, amt, recency in month_groups[ym]:
                pid_sum_amt[pid] = pid_sum_amt.get(pid, 0.0) + amt
                if pid not in pid_first_rec:
                    pid_first_rec[pid] = recency

            month_pids, month_amts, month_recs = [], [], []
            for pid in sorted(pid_sum_amt.keys()):
                month_pids.append(pid)
                month_amts.append(pid_sum_amt[pid])
                month_recs.append(pid_first_rec[pid])

            pids_per_month.append(month_pids)
            amts_per_month.append(month_amts)
            recs_per_month.append(month_recs)

        batch_pids.append(pids_per_month)
        batch_amts.append(amts_per_month)
        batch_recencies.append(recs_per_month)
        batch_deltas.append(deltas)
        batch_seq_lens.append(len(sorted_yms))
        batch_nprods.append([len({pid for pid, _, _ in month_groups[ym]}) for ym in sorted_yms] or [0])

    max_months = max(batch_seq_lens) if batch_seq_lens else 1
    max_prods = max(max(n) for n in batch_nprods) if batch_nprods and any(batch_nprods) else 1

    if fixed_max_months is not None:
        max_months = fixed_max_months
    if fixed_max_prods is not None:
        max_prods = fixed_max_prods

    def _pad_deltas(seqs, m_m):
        arr = np.zeros((len(seqs), m_m), dtype=np.float32)
        for i, seq in enumerate(seqs):
            keep = seq[-m_m:]
            arr[i, :len(keep)] = np.array(keep, dtype=np.float32)
            if len(keep) > 0:
                arr[i, 0] = 0.0
        return arr

    def pad_3d(seqs, dtype=np.float32):
        arr = np.zeros((len(seqs), max_months, max_prods), dtype=dtype)
        for i, months in enumerate(seqs):
            keep = months[-max_months:]
            for m, products in enumerate(keep):
                arr[i, m, :min(len(products), max_prods)] = np.array(products[:max_prods], dtype=dtype)
        return arr

    def pad_2d(seqs, dtype=np.float32):
        arr = np.zeros((len(seqs), max_months), dtype=dtype)
        for i, seq in enumerate(seqs):
            keep = seq[-max_months:]
            arr[i, :len(keep)] = np.array(keep, dtype=dtype)
        return arr

    return {
        "product_ids": pad_3d(batch_pids, np.int32),
        "amounts": pad_3d(batch_amts, np.float32),
        "recencies": pad_3d(batch_recencies, np.float32),
        "time_deltas": _pad_deltas(batch_deltas, max_months),
        "seq_lengths": np.minimum(np.array(batch_seq_lens, dtype=np.int32), max_months),
        "num_products_per_month": np.minimum(pad_2d(batch_nprods, np.int32), max_prods),
    }


def prepare_query_batch(
    query_seqs: List[List[List[Union[str, int]]]],
    report_times: List[datetime],
    query_vocab_org: Dict[str, int],
    query_vocab_reason: Dict[str, int],
    fixed_max_months: Optional[int] = None,
    fixed_max_per_month: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """
    将原始查询时序转换为月池化 QueryEncoder 模型输入张量:
    - org_ids: (B, T, P)
    - reason_ids: (B, T, P)
    - event_counts: (B, T, P)
    - recencies: (B, T, P)
    - time_deltas: (B, T)
    - seq_lengths: (B,)
    - num_per_month: (B, T)
    """
    batch_org_ids, batch_rsn_ids, batch_counts = [], [], []
    batch_recencies, batch_deltas = [], []
    batch_seq_lens, batch_npermonth = [], []

    for seq, report_dt in zip(query_seqs, report_times):
        report_ym = report_dt.year * 12 + report_dt.month
        month_groups = defaultdict(lambda: defaultdict(int))
        month_recency = {}

        for item in seq:
            date_str = item[0]
            org_type = item[1] if len(item) > 1 else ""
            reason = item[2] if len(item) > 2 else ""

            year = int(date_str[:4])
            month = int(date_str[5:7])
            recency = report_ym - (year * 12 + month)

            oid = query_vocab_org.get(org_type, 0)
            rid = query_vocab_reason.get(reason, 0)

            month_groups[(year, month)][(oid, rid)] += 1
            if (year, month) not in month_recency:
                month_recency[(year, month)] = recency

        sorted_yms = sorted(month_groups.keys())
        oids_per_month, rsns_per_month = [], []
        cnts_per_month, recs_per_month = [], []
        deltas = []
        prev_ym = None

        for ym in sorted_yms:
            if prev_ym is None:
                delta = 0
            else:
                delta = (ym[0] - prev_ym[0]) * 12 + (ym[1] - prev_ym[1])
            deltas.append(delta)
            prev_ym = ym

            month_oids, month_rsns, month_cnts, month_recs = [], [], [], []
            for (oid, rid), cnt in sorted(month_groups[ym].items()):
                month_oids.append(oid)
                month_rsns.append(rid)
                month_cnts.append(float(cnt))
                month_recs.append(float(month_recency[ym]))

            oids_per_month.append(month_oids)
            rsns_per_month.append(month_rsns)
            cnts_per_month.append(month_cnts)
            recs_per_month.append(month_recs)

        batch_org_ids.append(oids_per_month)
        batch_rsn_ids.append(rsns_per_month)
        batch_counts.append(cnts_per_month)
        batch_recencies.append(recs_per_month)
        batch_deltas.append(deltas)
        batch_seq_lens.append(len(sorted_yms))
        batch_npermonth.append([len(month_groups[ym]) for ym in sorted_yms] or [0])

    max_months = max(batch_seq_lens) if batch_seq_lens else 1
    max_per_month = max(max(n) for n in batch_npermonth) if batch_npermonth and any(batch_npermonth) else 1

    if fixed_max_months is not None:
        max_months = fixed_max_months
    if fixed_max_per_month is not None:
        max_per_month = fixed_max_per_month

    def _pad_deltas(seqs, m_m):
        arr = np.zeros((len(seqs), m_m), dtype=np.float32)
        for i, seq in enumerate(seqs):
            keep = seq[-m_m:]
            arr[i, :len(keep)] = np.array(keep, dtype=np.float32)
            if len(keep) > 0:
                arr[i, 0] = 0.0
        return arr

    def pad_3d(seqs, dtype=np.float32):
        arr = np.zeros((len(seqs), max_months, max_per_month), dtype=dtype)
        for i, months in enumerate(seqs):
            keep = months[-max_months:]
            for m, items in enumerate(keep):
                arr[i, m, :min(len(items), max_per_month)] = np.array(items[:max_per_month], dtype=dtype)
        return arr

    def pad_2d(seqs, dtype=np.float32):
        arr = np.zeros((len(seqs), max_months), dtype=dtype)
        for i, seq in enumerate(seqs):
            keep = seq[-max_months:]
            arr[i, :len(keep)] = np.array(keep, dtype=dtype)
        return arr

    return {
        "org_ids": pad_3d(batch_org_ids, np.int32),
        "reason_ids": pad_3d(batch_rsn_ids, np.int32),
        "event_counts": pad_3d(batch_counts, np.float32),
        "recencies": pad_3d(batch_recencies, np.float32),
        "time_deltas": _pad_deltas(batch_deltas, max_months),
        "seq_lengths": np.minimum(np.array(batch_seq_lens, dtype=np.int32), max_months),
        "num_per_month": np.minimum(pad_2d(batch_npermonth, np.int32), max_per_month),
    }


def print_mob_crosstab(name: str, y_mob6: np.ndarray, y_mob3: np.ndarray):
    """
    打印 MOB3 与 MOB6 跨期标签的交叉分布 (2×2 列联表) 及业务转化率。
    """
    n = len(y_mob6)
    y6 = np.asarray(y_mob6, dtype=np.int64)
    y3 = np.asarray(y_mob3, dtype=np.int64)

    m00 = int(((y6 == 0) & (y3 == 0)).sum())
    m01 = int(((y6 == 0) & (y3 == 1)).sum())  # 异常倒挂反转
    m10 = int(((y6 == 1) & (y3 == 0)).sum())  # 增量逾期: 3月内正常但6月内逾期
    m11 = int(((y6 == 1) & (y3 == 1)).sum())  # 早期逾期并持续逾期

    print(f"\n{'='*22} {name} (N={n:,}) — MOB3 × MOB6 交叉分布 {'='*22}")
    print(f"{'mob3 \\ mob6':<16}{'mob6=0':>12}{'mob6=1':>12}{'合计':>12}")
    print("-" * 54)
    print(f"{'mob3=0 (早期正常)':<16}{m00:>12,}{m10:>12,}{m00+m10:>12,}")
    print(f"{'mob3=1 (早期逾期)':<16}{m01:>12,}{m11:>12,}{m01+m11:>12,}")
    print(f"{'合计':<16}{m00+m01:>12,}{m10+m11:>12,}{n:>12,}")

    mob6_pos = m10 + m11
    mob3_pos = m01 + m11
    p_m11_m6 = (m11 / mob6_pos * 100) if mob6_pos > 0 else 0.0
    p_m11_m3 = (m11 / mob3_pos * 100) if mob3_pos > 0 else 0.0
    p_delta  = (m10 / (m00 + m10) * 100) if (m00 + m10) > 0 else 0.0

    print(f"\n  • 增量逾期转化率 P(mob6=1 | mob3=0) : {p_delta:6.2f}%  ({m10:,}/{m00+m10:,})  [ESMM 增量风险拟合核心]")
    print(f"  • mob3 恶化维持率 P(mob6=1 | mob3=1) : {p_m11_m3:6.2f}%  ({m11:,}/{mob3_pos:,})  [单调性检验: mob3=1 必然导致 mob6=1]")
    print(f"  • mob6 归因来自早期逾期占比           : {p_m11_m6:6.2f}%  ({m11:,}/{mob6_pos:,})")
    if m01 > 0:
        print(f"  • ⚠️ 异常反转样本 (mob3=1 但 mob6=0) : {m01:,}  (请核查标签定义)")
    else:
        print(f"  • 严格单调性先验成立 (mob3=1 且 mob6=0 样本数为 0)")


# ====================================================================
# 5. 全流程特征工程与数据集构建
# ====================================================================

def process_mesm_dataset(
    csv_path: str = DEFAULT_RAW_CSV_PATH,
    cache_save_path: Optional[str] = DEFAULT_CACHE_PATH,
) -> Dict[str, Any]:
    """
    全量执行 MESM 征信模型特征工程与时序切分流水线。
    """
    print(f"📖 正在从 CSV 读取数据: {csv_path}")
    df = pd.read_csv(csv_path, sep="\t", escapechar="\\")
    #df = pd.concat(map(lambda f: pd.read_csv(f, sep="\t", escapechar="\\"),sorted(glob.glob("/Users/mingxi/Documents/业务扩展/code2/user_sample_dd_out_new/part-*.csv"))))
    
    # 过滤无效 report_time
    df = df[df['report_time'].notna()].copy().reset_index(drop=True)
    N_total = len(df)
    print(f"总有效样本行数: {N_total:,}")

    # 严格按照业务时序切分训练集、验证集与测试集 (OOT)
    train_mask = df['report_time'] < '2025-09-15 00:00:00'
    valid_mask = (df['report_time'] >= '2025-09-15 00:00:00') & (df['report_time'] < '2025-12-01 00:00:00')
    test_mask = df['report_time'] >= '2025-12-01 00:00:00'

    df_train = df[train_mask].copy().reset_index(drop=True)
    df_valid = df[valid_mask].copy().reset_index(drop=True)
    df_test = df[test_mask].copy().reset_index(drop=True)

    print("\n" + "=" * 25 + " 时序切分完成 (Train / Valid / OOT Test) " + "=" * 25)
    print(f"训练集 (Train) : {len(df_train):,} 行 ({len(df_train)/N_total*100:.1f}%) | 报告时间: < 2025-09-15")
    print(f"验证集 (Valid) : {len(df_valid):,} 行 ({len(df_valid)/N_total*100:.1f}%) | 报告时间: 2025-09-15 ~ 2025-12-01")
    print(f"测试集 (Test)  : {len(df_test):,} 行 ({len(df_test)/N_total*100:.1f}%) | 报告时间: >= 2025-12-01")
    print("=" * 75)

    # 标签提取
    def extract_labels(d):
        y6 = d['loan_5y_after_6m_overdue_list'].apply(
            lambda x: 0 if (not isinstance(x, str) or x == '[]' or x == '') else 1
        ).values.astype(np.int64)
        y3 = d['loan_5y_after_3m_overdue_list'].apply(
            lambda x: 0 if (not isinstance(x, str) or x == '[]' or x == '') else 1
        ).values.astype(np.int64)
        return y6, y3

    y_train_mob6, y_train_mob3 = extract_labels(df_train)
    y_valid_mob6, y_valid_mob3 = extract_labels(df_valid)
    y_test_mob6, y_test_mob3 = extract_labels(df_test)

    print_mob_crosstab("训练集 (Train)", y_train_mob6, y_train_mob3)
    print_mob_crosstab("验证集 (Valid)", y_valid_mob6, y_valid_mob3)
    print_mob_crosstab("测试集 (Test/OOT)", y_test_mob6, y_test_mob3)

    # 1. 数值基础特征提取与缺失填充 (严格对齐 notebook: 缺失值填充 0.0)
    num_cols_valid = [c for c in NUMERIC_COLS if c in df.columns]
    X_train_num = df_train[num_cols_valid].fillna(0.0).values.astype(np.float32)
    X_valid_num = df_valid[num_cols_valid].fillna(0.0).values.astype(np.float32)
    X_test_num = df_test[num_cols_valid].fillna(0.0).values.astype(np.float32)

    scaler_num = StandardScaler()
    X_train_num_s = scaler_num.fit_transform(X_train_num).astype(np.float32)
    X_valid_num_s = scaler_num.transform(X_valid_num).astype(np.float32)
    X_test_num_s = scaler_num.transform(X_test_num).astype(np.float32)

    # 2. 类别特征编码
    string_cols_valid = [c for c in STRING_COLS if c in df.columns]
    cat_encoders = {}
    cat_vocab_sizes = {}
    X_train_cat_list, X_valid_cat_list, X_test_cat_list = [], [], []

    for col in string_cols_valid:
        ser_train = df_train[col].fillna('__MISSING__').astype(str)
        ser_valid = df_valid[col].fillna('__MISSING__').astype(str)
        ser_test = df_test[col].fillna('__MISSING__').astype(str)

        if col == 'district':
            top200 = set(ser_train.value_counts().head(200).index.tolist())
            cat_encoders['district_top'] = top200
            ser_train = ser_train.where(ser_train.isin(top200), '<UNK>')
            ser_valid = ser_valid.where(ser_valid.isin(top200), '<UNK>')
            ser_test = ser_test.where(ser_test.isin(top200), '<UNK>')

        le = LabelEncoder()
        vals_train = le.fit_transform(ser_train).astype(np.int64)
        cat_encoders[col] = le
        cat_vocab_sizes[col] = len(le.classes_) + 1

        known = set(le.classes_)
        vals_valid = np.array([le.transform([v])[0] if v in known else (len(le.classes_)) for v in ser_valid], dtype=np.int64)
        vals_test = np.array([le.transform([v])[0] if v in known else (len(le.classes_)) for v in ser_test], dtype=np.int64)

        X_train_cat_list.append(vals_train.reshape(-1, 1))
        X_valid_cat_list.append(vals_valid.reshape(-1, 1))
        X_test_cat_list.append(vals_test.reshape(-1, 1))

    X_train_cat = np.concatenate(X_train_cat_list, axis=1) if X_train_cat_list else np.empty((len(df_train), 0), dtype=np.int64)
    X_valid_cat = np.concatenate(X_valid_cat_list, axis=1) if X_valid_cat_list else np.empty((len(df_valid), 0), dtype=np.int64)
    X_test_cat = np.concatenate(X_test_cat_list, axis=1) if X_test_cat_list else np.empty((len(df_test), 0), dtype=np.int64)

    # 3. 序列提取与报告时间转换
    train_loan_seqs = df_train['loan_seq_before_6m_list'].apply(parse_loan_seq).tolist()
    valid_loan_seqs = df_valid['loan_seq_before_6m_list'].apply(parse_loan_seq).tolist()
    test_loan_seqs = df_test['loan_seq_before_6m_list'].apply(parse_loan_seq).tolist()

    train_query_seqs = df_train['query_req_before_6m_list'].apply(parse_query_seq).tolist()
    valid_query_seqs = df_valid['query_req_before_6m_list'].apply(parse_query_seq).tolist()
    test_query_seqs = df_test['query_req_before_6m_list'].apply(parse_query_seq).tolist()

    train_times = df_train['report_time'].apply(parse_report_time).tolist()
    valid_times = df_valid['report_time'].apply(parse_report_time).tolist()
    test_times = df_test['report_time'].apply(parse_report_time).tolist()

    # 4. 词汇表 (严格仅基于训练集拟合)
    loan_vocab = build_product_vocab(train_loan_seqs)
    query_vocab_org = build_vocab(train_query_seqs, key_idx=1)
    query_vocab_reason = build_vocab(train_query_seqs, key_idx=2)

    # 5. 手工统计聚合特征与标准化
    print("\n⏳ 正在提取贷款与查询序列的手工聚合特征 (137 维贷款 + 57 维查询)...")
    loan_agg_train = extract_loan_agg_features(train_loan_seqs, train_times)
    loan_agg_valid = extract_loan_agg_features(valid_loan_seqs, valid_times)
    loan_agg_test = extract_loan_agg_features(test_loan_seqs, test_times)

    query_agg_train = extract_query_agg_features(train_query_seqs, train_times)
    query_agg_valid = extract_query_agg_features(valid_query_seqs, valid_times)
    query_agg_test = extract_query_agg_features(test_query_seqs, test_times)

    loan_agg_scaler = StandardScaler()
    query_agg_scaler = StandardScaler()

    loan_agg_train_s = loan_agg_scaler.fit_transform(loan_agg_train).astype(np.float32)
    loan_agg_valid_s = loan_agg_scaler.transform(loan_agg_valid).astype(np.float32)
    loan_agg_test_s = loan_agg_scaler.transform(loan_agg_test).astype(np.float32)

    query_agg_train_s = query_agg_scaler.fit_transform(query_agg_train).astype(np.float32)
    query_agg_valid_s = query_agg_scaler.transform(query_agg_valid).astype(np.float32)
    query_agg_test_s = query_agg_scaler.transform(query_agg_test).astype(np.float32)

    # 拼接 Dense 静态输入
    X_train_static = np.concatenate([X_train_num_s, loan_agg_train_s, query_agg_train_s], axis=1).astype(np.float32)
    X_valid_static = np.concatenate([X_valid_num_s, loan_agg_valid_s, query_agg_valid_s], axis=1).astype(np.float32)
    X_test_static = np.concatenate([X_test_num_s, loan_agg_test_s, query_agg_test_s], axis=1).astype(np.float32)

    # 6. 时序模型输入张量化 (固定序列长度超参)
    max_months = 75
    max_prods = 7
    max_query_months = 20
    max_query_per_month = 10

    batch_loan_train = prepare_loan_batch(train_loan_seqs, train_times, loan_vocab, fixed_max_months=max_months, fixed_max_prods=max_prods)
    batch_loan_valid = prepare_loan_batch(valid_loan_seqs, valid_times, loan_vocab, fixed_max_months=max_months, fixed_max_prods=max_prods)
    batch_loan_test = prepare_loan_batch(test_loan_seqs, test_times, loan_vocab, fixed_max_months=max_months, fixed_max_prods=max_prods)

    batch_query_train = prepare_query_batch(train_query_seqs, train_times, query_vocab_org, query_vocab_reason, fixed_max_months=max_query_months, fixed_max_per_month=max_query_per_month)
    batch_query_valid = prepare_query_batch(valid_query_seqs, valid_times, query_vocab_org, query_vocab_reason, fixed_max_months=max_query_months, fixed_max_per_month=max_query_per_month)
    batch_query_test = prepare_query_batch(test_query_seqs, test_times, query_vocab_org, query_vocab_reason, fixed_max_months=max_query_months, fixed_max_per_month=max_query_per_month)

    # 7. 构建特征名称列表 (供 SHAP 及可解释性分析使用)
    feature_names = [f"num_{c}" for c in num_cols_valid] + \
                    [f"loan_agg_{i}" for i in range(loan_agg_train.shape[1])] + \
                    [f"query_agg_{i}" for i in range(query_agg_train.shape[1])] + \
                    [f"cat_{c}" for c in string_cols_valid] + \
                    ['loan_seq (overall)', 'query_seq (overall)']

    dataset: Dict[str, Any] = {
        "X_train_static": X_train_static,
        "X_train_cat": X_train_cat,
        "batch_loan_train": batch_loan_train,
        "batch_query_train": batch_query_train,
        "y_train_mob6": y_train_mob6,
        "y_train_mob3": y_train_mob3,

        "X_valid_static": X_valid_static,
        "X_valid_cat": X_valid_cat,
        "batch_loan_valid": batch_loan_valid,
        "batch_query_valid": batch_query_valid,
        "y_valid_mob6": y_valid_mob6,
        "y_valid_mob3": y_valid_mob3,

        "X_test_static": X_test_static,
        "X_test_cat": X_test_cat,
        "batch_loan_test": batch_loan_test,
        "batch_query_test": batch_query_test,
        "y_test_mob6": y_test_mob6,
        "y_test_mob3": y_test_mob3,

        "cat_vocab_sizes": cat_vocab_sizes,
        "loan_vocab": loan_vocab,
        "query_vocab_org": query_vocab_org,
        "query_vocab_reason": query_vocab_reason,
        "feature_names": feature_names,
        "num_static_features": X_train_static.shape[1],
    }

    if cache_save_path:
        with open(cache_save_path, "wb") as f:
            pickle.dump(dataset, f)
        print(f"✅ MESM 数据集已成功持久化缓存至: {cache_save_path}")

    return dataset


def load_data(
    csv_path: str = DEFAULT_RAW_CSV_PATH,
    cache_path: str = DEFAULT_CACHE_PATH,
    force_recompute: bool = False,
) -> Dict[str, Any]:
    """
    加载数据 (缓存优先)。
    1. 若 cache_path 存在且 force_recompute=False，直接秒级反序列化读取；
    2. 否则重新执行全量特征工程，并保存缓存。
    """
    if os.path.exists(cache_path) and not force_recompute:
        print(f"📦 发现本地数据集缓存: {cache_path}，正在直接加载...")
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
        return data

    print(f"🚀 未找到本地缓存 {cache_path} 或设置了 force_recompute=True，启动全量数据处理...")
    return process_mesm_dataset(csv_path=csv_path, cache_save_path=cache_path)


if __name__ == "__main__":
    ds = load_data(force_recompute=True)
    print("\n✅ 数据集校验成功:")
    print(f"  • X_train_static shape: {ds['X_train_static'].shape}")
    print(f"  • X_valid_static shape: {ds['X_valid_static'].shape}")
    print(f"  • X_test_static shape : {ds['X_test_static'].shape}")
    print(f"  • 字符类别特征列数: {ds['X_train_cat'].shape[1]}")
    print(f"  • 贷款产品种类数: {len(ds['loan_vocab'])}")
    print(f"  • 查询机构种类数: {len(ds['query_vocab_org'])}")
    print(f"  • 查询原因种类数: {len(ds['query_vocab_reason'])}")
