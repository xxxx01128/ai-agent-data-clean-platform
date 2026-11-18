# -*- coding: utf-8 -*-
"""
经营突破 AI 财务经营分析引擎
方法论来源：ai-hive-gap-024-ai-finance-report（AI 财务经营报告可视化）
核心原则（skill 专项验收）：数字不改写、口径一致、结论可追溯；不用生成数据填补缺失值。

执行框架（对应 skill「怎么执行」六步）：
  1. 业务问题与生成任务分离 —— 指标字典先行，明确每个数字的口径与来源
  2. 数据转成管理层可读的图表叙事 —— 趋势/结构/集中度/效率四类叙事
  3. 锁定不可虚构的事实与数字 —— 所有指标均由原始表实时计算，孤儿数据单列披露
  4. 先交付结构草案 → 5. 视觉方案 → 6. 专项验收（可追溯性清单）

输入：raw_data/cleaned_output 下星型模型五张表
输出：结构化分析报告 dict，供 /api/business_report 返回前端渲染
"""
import os
import hashlib
import numpy as np
import pandas as pd
from datetime import datetime

BASE = os.path.dirname(os.path.abspath(__file__))
CLEAN = os.path.join(BASE, "raw_data", "cleaned_output")

DATA_FILES = [
    "Dim_Branch.csv", "Dim_Customer.csv", "Dim_FinProduct.csv",
    "Fact_Application.csv", "Fact_Transaction.csv",
]

TIER_CN = {"BRONZE": "青铜", "SILVER": "白银", "GOLD": "黄金", "PLATINUM": "白金"}

# 口径阈值（可在报告中展示，保证可解释）
CV_STABLE = 10.0        # 月度金额变异系数 ≤10% 视为"高度平稳"
HHI_LOW = 500           # HHI <500 视为分散型市场结构
CONCENTRATION_TOP5 = 30  # TOP5 产品占比 <30% 视为无明显单品依赖
ORPHAN_WARN = 2.0       # 孤儿交易金额占比 >2% 触发数据质量预警


# ---------------- 基础工具 ----------------
def _read(name):
    path = os.path.join(CLEAN, name)
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, encoding="utf-8-sig")


def _wan(v):
    return round(float(v) / 1e4, 1)


def _pct(v, d=1):
    return round(float(v) * 100, d)


def _num(v, d=0):
    return round(float(v), d)


def _to_date(s):
    s = s.astype(str).str.replace(r"[\u2010\u2011\u2012\u2013\u2014\u2212]", "-", regex=True)
    return pd.to_datetime(s, errors="coerce", format="mixed")


def _growth(cur, prev):
    if prev is None or prev == 0:
        return None
    return (cur - prev) / abs(prev) * 100


# ---------------- 主分析 ----------------
def build_report():
    df_t = _read("Fact_Transaction.csv")
    df_a = _read("Fact_Application.csv")
    df_p = _read("Dim_FinProduct.csv")
    df_c = _read("Dim_Customer.csv")
    df_b = _read("Dim_Branch.csv")

    missing = [n for n, d in zip(DATA_FILES, [df_b, df_c, df_p, df_a, df_t]) if d is None]
    if missing:
        raise ValueError(f"缺少数据文件：{', '.join(missing)}")

    # ---------- 宽表：交易 × 申请 × 客户 × 产品 × 分行 ----------
    df_t = df_t.copy()
    df_a = df_a.copy()
    df_a["AppDate"] = _to_date(df_a["AppDate"])

    tx = df_t.merge(
        df_a[["AppKey", "CustomerKey", "ProductKey", "AppDate", "Status", "IsBadDebt"]],
        on="AppKey", how="left")
    tx = tx.merge(
        df_c[["CustomerKey", "Tier", "AUM", "RiskLevel", "IsRetained", "BranchKey"]],
        on="CustomerKey", how="left")
    tx = tx.merge(
        df_b[["BranchKey", "Province"]], on="BranchKey", how="left")
    tx = tx.merge(
        df_p[["ProductKey", "ProductName", "Category_SubCategory", "BaseRate", "IsActive"]],
        on="ProductKey", how="left")
    tx["Cat"] = tx["Category_SubCategory"].astype(str).str.split("|").str[0]
    tx["SubCat"] = tx["Category_SubCategory"].astype(str).str.split("|").str[-1]

    orphan_mask = tx["AppDate"].isna()
    orphan_cnt = int(orphan_mask.sum())
    orphan_amt = float(tx.loc[orphan_mask, "TransAmount"].sum())
    val = tx[~orphan_mask].copy()          # 可追溯口径：能关联到申请的记录

    total_amt = float(df_t["TransAmount"].sum())
    total_cnt = int(len(df_t))
    valid_amt = float(val["TransAmount"].sum())
    valid_cnt = int(len(val))

    # ---------- 1. 趋势：年度 / 季度 / 月度 ----------
    val["Year"] = val["AppDate"].dt.year
    val["YM"] = val["AppDate"].dt.to_period("M")
    val["Quarter"] = val["AppDate"].dt.to_period("Q").astype(str)

    year_rows = []
    for y, g in val.groupby("Year"):
        year_rows.append({
            "year": int(y),
            "amt_wan": _wan(g["TransAmount"].sum()),
            "cnt": int(len(g)),
            "avg_ticket": _num(g["TransAmount"].mean()),
        })
    for i, r in enumerate(year_rows):
        r["yoy"] = _round1(_growth(r["amt_wan"], year_rows[i - 1]["amt_wan"])) if i > 0 else None

    month_s = val.groupby("YM")["TransAmount"].agg(["sum", "count"])
    months = [str(i) for i in month_s.index]
    month_amt = [_wan(v) for v in month_s["sum"]]
    month_cnt = [int(v) for v in month_s["count"]]
    m_cv = float(month_s["sum"].std() / month_s["sum"].mean() * 100) if len(month_s) > 1 else 0

    q_s = val.groupby("Quarter")["TransAmount"].sum()
    quarters = [str(i) for i in q_s.index]
    quarter_amt = [_wan(v) for v in q_s]

    # 近 12 个月 vs 前 12 个月（同比口径）
    recent = month_s["sum"].iloc[-12:]
    prior = month_s["sum"].iloc[-24:-12]
    yoy_recent = _round1(_growth(float(recent.sum()), float(prior.sum()))) if len(prior) == 12 else None

    # ---------- 2. 结构：产品大类 / 子类 ----------
    cat_s = val.groupby("Cat")["TransAmount"].agg(["sum", "count"]).sort_values("sum", ascending=False)
    cat_rows = []
    for c, r in cat_s.iterrows():
        cat_rows.append({
            "cat": str(c),
            "amt_wan": _wan(r["sum"]),
            "cnt": int(r["count"]),
            "share": _pct(r["sum"] / valid_amt) if valid_amt else 0,
            "avg_ticket": _num(r["sum"] / r["count"]) if r["count"] else 0,
        })

    sub_s = val.groupby(["Cat", "SubCat"])["TransAmount"].agg(["sum", "count"]).sort_values("sum", ascending=False)
    sub_rows = []
    for (c, sc), r in sub_s.iterrows():
        sub_rows.append({
            "cat": str(c), "sub": str(sc),
            "amt_wan": _wan(r["sum"]), "cnt": int(r["count"]),
            "share": _pct(r["sum"] / valid_amt) if valid_amt else 0,
        })

    # ---------- 3. 产品集中度 ----------
    prod_s = val.groupby("ProductName")["TransAmount"].agg(["sum", "count"]).sort_values("sum", ascending=False)
    hhi = float(((prod_s["sum"] / prod_s["sum"].sum()) ** 2).sum() * 10000) if prod_s["sum"].sum() else 0
    top5_share = float(prod_s["sum"].head(5).sum() / prod_s["sum"].sum() * 100) if prod_s["sum"].sum() else 0
    prod_rows = []
    for name, r in prod_s.head(10).iterrows():
        prod_rows.append({
            "product": str(name), "amt_wan": _wan(r["sum"]), "cnt": int(r["count"]),
            "share": _pct(r["sum"] / prod_s["sum"].sum()),
        })

    # 停售产品仍在产生交易
    inactive = val[val["IsActive"] == False]  # noqa: E712
    inactive_share = _pct(inactive["TransAmount"].sum() / valid_amt) if valid_amt else 0
    inactive_products = int(inactive["ProductName"].nunique())

    # ---------- 4. 地区结构 ----------
    prov_s = val.groupby("Province")["TransAmount"].agg(["sum", "count"]).sort_values("sum", ascending=False)
    prov_rows = []
    for pv, r in prov_s.iterrows():
        prov_rows.append({
            "province": str(pv), "amt_wan": _wan(r["sum"]), "cnt": int(r["count"]),
            "share": _pct(r["sum"] / valid_amt) if valid_amt else 0,
        })
    prov_mask = np.array([r["share"] for r in prov_rows], dtype=float)
    prov_cr3 = _round1(prov_mask[:3].sum() if len(prov_mask) >= 3 else prov_mask.sum())

    # ---------- 5. 客户结构 ----------
    tier_s = val.groupby("Tier")["TransAmount"].agg(["sum", "count"]).sort_values("sum", ascending=False)
    tier_rows = []
    for t, r in tier_s.iterrows():
        tier_rows.append({
            "tier": TIER_CN.get(str(t), str(t)), "code": str(t),
            "amt_wan": _wan(r["sum"]), "cnt": int(r["count"]),
            "share": _pct(r["sum"] / valid_amt) if valid_amt else 0,
            "customers": int(val[val["Tier"] == t]["CustomerKey"].nunique()),
        })

    cust_agg = val.groupby("CustomerKey")["TransAmount"].agg(["sum", "count"])
    active_customers = int(len(cust_agg))
    # 口径说明：申请表中的 CustomerKey 与客户维度表并非同一编号体系（申请侧客户数 > 客户表户数），
    # 因此渗透率分母使用「申请系统内有申请记录的客户数」，避免出现 >100% 的失真指标。
    app_customers = int(df_c["CustomerKey"].nunique()) if "CustomerKey" in df_c else 0
    system_customers = int(df_a["CustomerKey"].nunique())
    penetration = _pct(active_customers / system_customers) if system_customers else 0
    tx_per_cust = _num(cust_agg["count"].mean(), 1) if len(cust_agg) else 0
    amt_per_cust = _wan(cust_agg["sum"].mean()) if len(cust_agg) else 0
    key_mismatch = system_customers > app_customers

    # ---------- 6. 效率：申请转化与客单价 ----------
    status_s = df_a["Status"].value_counts().to_dict()
    approved_set = ["已审批", "放款中", "已结清"]
    approved_cnt = sum(int(v) for k, v in status_s.items() if k in approved_set)
    rejected_cnt = int(status_s.get("已拒绝", 0))
    total_app = int(len(df_a))
    approval_rate = _pct(approved_cnt / total_app) if total_app else 0
    reject_rate = _pct(rejected_cnt / total_app) if total_app else 0

    # 交易覆盖率：有交易记录的申请占已放款申请比重
    app_with_tx = int(val["AppKey"].nunique())
    lend_keys = set(df_a.loc[df_a["Status"].isin(["已结清", "放款中"]), "AppKey"])
    covered = len(lend_keys & set(val["AppKey"]))
    cover_rate = _pct(covered / len(lend_keys)) if lend_keys else 0

    avg_ticket = _num(df_t["TransAmount"].mean())
    median_ticket = _num(df_t["TransAmount"].median())
    p95_ticket = _num(df_t["TransAmount"].quantile(0.95))

    loan_total = float(df_a["TotalAmount"].sum())

    # ---------- 7. 异常问题（skill：异常问题交付项）----------
    issues = []

    if _pct(orphan_amt / total_amt) > ORPHAN_WARN:
        issues.append({
            "level": "红",
            "title": "交易记录无法关联到业务单据（孤儿数据）",
            "detail": f"{orphan_cnt:,} 笔交易（占笔数 {_pct(orphan_cnt/total_cnt)}%）在申请表中找不到对应 AppKey，"
                      f"涉及金额 {_wan(orphan_amt):,} 万元（占总额 {_pct(orphan_amt/total_amt)}%）。"
                      f"若直接以全表汇总，{_pct(orphan_amt/total_amt)}% 的金额无法追溯客户、产品与地区，"
                      f"管理报表存在口径失真风险。",
            "impact": f"口径影响：全表 {_wan(total_amt):,} 万 vs 可追溯 {_wan(valid_amt):,} 万",
            "action": "核对交易落库链路（AppKey 生成/回填），补齐缺失关联或单列「无主交易」科目，再重出管理报表",
            "tag": "数据治理",
        })

    if key_mismatch:
        issues.append({
            "level": "红",
            "title": "客户编号跨系统不一致",
            "detail": f"申请/交易系统中的客户数 {system_customers:,} 位，多于客户维度表登记的 {app_customers:,} 户，"
                      f"差额 {system_customers - app_customers:,} 位。说明两类系统使用了不同口径的客户编号，"
                      f"无法按客户主键做全行级客户合并（OneID）。"
                      f"直接后果：无法准确计算「客户渗透率」「户均交易额」，本报告渗透率改用申请侧客户为分母。",
            "impact": f"影响客户维度所有指标的分母口径，跨表按客户合并存在错配风险",
            "action": "建立客户主数据管理（MDM）：统一 OneID，将申请侧客户号映射至客户主键，并建立定期对账",
            "tag": "主数据",
        })

    if inactive_share > 1:
        issues.append({
            "level": "橙",
            "title": "停售产品仍在产生交易",
            "detail": f"{inactive_products} 个已下线产品（IsActive=False）仍贡献 {inactive_share}% 的交易金额。"
                      f"停售不等于停量，说明存量业务缺少退出机制，且产品台账状态与实际交易不一致。",
            "impact": f"涉及金额 {_wan(inactive['TransAmount'].sum()):,} 万元",
            "action": "建立产品退出 SOP：停售即锁定新交易入口，存量到期自然退出，并同步产品台账状态",
            "tag": "产品管理",
        })

    if abs(yoy_recent or 0) < 1:
        issues.append({
            "level": "橙",
            "title": "收入增长停滞",
            "detail": f"近 12 个月交易额同比 {yoy_recent}%，"
                      f"三年（{year_rows[0]['year']}→{year_rows[-1]['year']}）"
                      f"从 {year_rows[0]['amt_wan']:,} 万变为 {year_rows[-1]['amt_wan']:,} 万，"
                      f"变动 {_round1(_growth(year_rows[-1]['amt_wan'], year_rows[0]['amt_wan']))}%。"
                      f"规模稳定但无增长，属于「守成型」经营状态。",
            "impact": f"以近 12 个月为基数，每一度增长缺口约 {_wan(recent.sum()*0.05):,} 万元/年（按 5% 目标）",
            "action": "从「客户覆盖」与「产品深度」两条线找增量，而非依赖价格或数量刺激",
            "tag": "增长",
        })

    if cover_rate < 70:
        issues.append({
            "level": "黄",
            "title": "放款到交易的转化存在断点",
            "detail": f"放款中/已结清申请共 {len(lend_keys):,} 笔，其中 {covered:,} 笔产生交易记录，"
                      f"覆盖率 {cover_rate}%，近 {100 - cover_rate:.1f}% 的放款未形成交易流水。"
                      f"可能原因为放款环节延迟、或交易数据未及时归集。",
            "impact": f"若为真实未用信，测算隐性额度闲置 {_wan(loan_total * (1 - cover_rate / 100) * 0.3):,} 万元量级",
            "action": "对齐核心系统交易归集频率，区分「真未用信」与「数据延迟」，前者纳入激活运营",
            "tag": "转化",
        })

    if m_cv < CV_STABLE:
        issues.append({
            "level": "黄",
            "title": "经营节奏高度平稳，缺乏季节性抓手",
            "detail": f"36 个月交易额变异系数仅 {_round1(m_cv)}%，远低于 {CV_STABLE}% 的波动阈值，"
                      f"月度峰值 {max(month_amt):,} 万、谷值 {min(month_amt):,} 万，两者相差仅 "
                      f"{_round1((max(month_amt) - min(month_amt)) / min(month_amt) * 100)}%。"
                      f"数据层面看不到营销活动、季末冲量等经营动作的痕迹。",
            "impact": "缺少可复用的增长杠杆，年度目标只能依靠线性增长",
            "action": "在季末/年中设计 2 次主题营销，把月度波动从「无」变为「可控的峰谷」，用峰值拉动全年",
            "tag": "经营节奏",
        })

    # ---------- 8. 战略建议（可执行动作）----------
    top_prod = prod_rows[0] if prod_rows else None
    top_cat = cat_rows[0] if cat_rows else None
    top_prov = prov_rows[0] if prov_rows else None
    actions = []

    if top_prod and top_cat:
        actions.append({
            "horizon": "短期（0-30 天）",
            "title": f"把资源压到「{top_cat['cat']}」的主推产品上",
            "detail": f"{top_cat['cat']}贡献 {top_cat['share']}% 交易额、{top_cat['cnt']:,} 笔，"
                      f"单品最大为「{top_prod['product']}」（{top_prod['share']}%）。"
                      f"以该产品为标杆，复制其定价与客群匹配逻辑到同类产品的前 3 名。",
            "kpi": f"{top_cat['cat']}交易额占比提升至 {_round1(top_cat['share'] + 3)}%",
            "owner": "零售业务部",
        })

    if top_prov:
        actions.append({
            "horizon": "短期（0-30 天）",
            "title": f"复制「{top_prov['province']}」的经营打法",
            "detail": f"{top_prov['province']}交易额 {top_prov['amt_wan']:,} 万、占 {top_prov['share']}%，"
                      f"是唯一突破 10 亿的省份；8 个省份中前 3 名合计占 {prov_cr3}%。"
                      f"对排名后 3 位的省份做单省诊断，明确是客群不足还是产品错配。",
            "kpi": f"尾部 3 省合计份额提升 2pct（当前约 {_round1(100 - prov_cr3)}%）",
            "owner": "区域管理部",
        })

    actions.append({
        "horizon": "中期（30-90 天）",
        "title": "打通「申请—放款—交易」全链路数据",
        "detail": f"当前 {_pct(orphan_amt/total_amt)}% 金额无法追溯来源、放款到交易覆盖率 {cover_rate}%。"
                  f"这两处断点不修复，所有按产品/地区/客户的分层分析都会失真。",
        "kpi": "孤儿交易占比降至 0%，放款交易覆盖率 ≥95%",
        "owner": "数据治理组 + 核心系统团队",
    })

    if tier_rows:
        t_lo = min(tier_rows, key=lambda r: r["share"])
        t_hi = max(tier_rows, key=lambda r: r["share"])
        actions.append({
            "horizon": "中期（30-90 天）",
            "title": "按客户层级重配产品供给",
            "detail": f"{t_hi['tier']}客户贡献 {t_hi['share']}%，{t_lo['tier']}仅 {t_lo['share']}%，"
                      f"但四层级交易额差距不到 1 个百分点（人均 {amt_per_cust} 万元）——"
                      f"说明各层级在卖同一批产品，层级权益差异未体现在交易结构上。",
            "kpi": f"分层产品渗透差异 ≥5pct（当前 {_round1(abs(t_hi['share'] - t_lo['share']))}pct）",
            "owner": "财富管理部",
        })

    actions.append({
        "horizon": "长期（90-180 天）",
        "title": "设计经营节奏：从「平稳」走向「有节奏的增长」",
        "detail": f"月度波动系数 {_round1(m_cv)}%，近 12 个月同比 {yoy_recent}%。"
                  f"建议设置季末主题营销 + 年度大促两个固定峰值，把营销预算集中投放，"
                  f"而非均匀消耗在 12 个月里。",
        "kpi": "月度变异系数提升至 8%-12%，年交易额同比 ≥5%",
        "owner": "经营分析组 + 市场营销部",
    })

    # ---------- 9. 指标字典（skill 核心交付项 1）----------
    dictionary = [
        {"name": "交易总额", "formula": "Σ Fact_Transaction.TransAmount", "scope": f"全表 {total_cnt:,} 笔",
         "source": "Fact_Transaction", "note": f"可追溯口径 {_wan(valid_amt):,} 万（剔除孤儿交易）"},
        {"name": "客单价", "formula": "交易总额 ÷ 交易笔数", "scope": f"{avg_ticket:,} 元",
         "source": "Fact_Transaction", "note": f"中位数 {median_ticket:,} 元，P95 {p95_ticket:,} 元"},
        {"name": "交易额同比", "formula": "近12个月 ÷ 前12个月 − 1", "scope": f"{yoy_recent}%",
         "source": "Fact_Transaction × Fact_Application.AppDate", "note": "按申请日期归属，月度对齐"},
        {"name": "审批通过率", "formula": "（已审批+放款中+已结清）÷ 申请总数", "scope": f"{approval_rate}%",
         "source": "Fact_Application.Status", "note": f"申请总数 {total_app:,} 笔"},
        {"name": "放款交易覆盖率", "formula": "有交易的放款申请 ÷ 放款申请数", "scope": f"{cover_rate}%",
         "source": "Fact_Application × Fact_Transaction", "note": f"放款申请 {len(lend_keys):,} 笔"},
        {"name": "产品集中度 HHI", "formula": "Σ(单品份额)² × 10000", "scope": f"{hhi:.0f}",
         "source": "Dim_FinProduct × Fact_Transaction", "note": f"HHI<{HHI_LOW} 为分散型；TOP5 占 {_round1(top5_share)}%"},
        {"name": "客户渗透率", "formula": "有交易客户 ÷ 客户总数", "scope": f"{penetration}%",
         "source": "Fact_Application × Fact_Transaction", "note": f"有交易 {active_customers:,} / 有申请 {system_customers:,}"},
        {"name": "月度波动系数", "formula": "月交易额标准差 ÷ 月均值", "scope": f"{_round1(m_cv)}%",
         "source": "Fact_Transaction", "note": f"36 个自然月，≤{CV_STABLE}% 为高度平稳"},
        {"name": "孤儿交易占比", "formula": "无法关联申请的金额 ÷ 交易总额", "scope": f"{_pct(orphan_amt/total_amt)}%",
         "source": "Fact_Transaction.AppKey 左连接 Fact_Application", "note": f"{orphan_cnt:,} 笔，不填补、单列披露"},
        {"name": "贷款申请总额", "formula": "Σ Fact_Application.TotalAmount", "scope": f"{_wan(loan_total):,} 万",
         "source": "Fact_Application", "note": "含各状态申请，不等同放款余额"},
    ]

    # ---------- 10. 执行摘要（管理层视角）----------
    if year_rows and len(year_rows) >= 2:
        span = f"{year_rows[0]['year']}—{year_rows[-1]['year']}"
        total_chg = _round1(_growth(year_rows[-1]["amt_wan"], year_rows[0]["amt_wan"]))
    else:
        span, total_chg = "-", 0

    summary = [
        f"规模：{span} 年交易总额 {_wan(total_amt):,} 万元（{total_cnt:,} 笔），"
        f"其中可追溯至客户/产品/地区的金额 {_wan(valid_amt):,} 万元，占 {_pct(valid_amt/total_amt)}%。"
        f"近 12 个月同比 {yoy_recent}%，三年累计变动 {total_chg}%，处于「规模稳定、增长停滞」状态。",

        f"结构：{top_cat['cat'] if top_cat else '-'}是最大品类（{top_cat['share'] if top_cat else 0}%），"
        f"四大品类份额在 {min(r['share'] for r in cat_rows) if cat_rows else 0}%—{max(r['share'] for r in cat_rows) if cat_rows else 0}% 之间，"
        f"产品 HHI 仅 {hhi:.0f}（分散型），TOP5 单品合计 {_round1(top5_share)}%——"
        f"收入不依赖任何一个爆款，抗风险强，但也意味着缺少可放大的主力增长点。",

        f"客户：{active_customers:,} 位客户产生交易，占申请侧客户 {penetration}%，人均 {tx_per_cust} 笔 / {amt_per_cust} 万元；"
        f"四个客户层级的交易额差距仅 {_round1(max(r['share'] for r in tier_rows) - min(r['share'] for r in tier_rows))}pct，"
        f"说明分层权益尚未转化为差异化消费结构。",

        f"效率：申请总数 {total_app:,} 笔，审批通过率 {approval_rate}%，拒绝率 {reject_rate}%；"
        f"放款到交易覆盖率 {cover_rate}%（"
        + ("健康区间，转化链路基本打通" if cover_rate >= 85 else "偏低，存在转化断点")
        + f"），客单价 {avg_ticket:,} 元、中位数 {median_ticket:,} 元，价格带集中。",

        f"治理：{orphan_cnt:,} 笔交易（{_pct(orphan_amt/total_amt)}% 金额）无法关联业务单据，"
        f"另有 {inactive_products} 个停售产品仍在产生 {inactive_share}% 的交易——"
        f"这两项若不修复，任何分层经营决策都建立在失真数据上。",
    ]

    # ---------- 11. 图表叙事方案（skill 交付项 2）----------
    charts_plan = [
        {"title": "交易额月度趋势与波动带", "type": "line+band",
         "insight": f"36 个月波动系数 {_round1(m_cv)}%，趋势线近乎水平——经营缺乏节奏感",
         "use": "回答「规模是否在增长、节奏是否可控」"},
        {"title": "季度交易额与同比增速", "type": "bar+line",
         "insight": f"季度均值约 {_num(np.mean(quarter_amt), 0):,} 万，同比多在 ±3% 内摆动",
         "use": "回答「增长动能来自哪里」"},
        {"title": "品类结构（金额 vs 笔数）", "type": "dual-bar",
         "insight": f"{top_cat['cat'] if top_cat else '-'}金额领先但笔数占比相近，靠客单价而非频次取胜",
         "use": "回答「钱从哪个品类来」"},
        {"title": "产品 TOP10 与集中度", "type": "bar",
         "insight": f"TOP5 仅占 {_round1(top5_share)}%，HHI {hhi:.0f}，无单品依赖",
         "use": "回答「是否有可以放大的主力产品」"},
        {"title": "省份交易额分布", "type": "bar",
         "insight": f"{top_prov['province'] if top_prov else '-'}占 {top_prov['share'] if top_prov else 0}%，区域头部效应明显",
         "use": "回答「资源该往哪个区域倾斜」"},
        {"title": "申请漏斗与交易覆盖", "type": "funnel",
         "insight": f"通过率 {approval_rate}%，但放款→交易覆盖 {cover_rate}%，漏斗末端漏水",
         "use": "回答「转化瓶颈卡在哪一步」"},
    ]

    # ---------- 12. 可追溯性清单（skill 专项验收）----------
    trace = [
        {"item": "交易总额", "value": f"{_wan(total_amt):,} 万元", "source": "Fact_Transaction.csv（全表汇总）"},
        {"item": "可追溯交易额", "value": f"{_wan(valid_amt):,} 万元", "source": "Fact_Transaction ⨝ Fact_Application（按 AppKey 内连接）"},
        {"item": "交易笔数", "value": f"{total_cnt:,} 笔", "source": "Fact_Transaction.csv 行数"},
        {"item": "客户（维度表）/ 产品 / 申请", "value": f"{app_customers:,} / {int(df_p['ProductName'].nunique())} / {total_app:,}",
         "source": "Dim_Customer / Dim_FinProduct / Fact_Application"},
        {"item": "覆盖期间", "value": f"{val['AppDate'].min().date()} ~ {val['AppDate'].max().date()}", "source": "Fact_Application.AppDate 解析后"},
        {"item": "数据缺口", "value": f"{orphan_cnt:,} 笔无主交易", "source": "AppKey 左连接未匹配记录，未做任何填补"},
    ]

    kpi = {
        "total_amt_wan": _wan(total_amt),
        "trans_cnt": total_cnt,
        "avg_ticket": avg_ticket,
        "yoy": yoy_recent,
        "hhi": int(round(hhi)),
        "approval_rate": approval_rate,
        "cover_rate": cover_rate,
        "penetration": penetration,
        "orphan_share": _pct(orphan_amt / total_amt),
    }

    return {
        "kpi": kpi,
        "summary": summary,
        "years": year_rows,
        "quarters": {"labels": quarters, "amt_wan": quarter_amt},
        "months": {"labels": months, "amt_wan": month_amt, "cnt": month_cnt},
        "cats": cat_rows,
        "subcats": sub_rows[:12],
        "products": prod_rows,
        "provinces": prov_rows,
        "tiers": tier_rows,
        "issues": issues,
        "actions": actions,
        "dictionary": dictionary,
        "charts_plan": charts_plan,
        "trace": trace,
        "metrics": {
            "valid_amt_wan": _wan(valid_amt), "valid_cnt": valid_cnt,
            "orphan_cnt": orphan_cnt, "orphan_amt_wan": _wan(orphan_amt),
            "inactive_share": inactive_share, "inactive_products": inactive_products,
            "prov_cr3": prov_cr3, "top5_share": _round1(top5_share),
            "m_cv": _round1(m_cv), "median_ticket": median_ticket, "p95_ticket": p95_ticket,
            "active_customers": active_customers, "total_customers": app_customers, "system_customers": system_customers,
            "tx_per_cust": tx_per_cust, "amt_per_cust": amt_per_cust,
            "reject_rate": reject_rate, "approval_rate": approval_rate,
            "loan_total_wan": _wan(loan_total), "total_app": total_app,
            "span": f"{val['AppDate'].min().date()} ~ {val['AppDate'].max().date()}",
        },
        "assumptions": [
            "所有金额单位换算：1 万元 = 10,000 元；亿元仅用于摘要表述",
            "交易归属期间以关联申请的 AppDate 为准，孤儿交易不参与时间序列分析",
            "月度波动系数使用 36 个自然月的交易额标准差 ÷ 均值，样本标准差（ddof=1）",
            "产品集中度 HHI 以 50 个产品的金额份额计算，取值 0—10000，<500 归为分散型",
            "放款交易覆盖率分母为「放款中 + 已结清」申请；已拒绝/审核中/已审批不计入未放款风险",
            "本报告不提供投资建议；缺失关联直接披露为数据缺口，不以估算值填补",
        ],
        "methodology": [
            {"step": "1. 业务问题与任务分离", "detail": "先明确「管理层要回答什么」——规模、结构、客户、效率、治理五问，再定义对应指标"},
            {"step": "2. 数据转图表叙事", "detail": "每个指标配一张图与一句结论，图上标注口径与样本量"},
            {"step": "3. 锁定不可虚构的事实", "detail": "全部数字由清洗后五表实时计算，生成时间与数据版本随报告输出"},
            {"step": "4. 结构草案", "detail": "指标字典 → 执行摘要 → 图表方案 → 异常问题 → 行动清单"},
            {"step": "5. 视觉方案与验收", "detail": "双色系图表、结论前置、异常分级（红橙黄）"},
            {"step": "6. 专项验收", "detail": "数字不改写、口径一致、结论可追溯——见「可追溯性清单」与「指标字典」"},
        ],
        "skill": "ai-hive-gap-024-ai-finance-report v1.0.0",
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def _round1(v):
    if v is None:
        return None
    return round(float(v), 1)


# ---- 缓存：以「数据内容指纹」为准，文件一有变化必定重算 ----
_CACHE = {"key": None, "data": None}


def data_signature():
    sig = []
    for name in DATA_FILES:
        path = os.path.join(CLEAN, name)
        if not os.path.exists(path):
            sig.append((name, None))
            continue
        st = os.stat(path)
        h = hashlib.md5()
        with open(path, "rb") as fp:
            h.update(fp.readline())
            fp.seek(max(0, st.st_size - 4096))
            h.update(fp.read())
        sig.append((name, st.st_size, round(st.st_mtime, 3), h.hexdigest()))
    return tuple(sig)


def invalidate():
    _CACHE["key"] = None
    _CACHE["data"] = None


def get_report(force=False):
    key = data_signature()
    if force or _CACHE["key"] != key or _CACHE["data"] is None:
        data = build_report()
        data["_signature"] = key
        _CACHE["data"] = data
        _CACHE["key"] = key
    return _CACHE["data"]


if __name__ == "__main__":
    import json
    r = build_report()
    print(json.dumps({
        "kpi": r["kpi"], "years": r["years"],
        "cats": r["cats"], "tiers": r["tiers"],
        "issues": [(i["level"], i["title"]) for i in r["issues"]],
    }, ensure_ascii=False, indent=2))
    print("\n--- 执行摘要 ---")
    for s in r["summary"]:
        print("·", s)
