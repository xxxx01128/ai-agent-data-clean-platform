# -*- coding: utf-8 -*-
"""
CLV 智能分析引擎
方法论来源：customer-lifetime-value-banking skill
（分群 → FTP 调整后收入建模 → 成本建模 → 流失生存模型 → NPV 折现 CLV → 价值杠杆建议）

输入：raw_data/cleaned_output 下的星型模型五张表
输出：结构化分析报告 dict，供 /api/ai_report 返回给前端渲染
"""
import os
import numpy as np
import pandas as pd
from datetime import datetime

BASE = os.path.dirname(os.path.abspath(__file__))
CLEAN = os.path.join(BASE, "raw_data", "cleaned_output")

# ============ 模型参数（可在报告中展示，保证可解释）============
FTP_MARGIN = 0.016      # 存款 FTP 净息差
WEALTH_FEE = 0.0030     # 财富管理/资管综合费率（按 AUM）
LOAN_SPREAD = 0.028     # 贷款生息差
TRANS_FEE = 1.5         # 单笔交易手续费（元）
DISCOUNT = 0.12         # 风险调整折现率（权益成本 12%）
GROWTH = 0.03           # 收入年增长
HORIZON = 20            # 建模年限

SEGMENTS = [
    ("大众客户", 0, 200000, 1200),
    ("富裕客户", 200000, 1000000, 5000),
    ("高净值客户", 1000000, 10000000, 30000),
    ("超高净值客户", 10000000, float("inf"), 100000),
]
SEG_MAP_SERVICE = {"大众客户": 260, "富裕客户": 900, "高净值客户": 6500, "超高净值客户": 28000}


def _read(name):
    path = os.path.join(CLEAN, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"缺少数据表：{name}")
    return pd.read_csv(path, encoding="utf-8-sig")


def _parse_date(series):
    s = series.astype(str).str.replace(r"[‑–—−]", "-", regex=True)
    return pd.to_datetime(s, errors="coerce", format="mixed")


def _survival(churn_y1, years=HORIZON, floor=0.015):
    """衰减型风险函数：首年流失最高，随后逐年衰减，不低于 floor"""
    h = min(0.40, max(churn_y1, floor))
    probs, surv, s = [], [], 1.0
    for t in range(1, years + 1):
        ht = max(floor, h * (0.82 ** (t - 1)))
        s = s * (1 - ht)
        surv.append(s)
    return surv  # S(t) for t=1..years


def _clv(rev, cost, surv, r=DISCOUNT, g=GROWTH):
    total = 0.0
    for t, s in enumerate(surv, start=1):
        total += s * (rev * ((1 + g) ** (t - 1)) - cost) / ((1 + r) ** t)
    return total


def _wan(x):
    """元 → 万元"""
    return round(x / 10000.0, 1)


def _pct(x, d=1):
    return round(x * 100, d)


def build_report():
    now = datetime.now()
    df_cust = _read("Dim_Customer.csv")
    df_prod = _read("Dim_FinProduct.csv")
    df_app = _read("Fact_Application.csv")
    df_trans = _read("Fact_Transaction.csv")

    # ---------- 客户级特征 ----------
    df_cust["RegDate"] = _parse_date(df_cust["RegisterDate"])
    df_cust["Tenure"] = ((pd.Timestamp(now) - df_cust["RegDate"]).dt.days / 365.25).clip(0.1, 30)
    df_cust["Tenure"] = df_cust["Tenure"].fillna(1.0)
    df_cust["AUM"] = pd.to_numeric(df_cust["AUM"], errors="coerce").fillna(0)
    df_cust["IsRetained"] = df_cust["IsRetained"].astype(str).str.upper().isin(["TRUE", "1", "是"])

    # 产品持有数（历史申请中的不同产品数）
    prod_cnt = df_app.groupby("CustomerKey")["ProductKey"].nunique().rename("ProdCnt")
    loan_amt = df_app.groupby("CustomerKey")["TotalAmount"].sum().rename("LoanAmt")
    bad_amt = (df_app[df_app["IsBadDebt"] == 1].groupby("CustomerKey")["TotalAmount"]
               .sum().rename("BadAmt"))

    # 交易笔数（Transaction → Application → Customer）
    trans_cnt = (df_trans.merge(df_app[["AppKey", "CustomerKey"]], on="AppKey", how="inner")
                 .groupby("CustomerKey").size().rename("TransCnt"))

    df_cust = (df_cust.merge(prod_cnt, on="CustomerKey", how="left")
                      .merge(loan_amt, on="CustomerKey", how="left")
                      .merge(bad_amt, on="CustomerKey", how="left")
                      .merge(trans_cnt, on="CustomerKey", how="left"))
    for c in ["ProdCnt", "LoanAmt", "BadAmt", "TransCnt"]:
        df_cust[c] = df_cust[c].fillna(0)

    # ---------- 分群 ----------
    bins = [s[1] for s in SEGMENTS] + [float("inf")]
    labels = [s[0] for s in SEGMENTS]
    df_cust["Segment"] = pd.cut(df_cust["AUM"], bins=bins, labels=labels, right=False).astype(str)

    # ---------- 客户级年化收入 / 成本 ----------
    df_cust["RevYr"] = (df_cust["AUM"] * (FTP_MARGIN + WEALTH_FEE)
                        + (df_cust["LoanAmt"] / df_cust["Tenure"]) * LOAN_SPREAD
                        + (df_cust["TransCnt"] / df_cust["Tenure"]) * TRANS_FEE)
    df_cust["CostYr"] = (df_cust["Segment"].map(SEG_MAP_SERVICE).fillna(300)
                         + (df_cust["BadAmt"] / df_cust["Tenure"]))
    df_cust["Margin"] = df_cust["RevYr"] - df_cust["CostYr"]

    # ---------- 分群聚合 + CLV ----------
    seg_rows, chart_clv, chart_ratio = [], [], []
    surv_series = []
    year_axis = [f"Y{t}" for t in range(1, HORIZON + 1)]
    total_cust = len(df_cust)
    total_aum = float(df_cust["AUM"].sum())
    total_clv_sum = 0.0
    seg_detail = {}

    for name, lo, hi, cac in SEGMENTS:
        sub = df_cust[df_cust["Segment"] == name]
        n = len(sub)
        if n == 0:
            continue
        retain = float(sub["IsRetained"].mean())
        churn = 1 - retain
        surv = _survival(churn)
        rev = float(sub["RevYr"].mean())
        cost = float(sub["CostYr"].mean())
        clv = _clv(rev, cost, surv)
        lifetime = float(np.sum(surv))
        ratio = clv / cac if cac > 0 else 0

        total_clv_sum += clv * n
        seg_detail[name] = {
            "n": n, "retain": retain, "churn": churn, "rev": rev, "cost": cost,
            "clv": clv, "lifetime": lifetime, "cac": cac, "ratio": ratio,
            "aum": float(sub["AUM"].mean()),
        }
        seg_rows.append({
            "name": name,
            "count": n,
            "share": _pct(n / total_cust),
            "avg_aum": _wan(sub["AUM"].mean()),
            "avg_clv": _wan(clv),
            "lifetime": round(lifetime, 1),
            "rev_yr": _wan(rev),
            "cost_yr": _wan(cost),
            "cac": cac,
            "ratio": round(ratio, 1),
            "retention": _pct(retain),
            "total_clv_wan": _wan(clv * n),
        })
        chart_clv.append({"name": name, "value": _wan(clv)})
        chart_ratio.append({"name": name, "value": round(ratio, 1)})
        surv_series.append({"name": name, "data": [round(s * 100, 1) for s in surv]})

    df_cust["CLV"] = df_cust["Segment"].map(
        {k: v["clv"] for k, v in seg_detail.items()}).fillna(0)

    # ---------- 产品持有数 → 留存效应 ----------
    def bucket(k):
        if k <= 0:
            return "0个"
        return f"{int(min(k, 4))}个" if k < 4 else "4个及以上"

    df_cust["ProdBucket"] = df_cust["ProdCnt"].apply(bucket)
    order = ["1个", "2个", "3个", "4个及以上"]
    prod_effect = []
    for b in order:
        sub = df_cust[df_cust["ProdBucket"] == b]
        if len(sub) == 0:
            continue
        prod_effect.append({
            "products": b,
            "count": len(sub),
            "share": _pct(len(sub) / total_cust),
            "retention": _pct(float(sub["IsRetained"].mean())),
            "avg_aum": _wan(sub["AUM"].mean()),
            "avg_clv": _wan(float(sub["CLV"].mean())),
        })

    # ---------- 客户年限（生命周期阶段）----------
    def tenure_band(t):
        if t < 1:
            return "0-1年（新客）"
        if t < 3:
            return "1-3年（成长期）"
        if t < 5:
            return "3-5年（成熟期）"
        return "5年以上（深度客户）"

    df_cust["TenureBand"] = df_cust["Tenure"].apply(tenure_band)
    tenure_rows = []
    for b in ["0-1年（新客）", "1-3年（成长期）", "3-5年（成熟期）", "5年以上（深度客户）"]:
        sub = df_cust[df_cust["TenureBand"] == b]
        if len(sub) == 0:
            continue
        tenure_rows.append({
            "band": b, "count": len(sub), "share": _pct(len(sub) / total_cust),
            "retention": _pct(float(sub["IsRetained"].mean())),
            "avg_aum": _wan(sub["AUM"].mean()), "avg_clv": _wan(float(sub["CLV"].mean())),
        })

    # ---------- 高风险 / 流失客户价值敞口 ----------
    risk_high = df_cust[df_cust["RiskLevel"].astype(str).str.title().isin(["High", "Blacklist"])]
    churned = df_cust[~df_cust["IsRetained"]]
    hv_churn = churned[churned["AUM"] >= 1000000]
    risk_block = {
        "high_risk_cnt": int(len(risk_high)),
        "high_risk_share": _pct(len(risk_high) / total_cust),
        "high_risk_clv_wan": _wan(float(risk_high["CLV"].sum())),
        "churn_cnt": int(len(churned)),
        "churn_rate": _pct(len(churned) / total_cust),
        "churn_clv_wan": _wan(float(churned["CLV"].sum())),
        "hv_churn_cnt": int(len(hv_churn)),
        "hv_churn_clv_wan": _wan(float(hv_churn["CLV"].sum())),
        "bad_debt_cnt": int((df_app["IsBadDebt"] == 1).sum()),
        "bad_debt_rate": _pct(float((df_app["IsBadDebt"] == 1).mean())),
    }

    # ---------- 城市价值 TOP ----------
    city = (df_cust.groupby("City")
            .agg(count=("CustomerKey", "count"), clv=("CLV", "sum"), aum=("AUM", "sum"),
                 retain=("IsRetained", "mean"))
            .reset_index())
    city["avg_clv"] = city["clv"] / city["count"]
    city = city.sort_values("clv", ascending=False).head(8)
    city_rows = [{
        "city": r["City"], "count": int(r["count"]),
        "total_clv_wan": _wan(r["clv"]), "avg_clv": _wan(r["avg_clv"]),
        "retention": _pct(r["retain"]),
    } for _, r in city.iterrows()]

    # ---------- 四大价值杠杆（Extend / Expand / Economize / Acquire）----------
    seg_sorted_clv = sorted(seg_detail.items(), key=lambda kv: kv[1]["clv"], reverse=True)
    seg_sorted_ratio = sorted(seg_detail.items(), key=lambda kv: kv[1]["ratio"], reverse=True)
    top_clv_name = seg_sorted_clv[0][0]
    top_ratio_name = seg_sorted_ratio[0][0]
    retain_1 = next((p for p in prod_effect if p["products"] == "1个"), None)
    retain_2 = next((p for p in prod_effect if p["products"] == "2个"), None)
    lift = ((retain_2["retention"] - retain_1["retention"]) / 100.0) if retain_1 and retain_2 else 0.08
    lift = max(0.02, min(lift, 0.35))

    mass = seg_detail.get("大众客户")
    single_affluent = df_cust[(df_cust["Segment"].isin(["富裕客户", "高净值客户"]))
                              & (df_cust["ProdCnt"] <= 1)]
    levers = []

    # Extend：降低大众客户流失
    if mass and mass["n"] > 0:
        cut = 0.03  # 流失率降低 3 个百分点
        impact = cut * mass["n"] * mass["clv"]
        invest = cut * mass["n"] * 260
        levers.append({
            "lever": "Extend 延长关系周期", "priority": "P1",
            "target": "大众客户",
            "action": f"对大众客户（{mass['n']:,}人）实施激活式留存：绑定代发工资/自动扣款、"
                      f"睡眠户唤醒，目标把年流失率从 {_pct(mass['churn'])}% 降低 3 个百分点",
            "impact_wan": _wan(impact), "invest_wan": _wan(invest),
            "roi": round(impact / invest, 1) if invest > 0 else 0,
        })
    # Expand：单产品富裕客户交叉销售
    if len(single_affluent) > 0:
        uplift = float(single_affluent["CLV"].mean()) * lift
        impact = uplift * len(single_affluent)
        invest = len(single_affluent) * 320
        levers.append({
            "lever": "Expand 提升单客价值", "priority": "P1",
            "target": "富裕/高净值单产品客户",
            "action": f"对 {len(single_affluent):,} 名仅持 1 个产品的富裕/高净值客户做定向交叉销售"
                      f"（第2张产品），预计留存率提升 {_pct(lift)} 个百分点、单客 CLV 提升 "
                      f"{_wan(uplift)} 万元",
            "impact_wan": _wan(impact), "invest_wan": _wan(invest),
            "roi": round(impact / invest, 1) if invest > 0 else 0,
        })
    # Economize：大众客户服务渠道迁移
    if mass and mass["n"] > 0:
        save = mass["cost"] * 0.18 * mass["n"]
        invest = mass["n"] * 25
        levers.append({
            "lever": "Economize 降低服务成本", "priority": "P2",
            "target": "大众客户",
            "action": "把大众客户 18% 的柜面/人工服务量迁移到手机银行与智能客服，"
                      "并对低效网点做轻型化改造",
            "impact_wan": _wan(save), "invest_wan": _wan(invest),
            "roi": round(save / invest, 1) if invest > 0 else 0,
        })
    # Acquire：高 CLV/CAC 段加大获客
    best = seg_detail[top_ratio_name]
    budget = 300 * 10000  # 300万元新增获客预算
    new_cust = budget / best["cac"]
    impact = new_cust * best["clv"]
    levers.append({
        "lever": "Acquire 高效获客", "priority": "P1",
        "target": top_ratio_name,
        "action": f"{top_ratio_name}的 CLV/CAC 达 {best['ratio']:.1f}x（全行最高），"
                  f"建议追加 300 万元获客预算，按 CAC {best['cac']:,} 元计算可新增约 "
                  f"{int(new_cust):,} 户，同时把 CAC 上限控制在 CLV/3 以内",
        "impact_wan": _wan(impact), "invest_wan": 300,
        "roi": round(impact / budget, 1) if budget > 0 else 0,
    })

    # ---------- 执行摘要（数据驱动的动态结论）----------
    top_seg = seg_detail[top_clv_name]
    top_share = _pct(top_seg["n"] / total_cust)
    top_value_share = _pct((top_seg["clv"] * top_seg["n"]) / total_clv_sum) if total_clv_sum > 0 else 0
    avg_clv_all = total_clv_sum / total_cust if total_cust else 0
    total_cac_spend = sum(v["cac"] * v["n"] for v in seg_detail.values())
    overall_ratio = total_clv_sum / total_cac_spend if total_cac_spend > 0 else 0
    w_lifetime = (sum(v["lifetime"] * v["n"] for v in seg_detail.values()) / total_cust
                  if total_cust else 0)
    r1 = retain_1["retention"] if retain_1 else 0
    r2 = retain_2["retention"] if retain_2 else 0
    r4 = next((p["retention"] for p in prod_effect if p["products"] == "4个及以上"), 0)
    if r4 - r1 >= 3:
        prod_text = (f"留存是 CLV 的第一杠杆：产品持有数从 1 个提升到 2 个，留存率由 {r1}% 升至 {r2}%，"
                     f"持有 4 个及以上产品的客户留存率达 {r4}%，印证「产品数是最强留存预测因子」的行业规律。")
    else:
        prod_text = (f"产品数与留存的关系偏弱：1 个产品留存 {r1}%、2 个产品 {r2}%、4 个及以上 {r4}%，"
                     f"说明当前交叉销售更多停留在「产品数量叠加」，尚未形成主账户/代发工资式的关系绑定，"
                     f"留存提升空间在「关系深度」而非「产品个数」。")

    summary = [
        f"全行 {total_cust:,} 名客户合计 CLV 约 {_wan(total_clv_sum):,} 万元，人均 {_wan(avg_clv_all)} 万元，"
        f"客户加权平均预期关系年限 {w_lifetime:.1f} 年，整体 CLV/CAC 为 {overall_ratio:.1f}x。",

        (f"价值高度集中：{top_clv_name}仅占客户数 {top_share}%，却贡献了约 {top_value_share}% 的客户总价值，"
         f"人均 CLV {_wan(top_seg['clv'])} 万元，是大众客户的 "
         f"{top_seg['clv'] / seg_detail['大众客户']['clv']:.1f} 倍。"
         if "大众客户" in seg_detail and seg_detail["大众客户"]["clv"] > 0
         else f"价值高度集中：{top_clv_name}人均 CLV 最高，达 {_wan(top_seg['clv'])} 万元。"),

        prod_text,

        (f"当前流失客户 {risk_block['churn_cnt']:,} 户（流失率 {risk_block['churn_rate']}%），"
         f"对应 CLV 敞口约 {risk_block['churn_clv_wan']:,} 万元；其中高净值流失客户 "
         f"{risk_block['hv_churn_cnt']:,} 户，价值敞口 {risk_block['hv_churn_clv_wan']:,} 万元，应优先挽回。"),

        (f"获客效率最优的是{top_ratio_name}（CLV/CAC = {best['ratio']:.1f}x），建议把新增营销预算"
         f"向该客群倾斜，并按「CAC ≤ CLV/3」设置准入门槛。"),
    ]

    kpi = {
        "total_clv_wan": _wan(total_clv_sum),
        "avg_clv_wan": _wan(avg_clv_all),
        "avg_lifetime": round(w_lifetime, 1),
        "overall_ratio": round(overall_ratio, 1),
        "churn_rate": risk_block["churn_rate"],
        "at_risk_wan": risk_block["churn_clv_wan"],
    }

    actions = [
        {"when": "0-30 天", "owner": "零售业务部",
         "task": f"锁定 {risk_block['hv_churn_cnt']:,} 户高净值流失/预警客户，由客户经理 1v1 回访，"
                 f"配套费率优惠与专属权益",
         "kpi": f"挽回率 ≥ 30%，挽回 CLV ≥ {_wan(risk_block['hv_churn_clv_wan'] * 0.3)} 万元"},
        {"when": "0-30 天", "owner": "数据/营销中台",
         "task": f"建立「单产品富裕客户」清单（{len(single_affluent):,} 户），上线第二产品推荐模型与话术",
         "kpi": f"触达转化率 ≥ 8%，年化新增 CLV ≥ {_wan(float(single_affluent['CLV'].mean()) * lift * len(single_affluent) * 0.08)} 万元"},
        {"when": "30-90 天", "owner": "渠道管理部",
         "task": "大众客户柜面业务向手机银行迁移专项，设置迁移率周度看板与网点考核",
         "kpi": "电子渠道替代率提升 15 个百分点，年化节约服务成本 15%+"},
        {"when": "30-90 天", "owner": "市场营销部",
         "task": f"将获客预算向{top_ratio_name}倾斜，按 CAC ≤ {int(best['clv'] / 3):,} 元设定渠道准入线，"
                 f"淘汰 CLV/CAC < 2x 的渠道",
         "kpi": f"新增{top_ratio_name} ≥ {int(new_cust * 0.5):,} 户，整体 CLV/CAC 提升 10%"},
        {"when": "90-180 天", "owner": "风险管理部",
         "task": f"对 {risk_block['high_risk_cnt']:,} 户高风险/黑名单客户做额度与定价重检，控制信用成本",
         "kpi": f"高风险客群信用成本下降 10%，释放 CLV {_wan(risk_block['high_risk_clv_wan'] * 0.1)} 万元"},
    ]

    assumptions = [
        f"存款 FTP 净息差 {_pct(FTP_MARGIN,2)}%、财富管理综合费率 {_pct(WEALTH_FEE,2)}%（均按 AUM 计提）",
        f"贷款生息差 {_pct(LOAN_SPREAD,2)}%，按客户历史申请金额年化折算",
        f"单笔交易手续费 {TRANS_FEE} 元，按客户历史交易笔数年化",
        f"服务成本：大众 {SEG_MAP_SERVICE['大众客户']} 元/年、富裕 {SEG_MAP_SERVICE['富裕客户']} 元/年、"
        f"高净值 {SEG_MAP_SERVICE['高净值客户']} 元/年、超高净值 {SEG_MAP_SERVICE['超高净值客户']} 元/年",
        f"风险调整折现率 {_pct(DISCOUNT,0)}%（权益成本口径），收入年增长 {_pct(GROWTH,0)}%，建模年限 {HORIZON} 年",
        "流失率基线取客户表留存标记（IsRetained），并按衰减型风险函数建模：首年流失率 = 基线×1.8，逐年 ×0.82 衰减",
        "CAC（单客获客成本）采用行业经验值：大众 1,200 元、富裕 5,000 元、高净值 30,000 元、超高净值 100,000 元",
    ]

    methodology = [
        "Step 1 客户分群：按 AUM 划分大众 / 富裕 / 高净值 / 超高净值，满足可行动、可度量、有规模三项要求",
        "Step 2 收入建模：存款 NII（FTP 调整）+ 财富管理费 + 贷款生息 + 交易手续费，逐客户年化",
        "Step 3 成本建模：分群服务成本（渠道 + 客户经理）+ 预期信用成本",
        "Step 4 流失建模：衰减型风险函数构造生存曲线 S(t)，期望关系年限 = ΣS(t)",
        "Step 5 CLV 计算：CLV = Σ S(t)×(Revenue(t)−Cost(t))/(1+r)^t",
        "Step 6 分层分析：分群 CLV/CAC、产品数留存效应、客户年限阶段、城市价值、风险敞口",
        "Step 7 战略建议：按 Extend（延长）/ Expand（扩张）/ Economize（降本）/ Acquire（高效获客）四杠杆量化 ROI",
    ]

    return {
        "generated_at": now.strftime("%Y-%m-%d %H:%M:%S"),
        "data": {
            "customers": total_cust,
            "applications": int(len(df_app)),
            "transactions": int(len(df_trans)),
            "products": int(len(df_prod)),
            "total_aum_wan": _wan(total_aum),
        },
        "kpi": kpi,
        "summary": summary,
        "segments": seg_rows,
        "charts": {
            "year_axis": year_axis,
            "survival": surv_series,
            "clv_bar": chart_clv,
            "ratio_bar": chart_ratio,
        },
        "product_effect": prod_effect,
        "tenure": tenure_rows,
        "cities": city_rows,
        "risk": risk_block,
        "levers": levers,
        "actions": actions,
        "assumptions": assumptions,
        "methodology": methodology,
        "skill": "customer-lifetime-value-banking v1.0.0",
    }


# ---- 缓存：以「数据内容指纹」为准，文件一有变化必定重算 ----
_CACHE = {"key": None, "data": None}

DATA_FILES = [
    "Dim_Branch.csv", "Dim_Customer.csv", "Dim_FinProduct.csv",
    "Fact_Application.csv", "Fact_Transaction.csv",
]


def data_signature():
    """五张清洗表的内容指纹：文件名 + 大小 + mtime + 首尾行哈希。
    任何一个文件被上传覆盖（哪怕内容变短、时间戳精度不足）都会得到不同指纹。"""
    import hashlib
    sig = []
    for name in DATA_FILES:
        path = os.path.join(CLEAN, name)
        if not os.path.exists(path):
            sig.append((name, None))
            continue
        st = os.stat(path)
        h = hashlib.md5()
        with open(path, "rb") as fp:
            h.update(fp.readline())          # 表头
            fp.seek(max(0, st.st_size - 4096))  # 末尾 4KB（数据尾部）
            h.update(fp.read())
        sig.append((name, st.st_size, round(st.st_mtime, 3), h.hexdigest()))
    return tuple(sig)


def invalidate():
    """上传/清洗完成后调用，确保下一次分析一定基于新数据重算。"""
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
    print(json.dumps({k: r[k] for k in ["kpi", "segments", "summary"]}, ensure_ascii=False, indent=2))
