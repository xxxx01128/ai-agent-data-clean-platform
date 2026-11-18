# -*- coding: utf-8 -*-
"""
金融风控三页可视化看板 (Flask + ECharts)
真实字段匹配 + 全部图表支持点击下钻
"""
from flask import Flask, render_template, jsonify, request
import pandas as pd
import os
import shutil
import hashlib
from datetime import datetime
from werkzeug.utils import secure_filename
from eda_process import TABLE_NAMES, read_upload, clean_dataframe, match_table

app = Flask(__name__)

BASE = os.path.dirname(os.path.abspath(__file__))
CLEAN = os.path.join(BASE, "raw_data", "cleaned_output")
ORIGINAL = os.path.join(BASE, "raw_data", "original")
os.makedirs(ORIGINAL, exist_ok=True)

app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 上传上限 50MB

TIER_MAP = {"PLATINUM": "铂金客户", "GOLD": "黄金客户", "SILVER": "白银客户", "BRONZE": "青铜客户"}
RISK_MAP = {"Low": "低风险", "Medium": "中风险", "High": "高风险", "Blacklist": "黑名单"}
AUM_BINS = [0, 50000, 200000, 1000000, 1e12]
AUM_LABELS = ["0-5万", "5-20万", "20-100万", "100万以上"]


def read_all():
    df_branch = pd.read_csv(os.path.join(CLEAN, "Dim_Branch.csv"), encoding="utf-8-sig")
    df_customer = pd.read_csv(os.path.join(CLEAN, "Dim_Customer.csv"), encoding="utf-8-sig")
    df_product = pd.read_csv(os.path.join(CLEAN, "Dim_FinProduct.csv"), encoding="utf-8-sig")
    df_apply = pd.read_csv(os.path.join(CLEAN, "Fact_Application.csv"), encoding="utf-8-sig")
    df_trans = pd.read_csv(os.path.join(CLEAN, "Fact_Transaction.csv"), encoding="utf-8-sig")
    # 客户表无省份列，关联网点表取省份
    df_customer = pd.merge(df_customer, df_branch[["BranchKey", "Province"]], on="BranchKey", how="left")
    df_customer["Province"] = df_customer["Province"].fillna("未知")
    return df_branch, df_customer, df_product, df_apply, df_trans


def to_records(sr):
    df = sr.reset_index()
    df.columns = ["name", "value"]
    return df.to_dict("records")


@app.route("/")
def index():
    return render_template("index_nav.html")


@app.route("/risk")
def risk():
    df_branch, df_customer, df_product, df_apply, df_trans = read_all()
    total_customer = len(df_customer)
    total_apply = len(df_apply)
    bad_cnt = int((df_apply["IsBadDebt"] == 1).sum())
    bad_rate = round(bad_cnt / total_apply, 4) if total_apply > 0 else 0
    bad_amt = round(float(df_apply[df_apply["IsBadDebt"] == 1]["TotalAmount"].sum()), 2)
    avg_hour = round(float(df_apply["ProcessHours"].mean()), 2) if total_apply > 0 else 0

    df_customer["RiskCN"] = df_customer["RiskLevel"].map(RISK_MAP).fillna(df_customer["RiskLevel"])
    risk_json = to_records(df_customer["RiskCN"].value_counts())
    status_json = to_records(df_apply["Status"].value_counts())
    province_json = to_records(df_customer["Province"].value_counts())

    bad_prod = pd.merge(df_apply[df_apply["IsBadDebt"] == 1], df_product, on="ProductKey")
    bad_prod_json = to_records(bad_prod.groupby("ProductName")["TotalAmount"].sum())

    df_apply["AppMonth"] = pd.to_datetime(df_apply["AppDate"], errors="coerce").dt.to_period("M").astype(str)
    bad_month = df_apply[df_apply["IsBadDebt"] == 1].groupby("AppMonth")["TotalAmount"].sum()
    bad_month_json = to_records(bad_month)

    return render_template(
        "risk.html",
        total_customer=total_customer,
        total_apply=total_apply,
        bad_cnt=bad_cnt,
        bad_rate=bad_rate,
        bad_amt=bad_amt,
        avg_hour=avg_hour,
        risk_json=risk_json,
        status_json=status_json,
        province_json=province_json,
        bad_prod_json=bad_prod_json,
        bad_month_json=bad_month_json,
    )


@app.route("/marketing")
def marketing():
    df_branch, df_customer, df_product, df_apply, df_trans = read_all()
    total_customer = len(df_customer)
    retain_cnt = int((df_customer["IsRetained"] == True).sum())
    retain_rate = round(retain_cnt / total_customer, 4) if total_customer > 0 else 0
    avg_aum = round(float(df_customer["AUM"].mean()), 2)
    high_net_cnt = int((df_customer["AUM"] > 1000000).sum())

    df_customer["TierCN"] = df_customer["Tier"].map(TIER_MAP).fillna(df_customer["Tier"])
    tier_json = to_records(df_customer["TierCN"].value_counts())
    retain_json = to_records(df_customer["IsRetained"].map({True: "留存客户", False: "流失客户"}).value_counts())

    df_customer["AUMBin"] = pd.cut(df_customer["AUM"], bins=AUM_BINS, labels=AUM_LABELS).astype(str)
    aum_json = to_records(df_customer["AUMBin"].value_counts())
    city_json = to_records(df_customer["City"].value_counts().head(12))

    return render_template(
        "marketing.html",
        total_customer=total_customer,
        retain_cnt=retain_cnt,
        retain_rate=retain_rate,
        avg_aum=avg_aum,
        high_net_cnt=high_net_cnt,
        tier_json=tier_json,
        retain_json=retain_json,
        aum_json=aum_json,
        city_json=city_json,
    )


@app.route("/business")
def business():
    df_branch, df_customer, df_product, df_apply, df_trans = read_all()
    total_amt = round(float(df_trans["TransAmount"].sum()), 2)
    trans_cnt = len(df_trans)
    avg_single = round(float(df_trans["TransAmount"].mean()), 2) if trans_cnt > 0 else 0
    loan_total = round(float(df_apply["TotalAmount"].sum()), 2)

    df_apply["AppMonth"] = pd.to_datetime(df_apply["AppDate"], errors="coerce").dt.to_period("M").astype(str)
    df_trans_app = pd.merge(df_trans, df_apply[["AppKey", "AppMonth", "CustomerKey", "ProductKey"]], on="AppKey")

    trans_month_json = to_records(df_trans_app.groupby("AppMonth")["TransAmount"].sum())
    apply_month_json = to_records(df_apply["AppMonth"].value_counts().sort_index())
    prov_biz = pd.merge(df_trans_app, df_customer[["CustomerKey", "Province"]], on="CustomerKey")
    prov_biz_json = to_records(prov_biz.groupby("Province")["TransAmount"].sum())
    prod_biz = pd.merge(df_trans_app, df_product[["ProductKey", "ProductName"]], on="ProductKey")
    prod_biz_json = to_records(prod_biz.groupby("ProductName")["TransAmount"].sum())

    return render_template(
        "business.html",
        total_amt=total_amt,
        trans_cnt=trans_cnt,
        avg_single=avg_single,
        loan_total=loan_total,
        trans_month_json=trans_month_json,
        apply_month_json=apply_month_json,
        prov_biz_json=prov_biz_json,
        prod_biz_json=prod_biz_json,
    )


@app.route("/api/drill", methods=["POST"])
def api_drill():
    try:
        req = request.get_json()
        drill_type = req.get("type", "")
        key_val = req.get("value", "")
        df_branch, df_customer, df_product, df_apply, df_trans = read_all()

        df_customer["RiskCN"] = df_customer["RiskLevel"].map(RISK_MAP).fillna(df_customer["RiskLevel"])
        df_customer["TierCN"] = df_customer["Tier"].map(TIER_MAP).fillna(df_customer["Tier"])
        df_customer["AUMBin"] = pd.cut(df_customer["AUM"], bins=AUM_BINS, labels=AUM_LABELS).astype(str)
        df_apply["AppMonth"] = pd.to_datetime(df_apply["AppDate"], errors="coerce").dt.to_period("M").astype(str)

        res_data = []
        if drill_type == "province":
            sub = df_customer[df_customer["Province"] == key_val]
        elif drill_type == "risk":
            sub = df_customer[df_customer["RiskCN"] == key_val]
        elif drill_type == "tier":
            sub = df_customer[df_customer["TierCN"] == key_val]
        elif drill_type == "city":
            sub = df_customer[df_customer["City"] == key_val]
        elif drill_type == "aum":
            sub = df_customer[df_customer["AUMBin"] == key_val]
        elif drill_type == "status":
            sub = df_apply[df_apply["Status"] == key_val]
        elif drill_type == "product":
            prod_keys = df_product[df_product["ProductName"] == key_val]["ProductKey"].tolist()
            sub = df_apply[df_apply["ProductKey"].isin(prod_keys)]
        elif drill_type == "month":
            sub = df_apply[df_apply["AppMonth"] == key_val]
        elif drill_type == "retain":
            flag = (key_val == "留存客户")
            sub = df_customer[df_customer["IsRetained"] == flag]
        else:
            return jsonify([])

        res_data = sub.to_dict("records")
        return jsonify(res_data)
    except Exception as e:
        print(f"【下钻接口异常】{str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route("/upload")
def upload_page():
    return render_template("upload.html")


@app.route("/api/tables", methods=["GET"])
def api_tables():
    """返回当前五张表的基本信息（行数、更新时间）"""
    tables = []
    for name in TABLE_NAMES:
        path = os.path.join(CLEAN, name)
        info = {"name": name, "rows": 0, "updated": "-", "exists": os.path.exists(path)}
        if info["exists"]:
            try:
                info["rows"] = int(len(pd.read_csv(path, encoding="utf-8-sig")))
            except Exception:
                pass
            info["updated"] = datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y-%m-%d %H:%M:%S")
        tables.append(info)
    return jsonify({"version": data_version(), "tables": tables})


def data_version():
    """当前清洗数据的版本指纹：五张表的最新修改时间 + 行数组合。
    前端拿它跟报告里的版本号比对，就能判断报告是不是最新数据的。"""
    parts = []
    for name in TABLE_NAMES:
        path = os.path.join(CLEAN, name)
        if os.path.exists(path):
            st = os.stat(path)
            parts.append(f"{name}:{st.st_size}:{int(st.st_mtime)}")
        else:
            parts.append(f"{name}:-")
    return hashlib.md5("|".join(parts).encode()).hexdigest()[:12]


@app.route("/api/upload", methods=["POST"])
def api_upload():
    try:
        f = request.files.get("file")
        if f is None or f.filename == "":
            return jsonify({"ok": False, "msg": "未收到文件，请选择文件后上传"}), 400

        table = match_table(f.filename)
        if table is None:
            return jsonify({
                "ok": False,
                "msg": f"无法识别表类型，文件名需包含以下之一：{', '.join(t.replace('.csv', '') for t in TABLE_NAMES)}"
            }), 400

        # 读取上传文件
        df_raw = read_upload(f)
        raw_rows = len(df_raw)

        # 备份原有数据（保留最近一份）
        orig_path = os.path.join(ORIGINAL, table)
        clean_path = os.path.join(CLEAN, table)
        for p in (orig_path, clean_path):
            if os.path.exists(p):
                shutil.copy2(p, p + ".bak")

        # 清洗
        df_clean, steps = clean_dataframe(table, df_raw)

        # 覆盖保存
        df_raw.to_csv(orig_path, index=False, encoding="utf-8-sig")
        df_clean.to_csv(clean_path, index=False, encoding="utf-8-sig")

        # 数据已变 → 立即失效两个 AI 分析引擎的缓存，
        # 保证之后点击「AI 分析」拿到的一定是基于新数据的报告
        for module in ("clv_analysis", "risk_analysis", "business_analysis"):
            try:
                __import__(module).invalidate()
            except Exception as cache_err:
                print(f"【缓存失效警告】{module}: {cache_err}")

        # 追加清洗日志
        log_path = os.path.join(CLEAN, "clean_log.csv")
        log_row = pd.DataFrame([{
            "file": table,
            "before": raw_rows,
            "after": len(df_clean),
            "source": secure_filename(f.filename),
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }])
        if os.path.exists(log_path):
            try:
                old_log = pd.read_csv(log_path, encoding="utf-8-sig")
                log_row = pd.concat([old_log, log_row], ignore_index=True)
            except Exception:
                pass
        log_row.to_csv(log_path, index=False, encoding="utf-8-sig")

        # 预览清洗后数据（前 20 行，NaN 转空字符串）
        preview = df_clean.head(20).where(pd.notna(df_clean.head(20)), "")
        return jsonify({
            "ok": True,
            "table": table,
            "raw_rows": raw_rows,
            "clean_rows": int(len(df_clean)),
            "columns": list(df_clean.columns),
            "steps": steps,
            "preview": preview.to_dict("records"),
            "data_version": data_version(),
            "msg": f"{table} 已上传并完成清洗，看板数据已自动更新"
        })
    except Exception as e:
        return jsonify({"ok": False, "msg": f"上传/清洗失败：{str(e)}"}), 500


@app.route("/api/ai_report", methods=["GET"])
def api_ai_report():
    """CLV 智能分析报告（customer-lifetime-value-banking 方法论）"""
    try:
        import clv_analysis
        from clv_analysis import get_report
        force = request.args.get("force") == "1"
        report = get_report(force=force)   # 内部按数据指纹判断，数据变了必然重算
        report["data_version"] = data_version()
        sig = report.get("_signature")
        report["data_files"] = [s[1] for s in sig if s[1] is not None] if sig else []
        return jsonify(report)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/risk_report", methods=["GET"])
def api_risk_report():
    """信用风险 AI 分析报告（credit-risk-explanation 方法论）"""
    try:
        import risk_analysis
        from risk_analysis import get_report as get_risk_report
        force = request.args.get("force") == "1"
        report = get_risk_report(force=force)
        report["data_version"] = data_version()
        sig = report.get("_signature")
        report["data_files"] = [s[1] for s in sig if s[1] is not None] if sig else []
        return jsonify(report)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/business_report", methods=["GET"])
def api_business_report():
    """经营突破 AI 财务经营报告（ai-hive-gap-024-ai-finance-report 方法论）"""
    try:
        from business_analysis import get_report as get_biz_report
        force = request.args.get("force") == "1"
        report = get_biz_report(force=force)
        report["data_version"] = data_version()
        sig = report.get("_signature")
        report["data_files"] = [s[1] for s in sig if s[1] is not None] if sig else []
        return jsonify(report)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
