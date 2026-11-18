# -*- coding: utf-8 -*-
"""
数据清洗模块：既支持命令行批量清洗，也支持被 Flask 调用对单个上传文件清洗
用法1（命令行）: python eda_process.py
用法2（被调用）: from eda_process import clean_dataframe
"""
import pandas as pd
import os
import re
from datetime import datetime

INPUT_FOLDER = "raw_data/original"
OUTPUT_FOLDER = "raw_data/cleaned_output"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# 五张标准表（清洗逻辑按表名匹配）
TABLE_NAMES = [
    "Dim_Branch.csv",
    "Dim_Customer.csv",
    "Dim_FinProduct.csv",
    "Fact_Application.csv",
    "Fact_Transaction.csv",
]

# 城市 -> 正确省份映射（完全保留，不删）
city_to_province = {
    "郑州市":"河南",
    "新乡市":"河南",
    "洛阳市":"河南",
    "开封市":"河南",
    "安阳市":"河南",
    "广州市":"广东",
    "深圳市":"广东",
    "佛山市":"广东",
    "东莞市":"广东",
    "杭州市":"浙江",
    "宁波市":"浙江",
    "温州市":"浙江",
    "南京市":"江苏",
    "苏州市":"江苏",
    "无锡市":"江苏",
    "济南市":"山东",
    "青岛市":"山东",
    "成都市":"四川",
    "武汉市":"湖北",
    "长沙市":"湖南",
    "福州市":"福建",
    "厦门市":"福建",
    "合肥市":"安徽",
    "石家庄市":"河北",
    "重庆市":"重庆",
    "郑州":"河南",
    "新乡":"河南",
    "洛阳":"河南",
    "开封":"河南",
    "安阳":"河南",
    "广州":"广东",
    "深圳":"广东",
    "佛山":"广东",
    "东莞":"广东",
    "杭州":"浙江",
    "宁波":"浙江",
    "温州":"浙江",
    "南京":"江苏",
    "苏州":"江苏",
    "无锡":"江苏",
    "济南":"山东",
    "青岛":"山东",
    "成都":"四川",
    "武汉":"湖北",
    "长沙":"湖南",
    "福州":"福建",
    "厦门":"福建",
    "合肥":"安徽",
    "石家庄":"河北",
    "重庆":"重庆"
}


def read_safe(filepath):
    try:
        return pd.read_csv(filepath, encoding="utf-8-sig")
    except Exception:
        return pd.read_csv(filepath, encoding="gbk")


def read_upload(file_storage):
    """读取上传的文件对象，支持 csv / xls / xlsx，自动尝试多种编码"""
    filename = file_storage.filename.lower()
    if filename.endswith((".xls", ".xlsx")):
        return pd.read_excel(file_storage)
    # csv：依次尝试常见编码
    for enc in ("utf-8-sig", "gbk", "gb18030", "utf-8"):
        try:
            file_storage.stream.seek(0)
            return pd.read_csv(file_storage, encoding=enc)
        except Exception:
            continue
    raise ValueError("无法解析文件，请确认是 CSV 或 Excel 文件且编码为 UTF-8/GBK")


def format_datetime_rule(dt_str):
    """日期函数：兼容英文月份缩写（11-Dec-24），00:00只保留日期，其余保留时分秒，不删行"""
    if pd.isna(dt_str):
        return None
    dt_str = str(dt_str).strip()
    fmt_list = [
        "%Y-%m-%d %H:%M:%S",
        "%Y/%m/%d %H:%M:%S",
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%Y年%m月%d日",
        "%d-%b-%y",
        "%d-%b-%Y"
    ]
    dt = None
    for fmt in fmt_list:
        try:
            dt = datetime.strptime(dt_str, fmt)
            break
        except Exception:
            continue
    if dt is None:
        return dt_str
    if dt.hour == 0 and dt.minute == 0 and dt.second == 0:
        return dt.strftime("%Y-%m-%d")
    else:
        return dt.strftime("%Y-%m-%d %H:%M:%S")


def clean_dataframe(filename, df):
    """
    对单个 DataFrame 按表名执行清洗，返回 (清洗后df, 清洗日志list)
    filename 必须能匹配五张标准表之一
    """
    steps = []

    # ========== 1. Dim_Branch ==========
    if filename == "Dim_Branch.csv":
        if "City" in df.columns:
            def clean_city(txt):
                if pd.isna(txt):
                    return txt
                txt = str(txt)
                for p in list(city_to_province.values()):
                    txt = txt.replace(p, "")
                txt = txt.replace("省", "")
                txt = re.sub(r"支行|分行|营业部|网点", "", txt)
                txt = txt.strip()
                return txt
            df["City"] = df["City"].apply(clean_city)
            steps.append("城市列清洗（去除省份前缀与网点后缀）")

        if "Province" in df.columns and "City" in df.columns:
            def fix_province(row):
                c = row["City"]
                if pd.notna(c) and c in city_to_province:
                    return city_to_province[c]
                return row["Province"]
            df["Province"] = df.apply(fix_province, axis=1)
            steps.append("根据城市修正省份")

        def add_prov_suffix(p):
            if pd.isna(p):
                return p
            p = str(p).strip()
            if p == "重庆":
                return p
            if not p.endswith("省"):
                return p + "省"
            return p
        df["Province"] = df["Province"].apply(add_prov_suffix)

        def add_city_suffix(c):
            if pd.isna(c):
                return c
            c = str(c).strip()
            if not c.endswith("市"):
                return c + "市"
            return c
        df["City"] = df["City"].apply(add_city_suffix)
        steps.append("省份补『省』、城市补『市』后缀（重庆除外）")

    # ========== 2. Dim_Customer ==========
    elif filename == "Dim_Customer.csv":
        if "RegisterDate" in df.columns:
            df["RegisterDate"] = df["RegisterDate"].apply(format_datetime_rule)
            steps.append("注册日期标准化（兼容英文月份缩写）")
        if "Tier" in df.columns:
            df["Tier"] = df["Tier"].astype(str).str.upper().str.strip()
            steps.append("Tier 统一大写并去空格")

    # ========== 3. Dim_FinProduct ==========
    elif filename == "Dim_FinProduct.csv":
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].round(4)
        steps.append("数值列统一保留4位小数")

    # ========== 4. Fact_Application（保留全部数据，一条不删）==========
    elif filename == "Fact_Application.csv":
        if "AppDate" in df.columns:
            df["AppDate"] = df["AppDate"].apply(format_datetime_rule)
            steps.append("申请日期标准化（兼容英文月份缩写）")
        if "Status" in df.columns:
            df = df.sort_values("Status", ascending=True)
            steps.append("按 Status 分组排序（保留全部数据，不删除金额为0的记录）")

    # ========== 5. Fact_Transaction ==========
    elif filename == "Fact_Transaction.csv":
        time_col = "TransDate"
        if time_col in df.columns:
            df[time_col] = df[time_col].apply(format_datetime_rule)
            steps.append("交易时间标准化")
        else:
            steps.append("未找到时间列，跳过时间处理")

    before = len(df)
    df = df.dropna(how="all")
    df = df.drop_duplicates()
    after = len(df)
    steps.append(f"去除全空行与重复行：{before} 行 → {after} 行")
    return df, steps


def match_table(upload_filename):
    """根据上传文件名匹配标准表名，返回标准表名或 None"""
    name = os.path.basename(upload_filename)
    for t in TABLE_NAMES:
        stem = t.replace(".csv", "")
        if stem in name:
            return t
    return None


def run_batch():
    """命令行批量清洗入口（原有逻辑）"""
    log_list = []
    for filename in TABLE_NAMES:
        fp = os.path.join(INPUT_FOLDER, filename)
        if not os.path.exists(fp):
            print(f"⚠ 文件不存在，跳过：{filename}")
            continue
        df = read_safe(fp)
        raw_rows = len(df)
        df, steps = clean_dataframe(filename, df)
        out_path = os.path.join(OUTPUT_FOLDER, filename)
        df.to_csv(out_path, index=False, encoding="utf-8-sig")
        log_list.append({"file": filename, "before": raw_rows, "after": len(df)})
        print(f"【{filename}】原始{raw_rows}行 → 清洗后{len(df)}行")
        for s in steps:
            print(f"  - {s}")

    log_df = pd.DataFrame(log_list)
    log_df.to_csv(os.path.join(OUTPUT_FOLDER, "clean_log.csv"), index=False, encoding="utf-8-sig")
    print("\n===== 全部清洗结束 =====")


if __name__ == "__main__":
    run_batch()
