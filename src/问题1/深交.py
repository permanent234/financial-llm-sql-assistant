# -*- coding: utf-8 -*-
"""
财报 PDF 解析脚本（DeepDeSRT思想增强版）
====================================================
核心思路：
1. 表格优先：先提取并分类表格，再做字段抽取
2. 结构优先：识别表头列（本期/上期/同比/环比）
3. 文本兜底：表格失败后再走section文本与全文窗口
4. 单位统一：自动识别元/千元/万元/亿元并统一转万元
5. 自动校验：资产负债平衡、跨表勾稽、数值合理性
"""

import re
import os
import sys
import json
import csv
import logging
from typing import Optional, Dict, List, Tuple
from pathlib import Path

import pdfplumber
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ====================== 基础工具 ======================

def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    s = s.replace("\n", "").replace("\r", "").replace(" ", "").replace("\u3000", "")
    s = s.replace("（", "(").replace("）", ")")
    s = s.replace("：", ":").replace("－", "-").replace("—", "-")
    s = s.strip()
    return s


def clean_number(s: str) -> Optional[float]:
    if s is None:
        return None
    s = str(s).strip()
    if not s:
        return None

    s = s.replace(",", "").replace("，", "")
    s = s.replace(" ", "").replace("\u00a0", "")
    s = s.replace("（", "(").replace("）", ")")

    if s in ("—", "-", "－", "--", ""):
        return None

    # 百分号保留给外层逻辑判断，这里只去掉用于转换
    s = s.replace("%", "")
    s = s.replace("元", "").replace("万元", "").replace("亿元", "").replace("千元", "")

    # 括号负数
    m = re.match(r"^\(([0-9.]+)\)$", s)
    if m:
        try:
            return -float(m.group(1))
        except:
            return None

    try:
        return float(s)
    except:
        return None


def extract_numbers_with_type(line: str):
    if not line:
        return []
    nums = re.findall(r"[（(]?-?[\d,]+\.?\d*%?[)）]?", str(line))
    result = []
    for n in nums:
        is_pct = "%" in n
        val = clean_number(n)
        if val is not None:
            result.append((val, is_pct))
    return result


def extract_first_numeric(line: str, prefer_percent=False) -> Optional[float]:
    nums = extract_numbers_with_type(line)
    if prefer_percent:
        for v, is_pct in nums:
            if is_pct:
                return v
    else:
        for v, is_pct in nums:
            if not is_pct:
                return v
    return None


def safe_div(a, b):
    if a is None or b in (None, 0):
        return None
    try:
        return a / b
    except:
        return None


# ====================== 报告期与元信息 ======================

def _infer_period(text: str, filename: str) -> Tuple[str, Optional[int]]:
    fn = os.path.basename(filename)
    head = text[:4000]
    #norm = normalize_text(head + fn)

    year = None
    m = re.search(r"(20\d{2})年", head)
    if m:
        year = int(m.group(1))
    else:
        m2 = re.search(r"(20\d{2})", fn)
        if m2:
            year = int(m2.group(1))

    if "第一季度" in fn or "一季度" in fn:
        return "Q1", year
    elif "半年度" in fn or "半年报" in fn:
        return "HY", year
    elif "第三季度" in fn or "三季度" in fn:
        return "Q3", year
    elif "年度报告" in fn or "年报" in fn:
        return "FY", year

    # 文件名日期推断
    date_m = re.search(r"_(\d{4})(\d{2})(\d{2})_", fn)
    if date_m:
        y, mth = int(date_m.group(1)), int(date_m.group(2))
        if mth <= 5:
            return "FY", y - 1

    return "", year


def extract_meta(text: str, filename: str) -> dict:
    m = re.search(r"(公司代码|股票代码|证券代码)[：:]\s*([0-9]{6})", text)
    stock_code = m.group(2) if m else re.sub(r"[^0-9]", "", os.path.basename(filename))[:6]

    m = re.search(r"(公司简称|股票简称|证券简称)[：:]\s*([^\n\r\s，。、]+)", text)
    stock_abbr = m.group(2).strip() if m else ""

    if not stock_abbr:
        fn = os.path.basename(filename)
        m2 = re.search(r"([A-Za-z\u4e00-\u9fa5]+)", fn)
        if m2:
            stock_abbr = m2.group(1)

    report_period, report_year = _infer_period(text, filename)

    return {
        "stock_code": stock_code,
        "stock_abbr": stock_abbr,
        "report_period": report_period,
        "report_year": report_year,
        "serial_number": 1
    }


# ====================== 区间定位 ======================

SECTION_MARKERS = {
    "balance_sheet": ["合并资产负债表"],    #, "资产负债表(合并)", "资产负债表（合并）", "资产负债表", "资产及负债状况"
    "income_sheet": ["合并利润表", "合并年初到报告期末利润表"],      #, "利润表(合并)", "利润表（合并）", "利润表"
    "cash_flow": ["合并现金流量表", "合并年初到报告期末现金流量表"],      #, "现金流量表(合并)", "现金流量表（合并）", "现金流量表","现金流"
    "core_perf": ["主要会计数据和财务指标", "主要财务指标", "(二)主要财务指标", "主要会计数据", "(一)主要会计数据", "核心业绩指标", "(一)主要会计数据和财务指标"]
}


def locate_sections(lines: List[str]) -> dict:
    hits = {k: None for k in SECTION_MARKERS}
    for i, line in enumerate(lines):
        t = normalize_text(line)
        for key, markers in SECTION_MARKERS.items():
            if hits[key] is None and any(normalize_text(m) in t for m in markers):
                hits[key] = i

    all_starts = sorted([v for v in hits.values() if v is not None])
    sections = {}
    for key, start in hits.items():
        if start is None:
            continue
        later = [s for s in all_starts if s > start]
        end = later[0] + 20 if later else start + 500
        end = min(end, len(lines))
        sections[key] = (start, end)
        log.info(f"[section] {key}: {start} ~ {end}")
    return sections


# ====================== 字段别名 ======================

ALIASES = {
    "core_performance_indicators_sheet": {
        "eps": ["基本每股收益", "每股收益"],
        "total_operating_revenue": ["营业总收入", "营业收入"],
        "operating_revenue_yoy_growth": ["营业总收入同比增长", "营业总收入同比", "营业收入同比增长", "营业收入同比", "本期比上年同期增减"],
        "operating_revenue_qoq_growth": ["营业总收入季度环比增长", "营业收入季度环比增长", "环比增长"],
        "net_profit_10k_yuan": ["净利润", "归属于上市公司股东的净利润", "归属于上市公司股东的净利润（元）", "归属于母公司所有者的净利润", "归母净利润"],
        "net_profit_yoy_growth": ["净利润同比增长", "净利润同比"],   #, "归母净利润同比增长", "归母净利润同比"
        "net_profit_qoq_growth": ["净利润季度环比增长", "净利润环比增长"],
        "net_asset_per_share1": ["归属于上市公司股东的净资产"],
        "net_asset_per_share2": ["总股本"],
        "roe": ["加权平均净资产收益率", "净资产收益率"],
        "operating_cf_per_share": ["每股经营现金流量", "每股经营活动产生的现金流量净额"],
        "net_profit_excl_non_recurring": ["归属于上市公司股东的扣除非经常性损益的净利润（元）", "归属于上市公司股东的扣除非经常性损益的净利润", "扣除非经常性损益后的净利润", "扣非净利润"],
        "net_profit_excl_non_recurring_yoy": ["扣非净利润同比增长", "扣非净利润同比"],
        "gross_profit_margin": ["销售毛利率", "毛利率"],
        "net_profit_margin": ["销售净利率", "净利率"],
        "roe_weighted_excl_non_recurring": ["加权平均净资产收益率（扣非）", "扣除非经常性损益后的加权平均净资产收益率"]
    },
    "balance_sheet": {
        "asset_cash_and_cash_equivalents": ["货币资金", "现金及现金等价物"],
        "asset_accounts_receivable": ["应收账款", "应收票据及应收账款"],
        "asset_inventory": ["存货"],
        "asset_trading_financial_assets": ["交易性金融资产"],
        "asset_construction_in_progress": ["在建工程"],
        "asset_total_assets": ["资产总计"],
        "asset_total_assets_yoy_growth": ["总资产同比", "资产总计同比", "本期期末金额较上期期末变动比例"],
        "liability_accounts_payable": ["应付账款"],
        "liability_advance_from_customers": ["预收账款", "预收款项"],
        "liability_total_liabilities": ["负债合计"],
        "liability_total_liabilities_yoy_growth": ["总负债同比", "负债合计同比", "本期期末金额较上期期末变动比例"],
        "liability_contract_liabilities": ["合同负债"],
        "liability_short_term_loans": ["短期借款"],
        "asset_liability_ratio": ["资产负债率"],
        "equity_unappropriated_profit": ["未分配利润"],
        "equity_total_equity": ["所有者权益合计", "所有者权益（或股东权益）合计", "股东权益合计", "归属于上市公司股东的所有者权益"],
        "guben": ["股本"]
    },
    "income_sheet": {
        "net_profit": ["净利润", "归属于上市公司股东的净利润", "持续经营净利润"],
        "net_profit_yoy_growth": ["净利润同比增长", "净利润同比"],  #, "归母净利润同比增长", "归母净利润同比"
        "other_income": ["其他收益"],
        "total_operating_revenue": ["营业总收入", "营业收入", "一、营业总收入"],
        "operating_revenue_yoy_growth": ["营业总收入同比增长", "营业总收入同比", "营业收入同比增长", "营业收入同比"],
        "operating_expense_cost_of_sales": ["营业成本", "营业支出"],
        "operating_expense_selling_expenses": ["销售费用"],
        "operating_expense_administrative_expenses": ["管理费用"],
        "operating_expense_financial_expenses": ["财务费用"],
        "operating_expense_rnd_expenses": ["研发费用"],
        "operating_expense_taxes_and_surcharges": ["税金及附加"],
        "total_operating_expenses": ["营业总成本", "营业总支出"],
        "operating_profit": ["营业利润", "三、营业利润"],
        "total_profit": ["利润总额", "四、利润总额"],
        "asset_impairment_loss": ["资产减值损失"],
        "credit_impairment_loss": ["信用减值损失"]
    },
    "cash_flow_sheet": {
        "net_cash_flow": ["现金及现金等价物净增加额", "净现金流"],
        "net_cash_flow_yoy_growth": ["净现金流同比增长", "净现金流同比"],
        "operating_cf_net_amount": ["经营活动产生的现金流量净额", "经营性现金流净额"],
        "operating_cf_ratio_of_net_cf": ["经营性现金流占比", "经营性现金流净现金流占比"],
        "operating_cf_cash_from_sales": ["销售商品、提供劳务收到的现金", "销售商品收到的现金"],
        "investing_cf_net_amount": ["投资活动产生的现金流量净额", "投资性现金流净额"],
        "investing_cf_ratio_of_net_cf": ["投资性现金流占比"],
        "investing_cf_cash_for_investments": ["投资支付的现金"],
        "investing_cf_cash_from_investment_recovery": ["收回投资收到的现金"],
        "financing_cf_cash_from_borrowing": ["取得借款收到的现金"],
        "financing_cf_cash_for_debt_repayment": ["偿还债务支付的现金"],
        "financing_cf_net_amount": ["筹资活动产生的现金流量净额", "融资活动产生的现金流量净额", "融资性现金流净额"],
        "financing_cf_ratio_of_net_cf": ["融资性现金流占比"]
    }
}


# ====================== 单位识别 ======================

def detect_unit_multiplier(text: str) -> float:
    """
    返回原值对应的“元”倍数：
    元 -> 1
    千元 -> 1000
    万元 -> 10000
    亿元 -> 100000000
    """
    t = normalize_text(text[:2000] if text else "")
    if "单位:亿元" in t or "单位：亿元" in t:
        return 100000000
    if "单位:万元" in t or "单位：万元" in t:
        return 10000
    if "单位:千元" in t or "单位：千元" in t:
        return 1000
    return 1


def to_wan(v: Optional[float], unit_multiplier: float) -> Optional[float]:
    if v is None:
        return None
    try:
        return round(v * unit_multiplier / 10000, 2)
    except:
        return None


def to_yuan(v: Optional[float], unit_multiplier: float) -> Optional[float]:
    if v is None:
        return None
    try:
        return round(v * unit_multiplier, 2)
    except:
        return None


# ====================== 页面/表格筛选 ======================

def is_financial_statement_page(text: str) -> bool:
    t = normalize_text(text)
    keywords = [
        "合并资产负债表", "资产负债表",
        "合并利润表", "利润表",
        "合并现金流量表", "现金流量表",
        "主要财务指标", "主要会计数据",
        "货币资金", "营业收入", "净利润", "经营活动产生的现金流量净额"
    ]
    hit_count = sum(1 for k in keywords if k in t)
    return hit_count >= 2


def clean_table_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    df = df.copy()
    df = df.dropna(axis=1, how="all")
    df = df.dropna(axis=0, how="all")
    df = df.fillna("")
    keep_cols = []
    for col in df.columns:
        vals = [normalize_text(x) for x in df[col].tolist()]
        if any(v != "" for v in vals):
            keep_cols.append(col)
    if keep_cols:
        df = df[keep_cols]
    return df.reset_index(drop=True)


def merge_broken_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    merged_rows = []
    i = 0
    while i < len(df):
        current = df.iloc[i].tolist()
        current_text = "".join([normalize_text(x) for x in current[:3]])

        if i + 1 < len(df):
            next_row = df.iloc[i + 1].tolist()
            next_text = "".join([normalize_text(x) for x in next_row[:3]])

            current_has_num = extract_first_numeric(" ".join(map(str, current))) is not None
            next_has_num = extract_first_numeric(" ".join(map(str, next_row))) is not None

            # 当前行像半截标签，下一行继续
            if current_text and len(current_text) <= 18 and not current_has_num and next_text:
                merged = []
                max_len = max(len(current), len(next_row))
                for idx in range(max_len):
                    a = current[idx] if idx < len(current) else ""
                    b = next_row[idx] if idx < len(next_row) else ""
                    merged.append(f"{a}{b}" if (str(a) or str(b)) else "")
                merged_rows.append(merged)
                i += 2
                continue

        merged_rows.append(current)
        i += 1

    return pd.DataFrame(merged_rows).fillna("")


def classify_table(df: pd.DataFrame) -> Optional[str]:
    if df is None or df.empty:
        return None
    text = "".join([normalize_text(x) for x in df.fillna("").astype(str).values.flatten().tolist()])

    if any(k == text for k in ["合并资产负债表", "资产总计", "负债合计", "所有者权益合计", "货币资金", "期末余额"]):
        return "balance_sheet"
    if any(k == text for k in ["营业收入", "营业总收入", "营业利润", "利润总额", "净利润"]):
        return "income_sheet"
    if any(k == text for k in ["经营活动产生的现金流量净额", "投资活动产生的现金流量净额", "筹资活动产生的现金流量净额", "现金及现金等价物净增加额"]):
        return "cash_flow_sheet"
    if any(k == text for k in ["加权平均净资产收益率", "基本每股收益", "营业收入", "EPS", "净资产收益率", "ROE", "净利润"]):
        return "core_performance_indicators_sheet"
    return None


def extract_all_tables(pdf_path: str):
    all_tables = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_idx, page in enumerate(pdf.pages):
            page_text = page.extract_text(x_tolerance=3, y_tolerance=3) or ""
            if not is_financial_statement_page(page_text):
                continue
            try:
                unit_multiplier = detect_unit_multiplier(page_text)
                tables = page.extract_tables()
                for tb in tables:
                    if tb:
                        df = clean_table_df(pd.DataFrame(tb))
                        df = merge_broken_rows(df)
                        if df is not None and not df.empty:
                            all_tables.append({
                                "page": page_idx + 1,
                                "df": df,
                                "unit_multiplier": unit_multiplier,
                                "page_text": page_text
                            })
            except Exception:
                pass
    return all_tables


def group_tables_by_type(tables):
    grouped = {
        "balance_sheet": [],
        "income_sheet": [],
        "cash_flow_sheet": [],
        "core_performance_indicators_sheet": []
    }
    for tb in tables:
        table_type = classify_table(tb["df"])
        if table_type:
            grouped[table_type].append(tb)
    return grouped


# ====================== 表结构识别 ======================
def detect_header_info(df: pd.DataFrame, table_type=None):
    header_info = {
        "header_row_idx": None,
        "current": None,
        "current_date": None,  #记录识别到的具体日期
        "previous": None,
        "yoy": None,
        "qoq": None
    }

    #if table_type == "core_performance_indicators_sheet":
    #    # 行名在第0列 → 数据从第1列开始
    #    if len(df.columns) >= 2:
    #        header_info["header_row_idx"] = 0
    #        header_info["current"] = df.columns[1]  # ⭐第二列
    #    return header_info

    for i in range(min(10, len(df))):
        row = [normalize_text(x) for x in df.iloc[i].tolist()]
        hit = False
        for j, cell in enumerate(row):
            if any(k == cell for k in ["本报告期", "本期", "期末", "期末余额", "本年发生额", "本期发生额", "current", "current_date"]):
                header_info["current"] = j
                header_info["current_date"] = cell
                hit = True
            elif any(k == cell for k in ["上年同期", "上期", "年初余额", "期初", "上期发生额", "上年末"]):
                header_info["previous"] = j
                hit = True
            elif any(k == cell for k in ["同比", "同比增减", "比上年同期增减", "增减", "变动比例", "本期比上年同期增减", "本报告期比上年同期增减变动幅度", "本报告期比上年同期增减", "本期期末金额较上期期末变动比例"]):
                header_info["yoy"] = j
                hit = True
            elif any(k == cell for k in ["环比", "季度环比", "比上季度增减", "本报告期末比上年度末增减变动幅度"]):
                header_info["qoq"] = j
                hit = True
        if hit:
            header_info["header_row_idx"] = i
            break

    # 自动适配：根据列数+日期特征动态定位本期列
    #if header_info["current"] is None:
    #    # 优先从表头行识别日期列
    #    if header_info["header_row_idx"] is not None:
    #        header_row = [normalize_text(x) for x in df.iloc[header_info["header_row_idx"]].tolist()]
    #        # 遍历表头行找最新日期列（本期）
    #        date_cols = []
    #        for j, cell in enumerate(header_row[:]):
    #            # 匹配YYYY年MM月DD日格式
    #            if re.search(r"20\d{2}年\d{1,2}月\d{1,2}日", cell):
    #                date_cols.append((j, cell))
    #        if date_cols:
    #            # 按日期排序，取最新的为本期
    #            date_cols.sort(
    #                key=lambda x: pd.to_datetime(x[1].replace("年", "-").replace("月", "-").replace("日", "")))
    #            header_info["current"] = date_cols[-1][0]
    #            header_info["current_date"] = date_cols[-1][1]
    #            # 上期为前一个日期列
    #            if len(date_cols) >= 2:
    #                header_info["previous"] = date_cols[-2][0]
    #            return header_info
        # 兜底：按列数+报告期适配
    #    if len(df.columns) >= 2:
    #        header_info["current"] = len(df.columns) - 2
    #        header_info["previous"] = len(df.columns) - 1
        # ===== 强制兜底：如果没识别到yoy，但有4列以上 =====
        #if header_info["yoy"] is None and len(df.columns) >= 4:
        #    header_info["yoy"] = len(df.columns) - 2
    return header_info


def choose_target_col_by_field(field_name: str, header_info: dict):
    if "yoy" in field_name:
        return header_info.get("yoy")
    if "qoq" in field_name:
        return header_info.get("qoq")
    return header_info.get("current")


def get_row_label(row_values, max_label_cols=3):
    parts = []
    for x in row_values[:max_label_cols]:
        txt = normalize_text(x)
        if txt:
            parts.append(txt)
    return "".join(parts)


def extract_field_from_tables(table_items, aliases_map, field_name):
    keywords = aliases_map.get(field_name, [])
    if not keywords:
        return None, None, None

    for item in table_items:
        df = item["df"]
        unit_multiplier = item.get("unit_multiplier", 1)

        header_info = detect_header_info(df)
        start_row = header_info["header_row_idx"] + 1 if header_info["header_row_idx"] is not None else 0
        target_col = choose_target_col_by_field(field_name, header_info)

        for i in range(start_row, len(df)):
            raw_vals = df.iloc[i].tolist()
            row_vals = [normalize_text(x) for x in raw_vals]
            row_text = "".join(row_vals)

            if any(normalize_text(k) in row_text for k in keywords):
                # 优先按目标列取
                if target_col is not None:
                    if target_col >= len(df.columns):
                        target_col = None

                # 优先取目标列 + 全列兜底
                vals_to_try = []

                if target_col is not None and target_col < len(row_vals):
                    vals_to_try.append(row_vals[target_col])

                # 加入整行所有列（防止错列）
                vals_to_try.extend(row_vals)

                for v in vals_to_try:
                    val = clean_number(v)
                    if val is not None:
                        return val, unit_multiplier, f"table:p{item['page']}:r{i}"
                # 仅当目标列取数失败时，才从整行取数
                line = " ".join([str(x) if x is not None else "" for x in raw_vals])
                if "yoy" in field_name or "qoq" in field_name or "ratio" in field_name or "margin" in field_name or field_name in ["roe", "roe_weighted_excl_non_recurring"]:
                    val = extract_first_numeric(line, prefer_percent=True)
                else:
                    val = extract_first_numeric(line, prefer_percent=False)

                if val is not None:
                    return val, unit_multiplier, f"table-row:p{item['page']}:r{i}"

    return None, None, None


# ====================== 文本兜底 ======================

def find_value_in_lines(lines: List[str], patterns: List[str], start: int, end: int, target_type="current") -> Optional[float]:
    for i in range(start, min(end, len(lines))):
        line = lines[i]
        t = normalize_text(line)
        if any(normalize_text(p) in t for p in patterns):
            if target_type == "yoy":
                val = extract_first_numeric(line, prefer_percent=True)
            else:
                val = extract_first_numeric(line, prefer_percent=False)
            if val is not None:
                return val

            if i + 1 < end:
                nxt = lines[i + 1]
                if target_type == "yoy":
                    val = extract_first_numeric(nxt, prefer_percent=True)
                else:
                    val = extract_first_numeric(nxt, prefer_percent=False)
                if val is not None:
                    return val
    return None


def find_value_near_keyword(text: str, keywords: List[str], target_type="current", window=140) -> Optional[float]:
    text2 = text.replace("\n", "")
    norm_all = normalize_text(text2)

    for kw in keywords:
        kw2 = normalize_text(kw)
        idx = norm_all.find(kw2)
        if idx != -1:
            seg = text2[max(0, idx-window): idx+window]
            nums = extract_numbers_with_type(seg)
            if target_type == "yoy":
                for v, is_pct in nums:
                    if is_pct:
                        return v
            else:
                for v, is_pct in nums:
                    if not is_pct:
                        return v
    return None


def get_section_unit_multiplier(lines: List[str], section_range: Tuple[int, int]) -> float:
    start, end = section_range
    sample = "\n".join(lines[start:min(end, start+20)])
    return detect_unit_multiplier(sample)


def get_field_value(table_items, lines, full_text, section_range, table_aliases, field_name, target_type="current"):
    aliases = table_aliases.get(field_name, [])
    if not aliases:
        return None, None

    # 1. 表格优先
    val, unit_multiplier, source = extract_field_from_tables(table_items, table_aliases, field_name)
    if val is not None:
        return val, {"source": source, "unit_multiplier": unit_multiplier or 1}

    # 2. section文本兜底
    start, end = section_range
    val = find_value_in_lines(lines, aliases, start, end, target_type=target_type)
    if val is not None:
        unit_multiplier = get_section_unit_multiplier(lines, section_range)
        return val, {"source": f"section_text:{start}-{end}", "unit_multiplier": unit_multiplier}

    # 3. 全文附近窗口兜底
    val = find_value_near_keyword(full_text, aliases, target_type=target_type, window=160)
    if val is not None:
        unit_multiplier = detect_unit_multiplier(full_text[:3000])
        return val, {"source": "full_text_window", "unit_multiplier": unit_multiplier}

    return None, None


# ====================== 四张表解析 ======================

def parse_balance_sheet(meta, table_items, lines, full_text, sec_range):
    aliases = ALIASES["balance_sheet"]
    debug = {}

    total_assets_raw, info = get_field_value(table_items, lines, full_text, sec_range, aliases, "asset_total_assets")
    debug["asset_total_assets"] = info
    total_liab_raw, info = get_field_value(table_items, lines, full_text, sec_range, aliases, "liability_total_liabilities")
    debug["liability_total_liabilities"] = info
    equity_total_raw, info = get_field_value(table_items, lines, full_text, sec_range, aliases, "equity_total_equity")
    debug["equity_total_equity"] = info

    unit_assets = debug["asset_total_assets"]["unit_multiplier"] if debug["asset_total_assets"] else 1
    unit_liab = debug["liability_total_liabilities"]["unit_multiplier"] if debug["liability_total_liabilities"] else 1
    unit_equity = debug["equity_total_equity"]["unit_multiplier"] if debug["equity_total_equity"] else 1

    total_assets = to_wan(total_assets_raw, unit_assets)
    total_liab = to_wan(total_liab_raw, unit_liab)
    equity_total = to_wan(equity_total_raw, unit_equity)

    def amount_field(field):
        raw, info = get_field_value(table_items, lines, full_text, sec_range, aliases, field)
        debug[field] = info
        unit = info["unit_multiplier"] if info else 1
        return to_wan(raw, unit)

    def amount_field1(field):
        raw, info = get_field_value(table_items, lines, full_text, sec_range, aliases, field)
        debug[field] = info
        unit = info["unit_multiplier"] if info else 1
        return raw

    def ratio_field(field):
        raw, info = get_field_value(table_items, lines, full_text, sec_range, aliases, field, target_type="yoy")
        debug[field] = info
        return raw

    asset_liability_ratio = None
    if total_assets not in (None, 0) and total_liab is not None:
        asset_liability_ratio = round(total_liab / total_assets * 100, 4)
    else:
        asset_liability_ratio = ratio_field("asset_liability_ratio")

    data = {
        "serial_number": meta["serial_number"],
        "stock_code": meta["stock_code"],
        "stock_abbr": meta["stock_abbr"],
        "asset_cash_and_cash_equivalents": amount_field("asset_cash_and_cash_equivalents"),
        "asset_accounts_receivable": amount_field("asset_accounts_receivable"),
        "asset_inventory": amount_field("asset_inventory"),
        "asset_trading_financial_assets": amount_field("asset_trading_financial_assets"),
        "asset_construction_in_progress": amount_field("asset_construction_in_progress"),
        "asset_total_assets": total_assets,
        "asset_total_assets_yoy_growth": ratio_field("asset_total_assets_yoy_growth"),
        "liability_accounts_payable": amount_field("liability_accounts_payable"),
        "liability_advance_from_customers": amount_field("liability_advance_from_customers"),
        "liability_total_liabilities": total_liab,
        "liability_total_liabilities_yoy_growth": ratio_field("liability_total_liabilities_yoy_growth"),
        "liability_contract_liabilities": amount_field("liability_contract_liabilities"),
        "liability_short_term_loans": amount_field("liability_short_term_loans"),
        "asset_liability_ratio": asset_liability_ratio,
        "equity_unappropriated_profit": amount_field("equity_unappropriated_profit"),
        "equity_total_equity": equity_total,
        "report_period": meta["report_period"],
        "report_year": meta["report_year"],
        "guben": amount_field1("guben")
    }
    return data, debug


def parse_income_sheet(meta, table_items, lines, full_text, sec_range):
    aliases = ALIASES["income_sheet"]
    debug = {}

    def amount_field(field, target_period="current"):
        raw, info = get_field_value(table_items, lines, full_text, sec_range, aliases, field)
        debug[field] = info
        unit = info["unit_multiplier"] if info else 1
        return to_wan(raw, unit)

    def ratio_field(field):
        raw, info = get_field_value(table_items, lines, full_text, sec_range, aliases, field, target_type="yoy")
        debug[field] = info
        return raw

    revenue = amount_field("total_operating_revenue")
    cost = amount_field("operating_expense_cost_of_sales")
    selling = amount_field("operating_expense_selling_expenses")
    admin = amount_field("operating_expense_administrative_expenses")
    fin = amount_field("operating_expense_financial_expenses")
    rnd = amount_field("operating_expense_rnd_expenses")
    tax = amount_field("operating_expense_taxes_and_surcharges")
    total_exp = amount_field("total_operating_expenses")

    if total_exp is None:
        vals = [cost, selling, admin, fin, rnd, tax]
        if any(v is not None for v in vals):
            total_exp = round(sum(v or 0 for v in vals), 2)

    data = {
        "serial_number": meta["serial_number"],
        "stock_code": meta["stock_code"],
        "stock_abbr": meta["stock_abbr"],
        "net_profit": amount_field("net_profit"),
        "net_profit_yoy_growth": ratio_field("net_profit_yoy_growth"),
        "other_income": amount_field("other_income"),
        "total_operating_revenue": revenue,
        "operating_revenue_yoy_growth": ratio_field("operating_revenue_yoy_growth"),
        "operating_expense_cost_of_sales": cost,
        "operating_expense_selling_expenses": selling,
        "operating_expense_administrative_expenses": admin,
        "operating_expense_financial_expenses": fin,
        "operating_expense_rnd_expenses": rnd,
        "operating_expense_taxes_and_surcharges": tax,
        "total_operating_expenses": total_exp,
        "operating_profit": amount_field("operating_profit"),
        "total_profit": amount_field("total_profit"),
        "asset_impairment_loss": amount_field("asset_impairment_loss"),
        "credit_impairment_loss": amount_field("credit_impairment_loss"),
        "report_period": meta["report_period"],
        "report_year": meta["report_year"]
    }
    return data, debug


def parse_cash_flow_sheet(meta, table_items, lines, full_text, sec_range):
    aliases = ALIASES["cash_flow_sheet"]
    debug = {}

    def amount_field_wan(field):
        raw, info = get_field_value(table_items, lines, full_text, sec_range, aliases, field)
        debug[field] = info
        unit = info["unit_multiplier"] if info else 1
        return to_wan(raw, unit)

    def amount_field_yuan(field):
        raw, info = get_field_value(table_items, lines, full_text, sec_range, aliases, field)
        debug[field] = info
        unit = info["unit_multiplier"] if info else 1
        return to_yuan(raw, unit)

    def ratio_field(field):
        raw, info = get_field_value(table_items, lines, full_text, sec_range, aliases, field, target_type="yoy")
        debug[field] = info
        return raw

    net_cf = amount_field_yuan("net_cash_flow")
    op_cf = amount_field_wan("operating_cf_net_amount")
    inv_cf = amount_field_wan("investing_cf_net_amount")
    fin_cf = amount_field_wan("financing_cf_net_amount")

    data = {
        "serial_number": meta["serial_number"],
        "stock_code": meta["stock_code"],
        "stock_abbr": meta["stock_abbr"],
        "net_cash_flow": net_cf,
        "net_cash_flow_yoy_growth": ratio_field("net_cash_flow_yoy_growth"),
        "operating_cf_net_amount": op_cf,
        "operating_cf_ratio_of_net_cf": round((op_cf * 10000) / net_cf * 100, 4) if net_cf not in (None, 0) and op_cf is not None else None,
        "operating_cf_cash_from_sales": amount_field_wan("operating_cf_cash_from_sales"),
        "investing_cf_net_amount": inv_cf,
        "investing_cf_ratio_of_net_cf": round((inv_cf * 10000) / net_cf * 100, 4) if net_cf not in (None, 0) and inv_cf is not None else None,
        "investing_cf_cash_for_investments": amount_field_wan("investing_cf_cash_for_investments"),
        "investing_cf_cash_from_investment_recovery": amount_field_wan("investing_cf_cash_from_investment_recovery"),
        "financing_cf_cash_from_borrowing": amount_field_wan("financing_cf_cash_from_borrowing"),
        "financing_cf_cash_for_debt_repayment": amount_field_wan("financing_cf_cash_for_debt_repayment"),
        "financing_cf_net_amount": fin_cf,
        "financing_cf_ratio_of_net_cf": round((fin_cf * 10000) / net_cf * 100, 4) if net_cf not in (None, 0) and fin_cf is not None else None,
        "report_period": meta["report_period"],
        "report_year": meta["report_year"]
    }
    return data, debug


def parse_core_performance(meta, table_items, lines, full_text, sec_range, income_data, balance_data, cash_data):
    aliases = ALIASES["core_performance_indicators_sheet"]
    debug = {}

    def amount_field_wan(field):
        raw, info = get_field_value(table_items, lines, full_text, sec_range, aliases, field)
        debug[field] = info
        unit = info["unit_multiplier"] if info else 1
        return to_wan(raw, unit)

    def amount_field1(field):
        raw, info = get_field_value(table_items, lines, full_text, sec_range, aliases, field)
        debug[field] = info
        unit = info["unit_multiplier"] if info else 1
        return raw

    def normal_field(field, target_type="current"):
        raw, info = get_field_value(table_items, lines, full_text, sec_range, aliases, field)
        debug[field] = info
        return raw

    revenue = amount_field_wan("total_operating_revenue")
    if revenue is None:
        revenue = income_data.get("total_operating_revenue")

    net_profit = amount_field_wan("net_profit_10k_yuan")
    if net_profit is None:
        net_profit = income_data.get("net_profit")

    gross_profit_margin = normal_field("gross_profit_margin", "yoy")
    if gross_profit_margin is None:
        rev = income_data.get("total_operating_revenue")
        cost = income_data.get("operating_expense_cost_of_sales")
        if rev not in (None, 0) and cost is not None:
            gross_profit_margin = round((rev - cost) / rev * 100, 4)

    net_profit_margin = normal_field("net_profit_margin", "yoy")
    if net_profit_margin is None:
        rev = income_data.get("total_operating_revenue")
        np = income_data.get("net_profit")
        if rev not in (None, 0) and np is not None:
            net_profit_margin = round(np / rev * 100, 4)

    data = {
        "serial_number": meta["serial_number"],
        "stock_code": meta["stock_code"],
        "stock_abbr": meta["stock_abbr"],
        "eps": normal_field("eps"),
        "total_operating_revenue": revenue,
        "operating_revenue_yoy_growth": normal_field("operating_revenue_yoy_growth", "yoy"),
        "operating_revenue_qoq_growth": normal_field("operating_revenue_qoq_growth", "yoy"),
        "net_profit_10k_yuan": net_profit,
        "net_profit_yoy_growth": normal_field("net_profit_yoy_growth", "yoy"),
        "net_profit_qoq_growth": normal_field("net_profit_qoq_growth", "yoy"),
        "net_asset_per_share": normal_field("net_asset_per_share"),
        "roe": amount_field1("roe"),
        "operating_cf_per_share": normal_field("operating_cf_per_share"),
        "net_profit_excl_non_recurring": amount_field_wan("net_profit_excl_non_recurring"),
        "net_profit_excl_non_recurring_yoy": normal_field("net_profit_excl_non_recurring_yoy", "yoy"),
        "gross_profit_margin": gross_profit_margin,
        "net_profit_margin": net_profit_margin,
        "roe_weighted_excl_non_recurring": normal_field("roe_weighted_excl_non_recurring", "yoy"),
        "report_period": meta["report_period"],
        "report_year": meta["report_year"]
    }
    return data, debug


# ====================== 自动校验 ======================

def validate_consistency(result: dict) -> Tuple[bool, List[str]]:
    errors = []
    core = result.get("core_performance_indicators_sheet", {})
    bs = result.get("balance_sheet", {})
    inc = result.get("income_sheet", {})
    cf = result.get("cash_flow_sheet", {})

    total_a = bs.get("asset_total_assets")
    total_l = bs.get("liability_total_liabilities")
    total_e = bs.get("equity_total_equity")
    if total_a is not None and total_l is not None and total_e is not None:
        computed = total_l + total_e
        diff = abs(total_a - computed)
        if diff > 10:
            errors.append(f"资产负债表不平衡: 总资产={total_a}万 ≠ 负债+权益={computed}万 (差值={diff})")

    inc_np = inc.get("net_profit")
    core_np = core.get("net_profit_10k_yuan")
    if inc_np is not None and core_np is not None and abs(inc_np - core_np) > 0.01:
        errors.append(f"净利润勾稽失败: 利润表={inc_np}万 ≠ 核心指标={core_np}万")

    inc_rev = inc.get("total_operating_revenue")
    core_rev = core.get("total_operating_revenue")
    if inc_rev is not None and core_rev is not None and abs(inc_rev - core_rev) > 0.01:
        errors.append(f"营业收入勾稽失败: 利润表={inc_rev}万 ≠ 核心指标={core_rev}万")

    op_cf = cf.get("operating_cf_net_amount")
    inv_cf = cf.get("investing_cf_net_amount")
    fin_cf = cf.get("financing_cf_net_amount")
    net_cf = cf.get("net_cash_flow")
    if all(x is not None for x in [op_cf, inv_cf, fin_cf, net_cf]):
        sum_three = op_cf * 10000 + inv_cf * 10000 + fin_cf * 10000
        if abs(sum_three - net_cf) > 1000000:
            errors.append(f"现金流量表勾稽失败: 三大活动之和 ≠ 净增加额 (差值≈{abs(sum_three - net_cf)/10000:.2f}万)")

    eps = core.get("eps")
    np_10k = core.get("net_profit_10k_yuan")
    if eps is not None and np_10k is not None:
        if np_10k > 0 and eps <= 0:
            errors.append("EPS 与净利润正负矛盾（盈利却EPS≤0）")
        if abs(eps) > 100:
            errors.append(f"EPS 量级异常: {eps}")

    roe = core.get("roe")
    if roe is not None and (roe < -100 or roe > 100):
        errors.append(f"ROE 量级异常: {roe}")

    ratio = bs.get("asset_liability_ratio")
    if ratio is not None and (ratio < 0 or ratio > 100):
        errors.append(f"资产负债率超出合理范围: {ratio}%")

    return len(errors) == 0, errors


# ====================== 缺失报告 & 调试输出 ======================

def report_missing_fields(result: dict):
    print("\n缺失字段报告：")
    for table_name, data in result.items():
        if table_name == "_debug_sources":
            continue
        missing = [k for k, v in data.items() if v is None]
        print(f"\n[{table_name}] 缺失 {len(missing)} 个字段")
        for k in missing:
            print("  -", k)


def print_summary(result: dict):
    meta = result["core_performance_indicators_sheet"]
    print("\n" + "═" * 70)
    print(f"  解析完成  {meta['stock_abbr']}（{meta['stock_code']}）  {meta['report_year']} {meta['report_period']}")
    print("═" * 70)
    for name, data in result.items():
        if name == "_debug_sources":
            continue
        print(f"\n【{name}】共 {len(data)} 个字段")
        for k, v in list(data.items())[:10]:
            print(f"  {k:<45} {v}")


# ====================== 主解析 ======================

def parse_pdf(pdf_path: str) -> dict:
    log.info(f"正在读取: {os.path.basename(pdf_path)}")

    # 合并：只打开一次PDF + 摘要判断
    try:
        with pdfplumber.open(pdf_path) as pdf:
            # 先提取前2页判断是否摘要
            check_text = ""
            for page in pdf.pages[:2]:
                t = page.extract_text(x_tolerance=3, y_tolerance=3) or ""
                check_text += t

            # 摘要过滤
            skip_keywords = ["摘要", "内容摘要", "报告摘要", "摘要版", "简要报告", "简要"]
            for kw in skip_keywords:
                if kw in check_text:
                    log.info(f"✅ 检测为【年度报告摘要PDF】，自动跳过：{os.path.basename(pdf_path)}")
                    return {}  # 空 → 不保存

            # 再提取全部文本（只提取一次，不重复打开）
            pages_text = [page.extract_text(x_tolerance=3, y_tolerance=3) or "" for page in pdf.pages]
    except:
        return {}

    full_text = "\n".join(pages_text)
    lines = full_text.split("\n")

    meta = extract_meta(full_text, pdf_path)
    log.info(f"代码={meta['stock_code']} 简称={meta['stock_abbr']} 报告期={meta['report_period']} 年份={meta['report_year']}")

    sections = locate_sections(lines)
    all_tables = extract_all_tables(pdf_path)
    grouped_tables = group_tables_by_type(all_tables)

    log.info(f"共提取表格 {len(all_tables)} 个")
    for k, v in grouped_tables.items():
        log.info(f"  表格分类 {k}: {len(v)} 个")

    bs, bs_debug = parse_balance_sheet(meta, grouped_tables["balance_sheet"], lines, full_text, sections.get("balance_sheet", (0, 0)))
    inc, inc_debug = parse_income_sheet(meta, grouped_tables["income_sheet"], lines, full_text, sections.get("income_sheet", (0, 0)))
    cf, cf_debug = parse_cash_flow_sheet(meta, grouped_tables["cash_flow_sheet"], lines, full_text, sections.get("cash_flow", (0, 0)))
    core, core_debug = parse_core_performance(meta, grouped_tables["core_performance_indicators_sheet"], lines, full_text, sections.get("core_perf", (0, 0)), inc, bs, cf)

    result = {
        "core_performance_indicators_sheet": core,
        "balance_sheet": bs,
        "income_sheet": inc,
        "cash_flow_sheet": cf,
        "_debug_sources": {
            "core_performance_indicators_sheet": core_debug,
            "balance_sheet": bs_debug,
            "income_sheet": inc_debug,
            "cash_flow_sheet": cf_debug,
        }
    }
    return result


# ====================== 输出 ======================

def save_json(result: dict, output_dir: str):
    meta = result["core_performance_indicators_sheet"]
    name = f"{meta['stock_code']}_{meta['report_year']}_{meta['report_period']}.json"
    path = os.path.join(output_dir, name)
    os.makedirs(output_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    log.info(f"JSON 已保存: {path}")


def save_global_csvs(all_results: List[dict], output_dir: str):
    if not all_results:
        return
    os.makedirs(output_dir, exist_ok=True)

    table_names = [
        "core_performance_indicators_sheet",
        "balance_sheet",
        "income_sheet",
        "cash_flow_sheet"
    ]

    for table_name in table_names:
        rows = []
        for idx, res in enumerate(all_results, start=1):
            data = res[table_name].copy()
            data["serial_number"] = idx
            rows.append(data)

        if not rows:
            continue

        path = os.path.join(output_dir, f"{table_name}.csv")
        fieldnames = list(rows[0].keys())
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        log.info(f"全局 CSV 已保存: {path} （共 {len(rows)} 条记录）")


def save_debug_sources(all_results: List[dict], output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "debug_sources.json")
    debug_data = []
    for idx, res in enumerate(all_results, start=1):
        meta = res["core_performance_indicators_sheet"]
        debug_data.append({
            "serial_number": idx,
            "stock_code": meta["stock_code"],
            "report_year": meta["report_year"],
            "report_period": meta["report_period"],
            "debug_sources": res.get("_debug_sources", {})
        })
    with open(path, "w", encoding="utf-8") as f:
        json.dump(debug_data, f, ensure_ascii=False, indent=2)
    log.info(f"调试来源文件已保存: {path}")


# ====================== 批量入口 ======================

def process_file(pdf_path: str, output_dir: str) -> Optional[dict]:
    try:
        result = parse_pdf(pdf_path)

        # ===================== 新增：如果是空结果（摘要），直接返回 None，不保存
        if not result:
            log.info(f"✅ 该文件为摘要/空文件，已跳过，不保存到CSV：{os.path.basename(pdf_path)}")
            return None
        # =====================

        is_valid, errors = validate_consistency(result)
        if not is_valid:
            log.warning(f"[{os.path.basename(pdf_path)}] 自动校验发现 {len(errors)} 条问题（仍入库）: {errors}")
        else:
            log.info(f"[{os.path.basename(pdf_path)}] 自动校验全部通过 ✓")

        print_summary(result)
        report_missing_fields(result)
        #save_json(result, output_dir)
        return result

    except Exception as e:
        log.error(f"处理 {pdf_path} 失败: {e}")
        return None


def main():
    import argparse
    ap = argparse.ArgumentParser(description="财报 PDF 批量解析（DeepDeSRT思想增强版）")
    ap.add_argument("input", nargs="?", help="单个PDF路径 或 文件夹路径")
    ap.add_argument("--input-dir", help="批量处理文件夹（所有 .pdf 文件）")
    ap.add_argument("--output-dir", default="./output", help="输出目录")
    args = ap.parse_args()

    output_dir = args.output_dir

    if args.input_dir:
        input_dir = Path(args.input_dir)
        if not input_dir.is_dir():
            log.error(f"输入文件夹不存在: {input_dir}")
            sys.exit(1)
        pdf_files = sorted(list(input_dir.glob("*.pdf")) + list(input_dir.glob("*.PDF")))
        if not pdf_files:
            log.warning(f"文件夹 {input_dir} 中未找到任何 PDF 文件")
            return
        log.info(f"发现 {len(pdf_files)} 个 PDF 文件，开始批量处理...")
    elif args.input:
        input_path = Path(args.input)
        if input_path.is_file() and input_path.suffix.lower() == ".pdf":
            pdf_files = [input_path]
        elif input_path.is_dir():
            pdf_files = sorted(list(input_path.glob("*.pdf")) + list(input_path.glob("*.PDF")))
        else:
            log.error(f"输入路径无效: {args.input}")
            sys.exit(1)
    else:
        log.error("请提供 --input-dir 或 单个PDF路径")
        sys.exit(1)

    all_results = []
    for pdf_path in pdf_files:
        res = process_file(str(pdf_path), output_dir)
        if res:
            all_results.append(res)

    save_global_csvs(all_results, output_dir)
    save_debug_sources(all_results, output_dir)
    log.info(f"全部处理完成！共 {len(all_results)} 份财报 → 全局 4 张 CSV 已生成")


if __name__ == "__main__":
    # 固定路径设置
    PDF_FOLDER = r"D:\竞赛\泰迪杯\B题数据及提交说明\附件2：财务报告\reports-深交所"
    OUTPUT_FOLDER = r"D:\竞赛\泰迪杯\深交所财务报告"  # 你也可以修改这个输出路径

    # 确保输出目录存在
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # 处理所有PDF文件
    all_results = []
    pdf_files = []
    for ext in ["*.pdf"]:
        pdf_files.extend(Path(PDF_FOLDER).glob(ext))

    if not pdf_files:
        print(f"在 {PDF_FOLDER} 中未找到PDF文件")
    else:
        print(f"找到 {len(pdf_files)} 个PDF文件，开始解析...")
        for pdf_path in pdf_files:
            result = process_file(str(pdf_path), OUTPUT_FOLDER)
            if result:
                all_results.append(result)

        # 保存结果
        if all_results:
            save_global_csvs(all_results, OUTPUT_FOLDER)
            save_debug_sources(all_results, OUTPUT_FOLDER)
            print(f"处理完成！共处理 {len(all_results)} 个PDF文件，结果保存在: {OUTPUT_FOLDER}")