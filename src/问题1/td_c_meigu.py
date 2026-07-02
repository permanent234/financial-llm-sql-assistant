import os
import re
import pandas as pd
import pdfplumber

# ====================== 配置 ======================
PDF_FOLDER = r"D:/竞赛/泰迪杯/B题数据及提交说明/附件2：财务报告/reports-深交所"  # 你的PDF文件夹路径
OUTPUT_FILE = "D:/竞赛/泰迪杯/财报提取结果.xlsx"

# ====================== 跳过摘要 ======================
def is_summary(text):
    check = text[:2000]
    for kw in ["摘要", "报告摘要", "简要报告"]:
        if kw in check:
            return True
    return False

# ====================== 提取元信息（照搬你代码的成熟逻辑） ======================
def extract_meta(text, filename):
    # 股票代码
    m = re.search(r"(公司代码|股票代码|证券代码)[：:]\s*([0-9]{6})", text)
    stock_code = m.group(2) if m else ""

    # 股票简称
    m = re.search(r"(公司简称|股票简称|证券简称)[：:]\s*([^\n\r\s，。、]+)", text)
    stock_abbr = m.group(2).strip() if m else ""

    if not stock_abbr:
        fn = os.path.basename(filename)
        m2 = re.search(r"([A-Za-z\u4e00-\u9fa5]+)", fn)
        if m2:
            stock_abbr = m2.group(1)

    # 报告期年份
    year = None
    m = re.search(r"(20\d{2})年", text[:4000])
    if m:
        year = int(m.group(1))
    else:
        m2 = re.search(r"(20\d{2})", filename)
        if m2:
            year = int(m2.group(1))

    # 报告期类型 FY/Q1/HY/Q3
    fn = filename
    if "第一季度" in fn or "一季度" in fn:
        period_type = "Q1"
    elif "半年度" in fn or "半年报" in fn:
        period_type = "HY"
    elif "第三季度" in fn or "三季度" in fn:
        period_type = "Q3"
    elif "年度报告" in fn or "年报" in fn:
        period_type = "FY"
    else:
        period_type = ""

    return {
        "股票代码": stock_code,
        "股票简称": stock_abbr,
        "报告期类型": period_type,
        "报告期年份": year
    }

# ====================== 清洗数字 ======================
def clean_num(s):
    if not s:
        return None
    s = str(s).replace(",", "").replace(" ", "").replace("(", "-").replace(")", "")
    try:
        return float(s)
    except:
        return None

# ====================== 提取财务3指标 ======================
def extract_financial(text):
    res = {
        "归属于上市公司股东的净资产(元)": None,
        "净资产收益率(%)": None,
        "扣非净利润(元)": None
    }

    # 1. 净资产
    patterns_net = [
        r"归属于上市公司股东的净资产\s*[:：]?\s*([\d,.]+)",
        r"归属于母公司所有者权益合计\s*[:：]?\s*([\d,.]+)"
    ]
    for p in patterns_net:
        match = re.search(p, text)
        if match:
            val = clean_num(match.group(1))
            if val:
                res["归属于上市公司股东的净资产(元)"] = val
                break

    # 2. ROE
    patterns_roe = [
        r"加权平均净资产收益率\s*[:：]?\s*([\d.]+)\s*%",
        r"净资产收益率\s*[:：]?\s*([\d.]+)\s*%"
    ]
    for p in patterns_roe:
        match = re.search(p, text)
        if match:
            val = clean_num(match.group(1))
            if val:
                res["净资产收益率(%)"] = val
                break

    # 3. 扣非净利润
    patterns_profit = [
        r"归属于.*?扣除非.*?净利润\s*[:：]?\s*([\d,.]+)",
        r"扣除非经常性损益后的净利润\s*[:：]?\s*([\d,.]+)",
        r"扣非净利润\s*[:：]?\s*([\d,.\(\)-]+)",
        r"归属于.*?扣非.*?净利润\s*[:：]?\s*([\d,.\(\)-]+)",
    ]
    for p in patterns_profit:
        match = re.search(p, text)
        if match:
            val = clean_num(match.group(1))
            if val:
                res["扣非净利润(元)"] = val
                break

    return res

# ====================== 处理单个PDF ======================
def process_pdf(path):
    try:
        with pdfplumber.open(path) as pdf:
            full_text = ""
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    full_text += t + "\n"

        if is_summary(full_text):
            print(f"⏭️  跳过摘要：{os.path.basename(path)}")
            return None

        meta = extract_meta(full_text, os.path.basename(path))
        finance = extract_financial(full_text)
        return {**meta, **finance}

    except Exception as e:
        print(f"❌ 失败：{os.path.basename(path)} | {str(e)}")
        return None

# ====================== 批量处理 ======================
def batch():
    data = []
    for f in os.listdir(PDF_FOLDER):
        if f.lower().endswith(".pdf"):
            p = os.path.join(PDF_FOLDER, f)
            res = process_pdf(p)
            if res:
                data.append(res)
                print(f"✅ 成功：{f} | {res['报告期年份']}{res['报告期类型']}")

    df = pd.DataFrame(data)
    df = df[[
        "股票代码", "股票简称", "报告期类型", "报告期年份",
        "归属于上市公司股东的净资产(元)", "净资产收益率(%)", "扣非净利润(元)"
    ]]
    df.to_excel(OUTPUT_FILE, index=False)
    print(f"\n🎉 全部完成！结果已保存到：{OUTPUT_FILE}")

if __name__ == "__main__":
    batch()