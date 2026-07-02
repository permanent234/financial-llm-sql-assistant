import os
import re
import json
import sqlite3
import logging
import argparse
from typing import List, Dict, Tuple, Optional
import pandas as pd
import torch
import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams, font_manager
from sentence_transformers import SentenceTransformer, util
from openai import OpenAI
import platform
# 设置中文字体(图片标题为中文）
system = platform.system()
rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
def set_chinese_font() -> Optional[str]:
    candidates = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Noto Sans CJK JP",
        "Source Han Sans SC",
        "WenQuanYi Zen Hei",
        "Arial Unicode MS",
        "PingFang SC",
        "Heiti SC",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            return name
_CN_FONT = set_chinese_font()
if _CN_FONT:
    rcParams["font.sans-serif"] = [_CN_FONT, "DejaVu Sans"]
else:
    rcParams["font.sans-serif"] = ["DejaVu Sans"]
rcParams["axes.unicode_minus"] = False
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)
TABLE_NAMES = [
    "core_performance_indicators_sheet",
    "balance_sheet",
    "income_sheet",
    "cash_flow_sheet",
]
# DeepSeek调用
class DeepSeekClient:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY", "")
        if not self.api_key:
            raise ValueError("请设置 DEEPSEEK_API_KEY 环境变量")
        self.client = OpenAI(api_key=self.api_key, base_url="https://api.deepseek.com")
    def chat(self, messages: List[Dict], temperature: float = 0.0, max_tokens: int = 512) -> str:
        resp = self.client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content.strip()
#SQLGPT主类
class SQLGPTInference:
    def __init__(self, data_dir: str, top_k: int = 2, llm_api_key: Optional[str] = None):
        self.data_dir = data_dir
        self.db_path = os.path.join(data_dir, "financial_reports.db")
        self.top_k = top_k
        self.history: List = []
        self._pending_clarify_query: Optional[str] = None
        self.last_result_df: Optional[pd.DataFrame] = None
        self.last_sql: str = ""
        self.last_effective_query: str = ""
        os.makedirs("./result", exist_ok=True)
        self._build_sqlite_db()
        self.table_schemas: Dict[str, str] = self._build_schemas_from_csv()
        self.company_list = self._get_company_list()
        self.latest_report_year = self._get_latest_report_year()
        self.metric_guardrails = {
            "未分配利润": ("equity_unappropriated_profit", "balance_sheet"),
            "股东权益-未分配利润": ("equity_unappropriated_profit", "balance_sheet"),
            "资产负债率": ("asset_liability_ratio", "balance_sheet"),
            "利润总额": ("total_profit", "income_sheet"),
            "经营活动现金流量净额": ("operating_cf_net_amount", "cash_flow_sheet"),
            "经营性现金流-现金流量净额": ("operating_cf_net_amount", "cash_flow_sheet"),
            "投资性现金流量净额":("investing_cf_net_amount","cash_flow_sheet"),
            "每股经营现金流量": ("operating_cf_per_share", "core_performance_indicators_sheet"),
            "销售毛利率": ("gross_profit_margin", "core_performance_indicators_sheet"),
            "营业总收入": ("total_operating_revenue", None),
            "净利润": ("net_profit_10k_yuan", "income_sheet"),
            "每股收益": ("eps", "core_performance_indicators_sheet"),
        }
        self.embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        self.schema_embeddings = self.embedder.encode(
            list(self.table_schemas.values()), convert_to_tensor=True, show_progress_bar=False
        )
        self.examples = self._load_financial_examples()
        self.llm = DeepSeekClient(llm_api_key)
        self.current_year = datetime.datetime.now().year
        log.info(f"当前年份设置为: {self.current_year}")
        log.info(f"数据最新年份: {self.latest_report_year}")
        log.info(f"图表中文字体: {_CN_FONT or '未检测到可用中文字体，可能仍会乱码'}")
        log.info("SQLGPT 初始化完成")
    #定义表结构
    def _build_schemas_from_csv(self) -> Dict[str, str]:
        schemas = {}
        schemas["core_performance_indicators_sheet"] = """
        核心业绩指标表 (core_performance_indicators_sheet)
        字段说明：
        - serial_number: 序号 (int)
        - stock_code: 股票代码 (varchar(20))
        - stock_abbr: 股票简称 (varchar(50))
        - eps: 每股收益(元) (decimal(10,4))
        - total_operating_revenue: 营业总收入(万元) (decimal(20,2))
        - operating_revenue_yoy_growth: 营业总收入-同比增长(%) (decimal(10,4))
        - operating_revenue_qoq_growth: 营业总收入-季度环比增长(%) (decimal(10,4))
        - net_profit_10k_yuan: 净利润(万元) (decimal(20,2))
        - net_profit_yoy_growth: 净利润-同比增长(%) (decimal(10,4))
        - net_profit_qoq_growth: 净利润-季度环比增长(%) (decimal(10,4))
        - net_asset_per_share: 每股净资产(元) (decimal(10,4))
        - roe: 净资产收益率(%) (decimal(10,4))
        - operating_cf_per_share: 每股经营现金流量(元) (decimal(10,4))
        - net_profit_excl_non_recurring: 扣非净利润（万元） (decimal(20,2))
        - net_profit_excl_non_recurring_yoy: 扣非净利润同比增长（%） (decimal(10,4))
        - gross_profit_margin: 销售毛利率(%) (decimal(10,4))
        - net_profit_margin: 销售净利率（%） (decimal(10,4))
        - roe_weighted_excl_non_recurring: 加权平均净资产收益率（扣非）（%） (decimal(10,4))
        - report_period: 报告期 (varchar(20))
        - report_year: 报告期-年份 (INT)
        适合查询：每股收益、净利润、ROE、毛利率、扣非净利润等核心指标。"""
        schemas["balance_sheet"] = """
        资产负债表 (balance_sheet)
        字段说明：
        - serial_number: 序号 (int)
        - stock_code: 股票代码 (varchar(20))
        - stock_abbr: 股票简称 (varchar(50))
        - asset_cash_and_cash_equivalents: 资产-货币资金(万元) (decimal(20,2))
        - asset_accounts_receivable: 资产-应收账款(万元) (decimal(20,2))
        - asset_inventory: 资产-存货(万元) (decimal(20,2))
        - asset_trading_financial_assets: 资产-交易性金融资产（万元） (decimal(20,2))
        - asset_construction_in_progress: 资产-在建工程（万元） (decimal(20,2))
        - asset_total_assets: 资产-总资产(万元) (decimal(20,2))
        - asset_total_assets_yoy_growth: 资产-总资产同比(%) (decimal(10,4))
        - liability_accounts_payable: 负债-应付账款(万元) (decimal(20,2))
        - liability_advance_from_customers: 负债-预收账款(万元) (decimal(20,2))
        - liability_total_liabilities: 负债-总负债(万元) (decimal(20,2))
        - liability_total_liabilities_yoy_growth: 负债-总负债同比(%) (decimal(10,4))
        - liability_contract_liabilities: 负债-合同负债（万元） (decimal(20,2))
        - liability_short_term_loans: 负债-短期借款（万元） (decimal(20,2))
        - asset_liability_ratio: 资产负债率(%) (decimal(10,4))
        - equity_unappropriated_profit: 股东权益-未分配利润（万元） (decimal(20,2))
        - equity_total_equity: 股东权益合计(万元) (decimal(20,2))
        - report_period: 报告期 (varchar(20))
        - report_year: 报告期-年份 (INT)
        适合查询：总资产、总负债、资产负债率、货币资金、存货等。"""
        schemas["income_sheet"] = """
        利润表 (income_sheet)
        字段说明：
        - serial_number: 序号 (int)
        - stock_code: 股票代码 (varchar(20))
        - stock_abbr: 股票简称 (varchar(50))
        - net_profit: 净利润(万元) (decimal(20,2))
        - net_profit_yoy_growth: 净利润同比(%) (decimal(10,4))
        - other_income: 其他收益（万元） (decimal(20,2))
        - total_operating_revenue: 营业总收入(万元) (decimal(20,2))
        - operating_revenue_yoy_growth: 营业总收入同比(%) (decimal(10,4))
        - operating_expense_cost_of_sales: 营业总支出-营业支出(万元) (decimal(20,2))
        - operating_expense_selling_expenses: 营业总支出-销售费用(万元) (decimal(20,2))
        - operating_expense_administrative_expenses: 营业总支出-管理费用(万元) (decimal(20,2))
        - operating_expense_financial_expenses: 营业总支出-财务费用(万元) (decimal(20,2))
        - operating_expense_rnd_expenses: 营业总支出-研发费用（万元） (decimal(20,2))
        - operating_expense_taxes_and_surcharges: 营业总支出-税金及附加（万元） (decimal(20,2))
        - total_operating_expenses: 营业总支出(万元) (decimal(20,2))
        - operating_profit: 营业利润(万元) (decimal(20,2))
        - total_profit: 利润总额(万元) (decimal(20,2))
        - asset_impairment_loss: 资产减值损失（万元） (decimal(20,2))
        - credit_impairment_loss: 信用减值损失（万元） (decimal(20,2))
        - report_period: 报告期 (varchar(20))
        - report_year: 报告期-年份 (INT)
        适合查询：净利润、利润总额、营业总收入、营业利润等。
        """
        schemas["cash_flow_sheet"] = """
        现金流量表 (cash_flow_sheet)
        字段说明：
        - serial_number: 序号 (int)
        - stock_code: 股票代码 (varchar(20))
        - stock_abbr: 股票简称 (varchar(50))
        - net_cash_flow: 净现金流(元) (decimal(20,2))
        - net_cash_flow_yoy_growth: 净现金流-同比增长(%) (decimal(10,4))
        - operating_cf_net_amount: 经营性现金流-现金流量净额(万元) (decimal(20,2))
        - operating_cf_ratio_of_net_cf: 经营性现金流-净现金流占比(%) (decimal(10,4))
        - operating_cf_cash_from_sales: 经营性现金流-销售商品收到的现金（万元） (decimal(20,2))
        - investing_cf_net_amount: 投资性现金流-现金流量净额(万元) (decimal(20,2))
        - investing_cf_ratio_of_net_cf: 投资性现金流-净现金流占比(%) (decimal(10,4))
        - investing_cf_cash_for_investments: 投资性现金流-投资支付的现金（万元） (decimal(20,2))
        - investing_cf_cash_from_investment_recovery: 投资性现金流-收回投资收到的现金（万元） (decimal(20,2))
        - financing_cf_cash_from_borrowing: 融资性现金流-取得借款收到的现金（万元） (decimal(20,2))
        - financing_cf_cash_for_debt_repayment: 融资性现金流-偿还债务支付的现金（万元） (decimal(20,2))
        - financing_cf_net_amount: 融资性现金流-现金流量净额(万元) (decimal(20,2))
        - financing_cf_ratio_of_net_cf: 融资性现金流-净现金流占比(%) (decimal(10,4))
        - report_period: 报告期 (varchar(20))
        - report_year: 报告期-年份 (INT)
        适合查询：经营活动现金流量净额、净现金流等现金流量指标。"""
        return schemas

    def _get_company_list(self) -> List[str]:
        company_set = set()
        try:
            for table_name in ["core_performance_indicators_sheet", "balance_sheet", "income_sheet", "cash_flow_sheet"]:
                csv_path = os.path.join(self.data_dir, f"{table_name}.csv")
                if os.path.exists(csv_path):
                    df = pd.read_csv(csv_path, encoding="utf-8-sig")
                    if "stock_abbr" in df.columns:
                        names = df["stock_abbr"].dropna().astype(str).str.strip().unique().tolist()
                        company_set.update(names)
        except Exception as e:
            log.warning(f"加载公司列表失败: {e}，使用默认列表")
            default_companies = [
            "金花股份", "华润三九", "片仔癀", "同仁堂", "太极集团",
            "中恒集团", "白云山", "东阿阿胶", "羚锐制药", "昆药集团"]
        # 构建公司简称映射（支持常见简称识别）
        self.company_abbr_map = {
            "金花": "金花股份",
            "三金": "桂林三金",
            "华润": "华润三九",
            "片仔": "片仔癀",
            "同仁": "同仁堂",
            "太极": "太极集团",
            "中恒": "中恒集团",
            "白云": "白云山",
            "东阿": "东阿阿胶",
            "羚锐": "羚锐制药",
            "昆药": "昆药集团",
            "云南白": "云南白药",
            "仁和": "仁和药业",
            "济川": "济川药业",
            "达仁": "达仁堂",
        }
        company_list = sorted(list(company_set)) if company_set else default_companies
        log.info(f"共加载 {len(company_list)} 家上市公司简称")
        return company_list
    # 定义现在的时间，便于计算年度趋势
    def _get_latest_report_year(self) -> int:
        years = []
        for table_name in TABLE_NAMES:
            csv_path = os.path.join(self.data_dir, f"{table_name}.csv")
            if not os.path.exists(csv_path):
                continue
            try:
                df = pd.read_csv(csv_path, usecols=["report_year"], encoding="utf-8-sig")
                vals = pd.to_numeric(df["report_year"], errors="coerce").dropna().astype(int)
                if not vals.empty:
                    years.append(int(vals.max()))
            except Exception:
                continue
        return max(years) if years else self.current_year if hasattr(self, "current_year") else 2025

    def _load_financial_examples(self) -> List[Tuple[str, str]]:
        y = self.latest_report_year
        return [
            ("金花股份2022年的每股收益是多少",
             "SELECT eps FROM core_performance_indicators_sheet WHERE stock_abbr = '金花股份' AND report_period = 'FY' AND report_year = 2022"),
            (f"{y}年利润总额最高的前10家公司是哪些",
             f"SELECT stock_abbr, total_profit FROM income_sheet WHERE report_period = 'FY' AND report_year = {y} ORDER BY total_profit DESC LIMIT 10"),
            ("华润三九2023年的净利润是多少",
             "SELECT net_profit FROM income_sheet WHERE stock_abbr = '华润三九' AND report_period = 'FY' AND report_year = 2023"),
            ("金花股份2022年的资产负债率是多少",
             "SELECT asset_liability_ratio FROM balance_sheet WHERE stock_abbr = '金花股份' AND report_period = 'FY' AND report_year = 2022"),
            (f"{y}年上半年总资产最大的三家公司",
             f"SELECT stock_abbr, asset_total_assets FROM balance_sheet WHERE report_period = 'HY' AND report_year = {y} ORDER BY asset_total_assets DESC LIMIT 3"),
            ("金花股份近三年的利润总额变化趋势是什么样的",
             f"SELECT report_year, total_profit FROM income_sheet WHERE stock_abbr = '金花股份' AND report_period = 'FY' AND report_year IN ({y-2}, {y-1}, {y}) ORDER BY report_year"),
            ("金花股份近三年的经营活动现金流量净额变化趋势是什么样的",
             f"SELECT report_year, operating_cf_net_amount FROM cash_flow_sheet WHERE stock_abbr = '金花股份' AND report_period = 'FY' AND report_year IN ({y-2}, {y-1}, {y}) ORDER BY report_year"),]

    def _build_sqlite_db(self):
        os.makedirs(self.data_dir, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        for table_name in TABLE_NAMES:
            csv_path = os.path.join(self.data_dir, f"{table_name}.csv")
            if not os.path.exists(csv_path):
                continue
            df = pd.read_csv(csv_path, encoding="utf-8-sig")
            df.to_sql(table_name, conn, if_exists="replace", index=False)
            log.info(f"[DB] {table_name} 已入库：{len(df)} 行")
        conn.close()

    #意图澄清部分
    def clarify_intent_if_needed(self, query: str) -> Optional[str]:
        if not query:
            return None
        q = query.strip()
        query_lower = q.lower()
        set_query_keywords = [
            "最高", "最低", "前", "排名", "最多", "最少", "最大", "最小",
            "所有公司", "哪些公司", "哪几家公司", "多少家", "公司有哪些", "股票代码",
            "趋势", "变化", "为正", "低于", "高于", "近三年", "近两年", "上半年", "下半年", "全年",
            "分布", "相关性", "对比", "比较", "占比", "比值", "行业", "平均", "名单变化",
            "统计", "排序", "筛选", "超过", "生成", "展示", "绘制", "不一致", "校对", "关联"
        ]
        if any(k in q for k in set_query_keywords):
            return None
        if self._is_time_fragment_followup(q):
            return None
        has_company = any(company in q for company in self.company_list)
        if not has_company:
            for abbr, full_name in self.company_abbr_map.items():
                if abbr in q:
                    has_company = True
                    break
        indicator_keywords = ["收益", "利润", "收入", "资产负债率", "eps", "roe", "净利润", "每股", "总资产", "总负债", "现金流", "毛利率", "净利率", "未分配利润"]
        has_indicator = any(k in query_lower for k in indicator_keywords)
        if has_indicator and not has_company:
            if any(x in query_lower for x in ["每股收益", "eps"]):
                return "请问你要查询哪一家公司的每股收益？（例如：金花股份）"
            if "未分配利润" in query_lower:
                return "请问你要查询哪一家公司的未分配利润？（例如：金花股份）"
            if any(x in query_lower for x in ["净利润", "利润"]):
                return "请问你要查询哪一家公司的净利润/利润总额？（例如：金花股份）"
            if "资产负债率" in query_lower:
                return "请问你要查询哪一家公司的资产负债率？（例如：金花股份）"
            if "总资产" in query_lower:
                return "请问你要查询哪一家公司的总资产？（例如：金花股份）"
            return "请问你要查询哪一家公司的财务数据？（例如：金花股份）"
        return None
    # 判断是否是时间片段的后续追问
    def _is_time_fragment_followup(self, query: str) -> bool:
        q = query.strip()
        if len(q) > 24:
            return False
        patterns = [
            r"^20\d{2}年$",
            r"^20\d{2}年(第一季度|第二季度|第三季度|第四季度|上半年|下半年|全年|年报|半年报)$",
            r"^(今年|去年|前年)$",
            r"^(第一季度|第二季度|第三季度|第四季度|上半年|下半年|全年|年报|半年报)$",
        ]
        return any(re.match(p, q) for p in patterns)
    def retrieve_schemas(self, query: str) -> List[str]:
        query_emb = self.embedder.encode(query, convert_to_tensor=True)
        cos_scores = util.cos_sim(query_emb, self.schema_embeddings)[0]
        k = min(max(self.top_k, 3), len(self.table_schemas))
        top_idx = torch.topk(cos_scores, k=k).indices.tolist()
        retrieved = [list(self.table_schemas.keys())[i] for i in top_idx]
        q = query
        boost_tables = []
        for phrase, (_, table_name) in self.metric_guardrails.items():
            if phrase in q and table_name and table_name not in boost_tables:
                boost_tables.append(table_name)
        for t in reversed(boost_tables):
            if t in retrieved:
                retrieved.remove(t)
            retrieved.insert(0, t)
        retrieved = retrieved[:k]
        log.info(f"RAG 检索到表（top-{k}）：{retrieved}")
        return retrieved
    # 构建提示词模块
    def build_prompt(self, query: str, retrieved_tables: List[str]) -> str:
        base_year = self.latest_report_year or self.current_year
        recent3 = f"({base_year-2}, {base_year-1}, {base_year})"
        recent2 = f"({base_year-1}, {base_year})"
        guardrail_lines = []
        for phrase, (field_name, table_name) in self.metric_guardrails.items():
            if phrase in query:
                if table_name:
                    guardrail_lines.append(f"- 用户问到{phrase}时，优先使用字段 {field_name}，表 {table_name}。")
                else:
                    guardrail_lines.append(f"- 用户问到{phrase}时，优先使用字段 {field_name}，不要映射成无关字段。")
        p_base = (
            "你是一位精通财务报表的 SQL 专家，数据库引擎是 SQLite。\n"
            "重要规则：\n"
            "1. 只输出一条 SQL 语句，不要任何解释。\n"
            "2. report_period 字段的值是 'FY' 表示全年，'Q1' 表示第一季度，'HY' 表示半年，'Q3' 表示第三季度。\n"
            "3. 如果用户提到具体年份，必须同时使用 report_period 和 report_year 两个字段。\n"
            "4. 如果查询中没有具体公司名称，且查询的是单家公司具体财务指标，才允许追问；如果是排名、筛选、统计、对比、哪几家公司等，必须直接返回 SQL。\n"
            "5. 严格使用 Schema 中真实存在的字段名。\n"
            "6. 禁止使用 CURDATE()、NOW()、YEAR() 等 MySQL 专有函数，SQLite 不支持。\n"
            f"   近三年默认写成：report_year IN {recent3}\n"
            f"   近两年默认写成：report_year IN {recent2}\n"
            f"7. 如果用户未明确指定未来年份，默认不要使用超过 {base_year} 的年份。\n"
            "8. 禁止把行业条件写成 stock_abbr LIKE '%中药%' 这类简称模糊匹配；没有行业字段时，可保留其他条件准确。\n"
            "9. 查询总量指标时，禁止误用每股指标。比如经营活动现金流量净额不能映射成 operating_cf_per_share，未分配利润不能映射成 net_asset_per_share。\n"
        )
        p_schema = "【数据库 Schema】\n" + "\n\n".join(self.table_schemas[t] for t in retrieved_tables)
        p_example = "【示例】\n" + "\n".join([f"查询: {q}\nSQL: {s}\n" for q, s in self.examples[:4]])
        p_guard = "【高风险字段约束】\n" + "\n".join(guardrail_lines) if guardrail_lines else ""
        history_str = ""
        if self.history:
            lines = []
            for item in self.history[-6:]:
                if isinstance(item, (list, tuple)):
                    q = str(item[0]) if len(item) > 0 else ""
                    s = str(item[1]) if len(item) > 1 else ""
                elif isinstance(item, dict):
                    q = str(item.get("rewritten", item.get("original", item.get("query", ""))))
                    s = str(item.get("sql_or_msg", item.get("sql", "")))
                else:
                    q = str(item)
                    s = ""
                lines.append(f"用户: {q}\nSQL: {s}")
            history_str = "【历史对话（请记住前面的公司和年份）】\n" + "\n".join(lines)

        p_query = f"【当前查询】\n{query}\n\n请直接输出一条 SQL 语句："
        full_prompt = "\n\n".join(filter(None, [p_base, p_schema, p_example, p_guard, history_str, p_query]))
        return full_prompt.strip()
    # 调用ai生成sql语句
    def generate_sql(self, prompt: str) -> str:
        messages = [
            {"role": "system", "content": "你是一个 SQLite SQL 生成助手，只输出一条 SQL 语句，不要任何解释。禁止使用 CURDATE()、NOW()、YEAR() 等 MySQL 专有函数。"},
            {"role": "user", "content": prompt}
        ]
        raw = self.llm.chat(messages, temperature=0.0, max_tokens=512)
        sql = self.extract_sql(raw)
        sql = self.correct_sql_columns(sql)
        log.info(f"生成 SQL: {sql}")
        return sql
    def correct_sql_columns(self, sql: str) -> str:
        if not sql or not isinstance(sql, str):
            return sql
        column_map = {
            "eps_basic": "eps",
            "operating_revenue": "total_operating_revenue",
        }
        for wrong, correct in column_map.items():
            sql = re.sub(rf"\b{wrong}\b", correct, sql, flags=re.IGNORECASE)
        #日期匹配
        sql = re.sub(r"report_period\s*=\s*'(\d{4})FY'", r"report_period = 'FY' AND report_year = \1", sql, flags=re.IGNORECASE)
        sql = re.sub(r"report_period\s*=\s*'(\d{4})Q([1-3])'", r"report_period = 'Q\2' AND report_year = \1", sql, flags=re.IGNORECASE)
        sql = re.sub(r"report_period\s*=\s*'(\d{4})HY'", r"report_period = 'HY' AND report_year = \1", sql, flags=re.IGNORECASE)
        sql = re.sub(r"report_period\s*=\s*'(\d{4})-Q([1-3])'",r"report_period = 'Q\2' AND report_year = \1", sql, flags=re.IGNORECASE)
        sql = re.sub(r"report_period\s*=\s*'(\d{4})-(HY|FY)'",r"report_period = '\2' AND report_year = \1", sql, flags=re.IGNORECASE)
        base_year = self.latest_report_year or self.current_year
        sql = re.sub(
            r"YEAR\s*\(\s*(?:CURDATE|NOW)\s*\(\s*\)\s*\)\s*-\s*(\d+)",
            lambda m: str(base_year - int(m.group(1))),
            sql, flags=re.IGNORECASE,
        )
        sql = re.sub(r"YEAR\s*\(\s*(?:CURDATE|NOW)\s*\(\s*\)\s*\)", str(base_year), sql, flags=re.IGNORECASE)
        if "未分配利润" in sql or "equity_unappropriated_profit" in sql.lower():
            sql = re.sub(r"FROM\s+\w+", "FROM balance_sheet", sql, flags=re.IGNORECASE)
            sql = re.sub(r"\bnet_asset_per_share\b", "equity_unappropriated_profit", sql, flags=re.IGNORECASE)
        if any(k in sql.lower() for k in ["operating_cf_net_amount", "asset_liability_ratio", "equity_unappropriated_profit"]):
            table_map = {
                "operating_cf_net_amount": "cash_flow_sheet",
                "asset_liability_ratio": "balance_sheet",
                "equity_unappropriated_profit": "balance_sheet",
            }
            for field, table_name in table_map.items():
                if field in sql.lower():
                    sql = re.sub(r"FROM\s+\w+", f"FROM {table_name}", sql, flags=re.IGNORECASE)
        if re.search(r"\beps\b", sql, re.IGNORECASE) and "core_performance_indicators_sheet" not in sql.lower():
            sql = re.sub(r"FROM\s+\w+", "FROM core_performance_indicators_sheet", sql, flags=re.IGNORECASE)
        if "operating_cf_per_share" in sql.lower() and any(x in sql.lower() for x in ["现金流量净额", "经营活动现金流量净额", "经营性现金流净额"]):
            sql = re.sub(r"\boperating_cf_per_share\b", "operating_cf_net_amount", sql, flags=re.IGNORECASE)
            sql = re.sub(r"FROM\s+\w+", "FROM cash_flow_sheet", sql, flags=re.IGNORECASE)
        if re.search(r"stock_abbr\s+like\s+'%[^']*(中药|医药|行业)[^']*%'", sql, re.IGNORECASE):
            sql = re.sub(r"\s+AND\s+stock_abbr\s+LIKE\s+'%[^']*%'", "", sql, flags=re.IGNORECASE)
            sql = re.sub(r"\s+WHERE\s+stock_abbr\s+LIKE\s+'%[^']*%'", "", sql, flags=re.IGNORECASE)
        return sql

    @staticmethod
    def extract_sql(text: str) -> str:
        text = text.strip()
        text = re.sub(r"^```sql\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"^```\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        m = re.search(r"(SELECT\b.*?)(?:;?\s*$)", text, flags=re.IGNORECASE | re.DOTALL)
        if m:
            sql = m.group(1).strip()
            return re.sub(r"\s+", " ", sql).rstrip(";")
        return text.rstrip(";")
    def repair_sql_with_llm(self, query: str, bad_sql: str, error_msg: str, retrieved_tables: List[str]) -> str:
        schema_str = "\n\n".join(self.table_schemas[t] for t in retrieved_tables)
        messages = [
            {"role": "system", "content": "你是一个谨慎的 SQLite SQL 修复助手。只做最小化修改，优先使用单表查询，绝不自行计算或加 JOIN。禁止使用 CURDATE()、NOW()、YEAR() 等 MySQL 专有函数。不要把总量指标映射成每股指标。"},
            {"role": "user", "content": f"""
            【数据库 Schema】
            {schema_str}
            【用户问题】
            {query}
            【错误 SQL】
            {bad_sql}
            【报错信息】
            {error_msg}
            请只进行最小化修复，只输出一条修复后的 SQL："""}
        ]
        fixed_sql = self.llm.chat(messages, temperature=0.0, max_tokens=512)
        fixed_sql = self.extract_sql(fixed_sql)
        fixed_sql = self.correct_sql_columns(fixed_sql)
        log.info(f"修复后 SQL: {fixed_sql}")
        return fixed_sql
    def validate_and_fix_sql(self, query: str, sql: str, retrieved_tables: List[str]) -> str:
        sql = self.correct_sql_columns(sql)
        return sql
    def execute_query(self, sql: str) -> pd.DataFrame:
        conn = sqlite3.connect(self.db_path)
        try:
            return pd.read_sql_query(sql, conn)
        except Exception as e:
            log.error(f"SQL 执行失败: {e}\nSQL: {sql}")
            return pd.DataFrame({"error": [str(e)], "sql": [sql]})
        finally:
            conn.close()
    def query(self, natural_language: str) -> dict:
        original_query = natural_language.strip()
        if self._pending_clarify_query is not None:
            merged_query = self._pending_clarify_query + " " + original_query
            log.info(f"[多轮意图合并] 待澄清问题='{self._pending_clarify_query}' + 补充='{original_query}' → '{merged_query}'")
            self._pending_clarify_query = None
            natural_language = merged_query
        else:
            natural_language = original_query
        rewritten_query = natural_language
        clarify_msg = self.clarify_intent_if_needed(rewritten_query)
        log.info(f"[澄清判断] 查询: {rewritten_query} | has_company: {any(c in rewritten_query for c in self.company_list)} | clarify: {clarify_msg is not None}")
        if self.history and len(self.history) > 0:
            last_item = self.history[-1]
            last_content = last_item[1] if isinstance(last_item, (list, tuple)) else str(last_item)
            # 如果上一轮是正常查询（有公司或年份），且本轮输入很短，则尝试继承
            if ("请问你要查询哪一" not in str(last_content) and
                len(original_query) <= 25 and
                not any(company in original_query for company in self.company_list)):
                # 查找最近的有效查询
                for i in range(len(self.history)-1, -1, -1):
                    prev = self.history[i]
                    prev_q = prev[0] if isinstance(prev, (list, tuple)) else str(prev)
                    if any(company in prev_q for company in self.company_list) or re.search(r'20\d{2}', prev_q):
                        natural_language = prev_q + " " + original_query
                        log.info(f"[多轮上下文继承] 成功合并: {natural_language}")
                        break
                else:
                    natural_language = original_query
            else:
                natural_language = original_query
        else:
            natural_language = original_query
        if clarify_msg:
            self._pending_clarify_query = rewritten_query
            self.history.append({
                "type": "clarify",
                "original": original_query,
                "rewritten": rewritten_query,
                "sql_or_msg": clarify_msg,
            })
            return {
                "query": original_query,
                "effective_query": rewritten_query,
                "sql": "",
                "retrieved": [],
                "result": pd.DataFrame({"message": [clarify_msg]}),
                "clarify": True,
            }
        retrieved = self.retrieve_schemas(rewritten_query)
        prompt = self.build_prompt(rewritten_query, retrieved)
        sql = self.generate_sql(prompt)
        sql = self.validate_and_fix_sql(rewritten_query, sql, retrieved)
        result = self.execute_query(sql)
        if "error" in result.columns:
            error_msg = result["error"].iloc[0]
            log.warning(f"首次执行失败，尝试 LLM 修复: {error_msg}")
            fixed_sql = self.repair_sql_with_llm(rewritten_query, sql, error_msg, retrieved)
            fixed_sql = self.validate_and_fix_sql(rewritten_query, fixed_sql, retrieved)
            result = self.execute_query(fixed_sql)
            sql = fixed_sql
        self.history.append({
            "type": "query",
            "original": original_query,
            "rewritten": rewritten_query,
            "sql": sql,
        })
        if result is not None and not result.empty and "error" not in result.columns:
            self.last_result_df = result.copy()
            self.last_sql = sql
            self.last_effective_query = rewritten_query
        return {
            "query": original_query,
            "effective_query": rewritten_query,
            "sql": sql,
            "retrieved": retrieved,
            "result": result,
            "clarify": False,
        }
    #清空历史纪录，创建新的聊天空间
    def reset_history(self):
        self.history.clear()
        self._pending_clarify_query = None
        self.last_result_df = None
        self.last_sql = ""
        self.last_effective_query = ""
    def generate_chart(
        self,
        df: pd.DataFrame,
        query: str,
        question_id: str,
        turn_idx: int = 1,
        original_query: str = "",
    ) -> Tuple[List[str], str]:
        if df is None or df.empty or "error" in df.columns:
            return [], "None"
        save_dir = "./result"
        os.makedirs(save_dir, exist_ok=True)
        images: List[str] = []
        plot_df = df.copy()
        numeric_cols = [
            c for c in plot_df.select_dtypes(include="number").columns.tolist()
            if c != "report_year"
        ]
        str_cols = [
            c for c in plot_df.select_dtypes(exclude="number").columns.tolist()
            if c != "report_period"
        ]
        for c in str_cols:
            plot_df[c] = plot_df[c].astype(str)
        img_path = f"{save_dir}/{question_id}_{turn_idx}.jpg"
        img_rel  = f"./result/{question_id}_{turn_idx}.jpg"
        preferred = self._infer_chart_type_from_query(original_query or query, df)
        def _make_title(num_cols, q_id):
            if num_cols:
                readable = [c.replace("_", " ") for c in num_cols[:2]]
                return " & ".join(readable)
            return f"Chart {q_id}"
        try:
            # 1. Line chart
            if (
                preferred == "Line Chart"
                and "report_year" in plot_df.columns
                and numeric_cols
                and len(plot_df) >= 2
            ):
                title = _make_title(numeric_cols, question_id) + " Trend"
                fig, ax = plt.subplots(figsize=(11, 6))
                for col in numeric_cols[:4]:
                    ax.plot(
                        plot_df["report_year"].astype(str),
                        plot_df[col],
                        marker="o",
                        linewidth=2,
                        label=col.replace("_", " "),
                    )
                ax.set_title(title, fontsize=14, pad=20)
                ax.set_xlabel("Report Year", fontsize=12)
                ax.set_ylabel("Value (10k CNY)", fontsize=12)
                ax.legend(fontsize=10)
                ax.grid(True, alpha=0.3)
                plt.xticks(rotation=45, ha="right")
                fig.tight_layout()
                fig.savefig(img_path, dpi=160, bbox_inches="tight")
                plt.close(fig)
                return [img_rel], "Line Chart"
            if (
                preferred == "Horizontal Bar Chart"
                and str_cols
                and numeric_cols
                and 2 <= len(plot_df) <= 20
            ):
                title = _make_title(numeric_cols, question_id) + " Ranking"
                fig, ax = plt.subplots(figsize=(12, 7))
                ranked_df = plot_df.copy().sort_values(by=numeric_cols[0], ascending=True)
                ax.barh(
                    ranked_df[str_cols[0]].astype(str),
                    ranked_df[numeric_cols[0]],
                )
                ax.set_title(title, fontsize=14, pad=20)
                ax.set_xlabel(numeric_cols[0].replace("_", " "), fontsize=12)
                ax.set_ylabel("Company", fontsize=12)
                ax.grid(axis="x", alpha=0.3)
                fig.tight_layout()
                fig.savefig(img_path, dpi=160, bbox_inches="tight")
                plt.close(fig)
                return [img_rel], "Horizontal Bar Chart"
            if (
                preferred == "Bar Chart"
                and str_cols
                and numeric_cols
                and 2 <= len(plot_df) <= 20
            ):
                title = _make_title(numeric_cols, question_id) + " Comparison"
                fig, ax = plt.subplots(figsize=(12, 7))
                ax.bar(
                    plot_df[str_cols[0]].astype(str),
                    plot_df[numeric_cols[0]],
                )
                ax.set_title(title, fontsize=14, pad=20)
                ax.set_xlabel("Company", fontsize=12)
                ax.set_ylabel(numeric_cols[0].replace("_", " "), fontsize=12)
                plt.xticks(rotation=45, ha="right")
                ax.grid(axis="y", alpha=0.3)
                fig.tight_layout()
                fig.savefig(img_path, dpi=160, bbox_inches="tight")
                plt.close(fig)
                return [img_rel], "Bar Chart"
            if (
                preferred == "Pie Chart"
                and numeric_cols
                and str_cols
                and 2 <= len(plot_df) <= 10
            ):
                title = _make_title(numeric_cols, question_id) + " Distribution"
                fig, ax = plt.subplots(figsize=(9, 9))
                values = (
                    pd.to_numeric(plot_df[numeric_cols[0]], errors="coerce")
                    .fillna(0)
                    .abs()
                )
                labels = plot_df[str_cols[0]].astype(str)
                ax.pie(
                    values,
                    labels=labels,
                    autopct="%1.1f%%",
                    startangle=90,
                    textprops={"fontsize": 10},
                )
                ax.set_title(title, fontsize=14, pad=20)
                fig.savefig(img_path, dpi=160, bbox_inches="tight")
                plt.close(fig)
                return [img_rel], "Pie Chart"
            if "report_year" in plot_df.columns and numeric_cols and len(plot_df) >= 2:
                title = _make_title(numeric_cols, question_id) + " Trend"
                fig, ax = plt.subplots(figsize=(11, 6))
                for col in numeric_cols[:3]:
                    ax.plot(
                        plot_df["report_year"].astype(str),
                        plot_df[col],
                        marker="o",
                        linewidth=2,
                        label=col.replace("_", " "),
                    )
                ax.set_title(title, fontsize=14, pad=20)
                ax.set_xlabel("Report Year", fontsize=12)
                ax.set_ylabel("Value (10k CNY)", fontsize=12)
                ax.legend(fontsize=10)
                ax.grid(True, alpha=0.3)
                plt.xticks(rotation=45, ha="right")
                fig.tight_layout()
                fig.savefig(img_path, dpi=160, bbox_inches="tight")
                plt.close(fig)
                return [img_rel], "Line Chart"
            if str_cols and numeric_cols and 2 <= len(plot_df) <= 20:
                title = _make_title(numeric_cols, question_id) + " Comparison"
                fig, ax = plt.subplots(figsize=(12, 7))
                ax.bar(
                    plot_df[str_cols[0]].astype(str),
                    plot_df[numeric_cols[0]],
                )
                ax.set_title(title, fontsize=14, pad=20)
                ax.set_xlabel("Company", fontsize=12)
                ax.set_ylabel(numeric_cols[0].replace("_", " "), fontsize=12)
                plt.xticks(rotation=45, ha="right")
                ax.grid(axis="y", alpha=0.3)
                fig.tight_layout()
                fig.savefig(img_path, dpi=160, bbox_inches="tight")
                plt.close(fig)
                return [img_rel], "Bar Chart"
        except Exception as e:
            log.warning(f"Chart generation failed: {e}")
        return images, "None"
    def _clean_plot_dataframe(self, df: pd.DataFrame):
        plot_df = df.copy()
        numeric_cols = [
            c for c in plot_df.select_dtypes(include="number").columns.tolist()
            if c != "report_year"
        ]
        str_cols = [
            c for c in plot_df.select_dtypes(exclude="number").columns.tolist()
            if c != "report_period"
        ]
        for c in str_cols:
            plot_df[c] = plot_df[c].astype(str)
        return plot_df, numeric_cols, str_cols
    def _infer_chart_type_from_query(self, query: str, df: pd.DataFrame) -> str:
        if "雷达" in query:
            return "Radar Chart"
        if "水平柱状" in query or "水平条形" in query:
            return "Horizontal Bar Chart"
        if "双条形" in query:
            return "Double Bar Chart"
        if "柱状" in query or "条形" in query:
            return "Bar Chart"
        if "饼图" in query:
            return "Pie Chart"
        if "散点" in query:
            return "Scatter Plot"
        if "直方" in query:
            return "Histogram"
        if "箱线" in query:
            return "Box Plot"
        if "折线" in query or "趋势" in query or "变化" in query or "近三年" in query or "近两年" in query:
            return "Line Chart"
        if df is not None and "report_year" in df.columns and len(df) >= 2:
            return "Line Chart"
        return "Bar Chart"
    def batch_process(self, questions_file: str = "data/附件4：问题汇总.xlsx", output_file: str = "result_2.xlsx"):
        if not os.path.exists(questions_file):
            log.error(f"找不到问题文件: {questions_file}")
            print(f"请把附件4重命名为 {questions_file} 并放在当前目录")
            return
        try:
            df_questions = pd.read_excel(questions_file)
            log.info(f"成功加载 {len(df_questions)} 条问题记录")
        except Exception as e:
            log.error(f"读取问题文件失败: {e}")
            return
        results = []
        for idx, row in df_questions.iterrows():
            q_id = str(row.get("编号", f"Q{idx+1}")).strip()
            raw_q = str(row.get("问题", row.get("Q", ""))).strip()
            print(f"\n{'='*60}")
            print(f"[{q_id}] 原始问题字段: {raw_q[:80]}")
            turns: List[str] = []
            try:
                parsed = json.loads(raw_q)
                if isinstance(parsed, list):
                    turns = [item.get("Q", "").strip() for item in parsed if item.get("Q")]
                elif isinstance(parsed, dict):
                    turns = [parsed.get("Q", raw_q).strip()]
            except (json.JSONDecodeError, TypeError):
                turns = [raw_q]
            if not turns:
                turns = [raw_q]
            self.reset_history()
            dialogue: List[Dict] = []
            final_sql = ""
            chart_type = "无"
            turn_idx = 0
            q_ptr = 0
            while q_ptr < len(turns):
                current_q = turns[q_ptr]
                print(f"  第{q_ptr+1}轮输入: {current_q}")
                out = self.query(current_q)
                turn_idx += 1
                if out.get("clarify"):
                    clarify_content = out["result"]["message"].iloc[0]
                    dialogue.append({"Q": current_q, "A": {"content": clarify_content, "image": []}})
                    print(f"  系统追问: {clarify_content}")
                    q_ptr += 1
                    continue
                result_df = out.get("result")
                sql = out.get("sql", "")
                effective_query = out.get("effective_query", current_q)
                if sql:
                    final_sql = sql
                turn_images, turn_chart_type = self.generate_chart(result_df, effective_query, q_id, turn_idx, original_query=current_q)
                if turn_images:
                    chart_type = turn_chart_type
                nl_answer = self.generate_natural_answer(effective_query, sql, result_df, turn_chart_type) if hasattr(self, 'generate_natural_answer') else f"SQL执行成功，共 {len(result_df)} 条记录。"
                dialogue.append({"Q": current_q, "A": {"content": nl_answer, "image": turn_images}})
                print(f"  SQL: {sql}")
                print(f"  改写后问题: {effective_query}")
                print(f"  回答: {nl_answer[:80]}...")
                q_ptr += 1
            question_col = json.dumps([{"Q": t} for t in turns], ensure_ascii=False)
            answer_col = json.dumps(dialogue, ensure_ascii=False)
            results.append({"编号": q_id, "问题": question_col, "SQL查询语句": final_sql, "图形格式": chart_type, "回答": answer_col})
        result_df = pd.DataFrame(results)
        result_df.to_excel(output_file, index=False)
        log.info(f"✅ 批量处理完成！结果已保存至 {output_file}")
        print(f"\n共处理 {len(results)} 个问题，结果文件：{output_file}")
    def generate_natural_answer(self, question: str, sql: str, df: pd.DataFrame, chart_type: str = "无") -> str:
        if df is None or df.empty:
            return "未查询到相关数据。"
        if "error" in df.columns:
            return f"查询执行出错：{df['error'].iloc[0]}"
        result_text = df.head(15).to_string(index=False)
        chart_hint = f"本轮已生成{chart_type}。" if chart_type != "无" else ""
        messages = [
            {"role": "system", "content": (
                "你是一位专业的财务分析师。根据用户问题和SQL查询结果，"
                "用简洁、专业、自然的中文给出回答。"
                "直接叙述数据结论，不要重复问题，不要解释SQL。"
                "如果有多行数据，可以简要列出关键数值并给出简短分析。"
            )},
            {"role": "user", "content": (
                f"用户问题：{question}\n\n"
                f"查询结果：\n{result_text}\n\n"
                f"补充信息：{chart_hint}\n\n"
                "请根据以上数据给出简洁的中文回答："
            )},
        ]
        try:
            answer = self.llm.chat(messages, temperature=0.3, max_tokens=400)
            return answer
        except Exception as e:
            log.warning(f"生成自然语言回答失败: {e}")
            return result_text
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["merge", "infer", "batch2"])
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--llm-api-key", default=os.getenv("DEEPSEEK_API_KEY", ""))
    args = parser.parse_args()
    if args.mode == "infer":
        agent = SQLGPTInference(data_dir=args.data_dir, top_k=args.top_k, llm_api_key=args.llm_api_key)
        print("\nSQLGPT 已就绪（多轮对话）\n输入 'exit' 退出，'reset' 清空历史\n")
        while True:
            try:
                q = input("💬 查询: ").strip()
            except Exception:
                break
            if q.lower() in ("exit", "quit"):
                break
            if q.lower() == "reset":
                agent.reset_history()
                continue
            out = agent.query(q)
            if out.get("clarify"):
                print(f"系统追问：{out['result']['message'].iloc[0]}\n")
                continue
            print(f"\nSQL：{out['sql']}")
            print(f"改写后问题：{out.get('effective_query', q)}")
            print(f"结果（前5行）：\n{out['result'].head().to_string(index=False)}\n")
    elif args.mode == "batch2":
        agent = SQLGPTInference(data_dir=args.data_dir, top_k=args.top_k, llm_api_key=args.llm_api_key)
        agent.batch_process(questions_file="data/附件4：问题汇总.xlsx", output_file="result_2.xlsx")
if __name__ == "__main__":
    main()

# 运行方式python question2.py --mode batch2 --data-dir ./data --llm-api-key "sk-90abab83f64c489f955fac6d1b4465bd"