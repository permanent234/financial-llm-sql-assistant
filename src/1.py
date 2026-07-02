%%writefile
question2.py
"""
SQLGPT - 严格按照附件3字段说明优化版
已修复：RAG 选表错误、意图澄清过度、修复模块过度复杂化
"""

import os
import re
import glob
import json
import sqlite3
import logging
import argparse
from typing import List, Dict, Tuple, Optional

import pandas as pd
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from sentence_transformers import SentenceTransformer, util
from openai import OpenAI

rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

TABLE_NAMES = [
    "core_performance_indicators_sheet",
    "balance_sheet",
    "income_sheet",
    "cash_flow_sheet",
]


# ══════════════════════════════════════════════════════════════════════════════
# 0. 合并 CSV
# ══════════════════════════════════════════════════════════════════════════════

def merge_csvs(data_dir: str):
    for table in TABLE_NAMES:
        pattern = os.path.join(data_dir, f"*_{table}.csv")
        files = glob.glob(pattern)
        global_csv = os.path.join(data_dir, f"{table}.csv")

        if not files:
            if os.path.exists(global_csv):
                log.info(f"[merge] {table} 全局表已存在，跳过")
            continue

        dfs = [pd.read_csv(f, encoding="utf-8-sig") for f in files]
        merged = pd.concat(dfs, ignore_index=True)
        if "stock_code" in merged.columns and "report_period" in merged.columns:
            merged = merged.drop_duplicates(subset=["stock_code", "report_period"], keep="last")

        merged.to_csv(global_csv, index=False, encoding="utf-8-sig")
        log.info(f"[merge] {table}：{len(merged)} 行")


# ══════════════════════════════════════════════════════════════════════════════
# 1. DeepSeek Client
# ══════════════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════════════
# 2. SQLGPTInference 主类（核心修复）
# ══════════════════════════════════════════════════════════════════════════════

class SQLGPTInference:
    def __init__(self, data_dir: str, top_k: int = 2, llm_api_key: Optional[str] = None):
        self.data_dir = data_dir
        self.db_path = os.path.join(data_dir, "financial_reports.db")
        self.top_k = top_k
        self.history: List[Tuple[str, str]] = []

        os.makedirs("./result", exist_ok=True)

        self._build_sqlite_db()
        self.table_schemas: Dict[str, str] = self._build_schemas_from_csv()  # 严格按附件3生成
        self.company_list = self._get_company_list()

        self.embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        self.schema_embeddings = self.embedder.encode(
            list(self.table_schemas.values()), convert_to_tensor=True, show_progress_bar=False
        )

        self.examples = self._load_financial_examples()
        self.llm = DeepSeekClient(llm_api_key)

        log.info("SQLGPT 初始化完成（已严格按照附件3字段说明优化）")

    def _build_schemas_from_csv(self) -> Dict[str, str]:
        """严格按照附件3字段说明构建每个表的完整schema描述（提升RAG准确率）"""
        schemas = {}

        # 核心业绩指标表
        schemas["core_performance_indicators_sheet"] = """
核心业绩指标表 (core_performance_indicators_sheet)
字段说明：
- stock_code: 股票代码 (str)
- stock_abbr: 股票简称 (str)
- eps: 每股收益 (decimal)
- roe: 净资产收益率 (decimal)
- report_period: 报告期 (如 'FY', 'Q3')
- report_year: 报告年份 (int, 如 2024)
适合查询：每股收益、ROE、核心指标等。
"""

        # 资产负债表
        schemas["balance_sheet"] = """
资产负债表 (balance_sheet)
字段说明：
- stock_code, stock_abbr
- asset_total_assets: 总资产
- liability_total_liabilities: 总负债
- asset_liability_ratio: 资产负债率
- equity_total_equity: 所有者权益
- report_period, report_year
适合查询：总资产、资产负债率、总负债等。
"""

        # 利润表
        schemas["income_sheet"] = """
利润表 (income_sheet)
字段说明：
- stock_code, stock_abbr
- total_operating_revenue: 营业总收入
- net_profit: 净利润
- total_profit: 利润总额
- operating_profit: 营业利润
- report_period, report_year
适合查询：净利润、利润总额、营业收入等。
"""

        # 现金流量表
        schemas["cash_flow_sheet"] = """
现金流量表 (cash_flow_sheet)
字段说明：
- stock_code, stock_abbr
- operating_cf_net_amount: 经营活动现金流量净额
- report_period, report_year
适合查询：经营活动现金流净额、现金流相关指标。
"""

        return schemas

    def _get_company_list(self) -> List[str]:
        """从所有表格中提取股票简称列表，用于意图澄清时的公司名检测"""
        company_set = set()

        try:
            for table_name in ["core_performance_indicators_sheet", "balance_sheet",
                               "income_sheet", "cash_flow_sheet"]:
                csv_path = os.path.join(self.data_dir, f"{table_name}.csv")
                if os.path.exists(csv_path):
                    df = pd.read_csv(csv_path, encoding="utf-8-sig")
                    if "stock_abbr" in df.columns:
                        company_set.update(df["stock_abbr"].dropna().unique().tolist())
        except Exception as e:
            log.warning(f"加载公司列表失败: {e}，使用默认列表")

        # 默认备选公司（防止CSV加载失败）
        default_companies = [
            "金花股份", "华润三九", "片仔癀", "同仁堂", "太极集团",
            "中恒集团", "达仁堂", "羚锐制药", "济川药业", "昆药集团"
        ]

        company_list = sorted(list(company_set)) if company_set else default_companies
        log.info(f"共加载 {len(company_list)} 家上市公司简称")
        return company_list

    def _load_financial_examples(self) -> List[Tuple[str, str]]:
        """加载领域特定的 few-shot 示例（强烈提升SQL生成准确率）"""
        return [
            ("金花股份2022年的每股收益是多少",
             "SELECT eps FROM core_performance_indicators_sheet WHERE stock_abbr = '金花股份' AND report_period = 'FY' AND report_year = 2022"),

            ("2024年利润总额最高的前10家公司是哪些",
             "SELECT stock_abbr, total_profit FROM income_sheet WHERE report_period = 'FY' AND report_year = 2024 ORDER BY total_profit DESC LIMIT 10"),

            ("华润三九2023年的净利润是多少",
             "SELECT net_profit FROM income_sheet WHERE stock_abbr = '华润三九' AND report_period = 'FY' AND report_year = 2023"),

            ("金花股份2022年的资产负债率是多少",
             "SELECT asset_liability_ratio FROM balance_sheet WHERE stock_abbr = '金花股份' AND report_period = 'FY' AND report_year = 2022"),

            ("2024年上半年总资产最大的三家公司",
             "SELECT stock_abbr, asset_total_assets FROM balance_sheet WHERE report_period = 'HY' AND report_year = 2024 ORDER BY asset_total_assets DESC LIMIT 3"),

            ("金花股份近三年的经营活动现金流净额变化趋势是什么",
             "SELECT report_period, operating_cf_net_amount FROM cash_flow_sheet WHERE stock_abbr = '金花股份' AND report_year IN (2022,2023,2024) ORDER BY report_year"),
        ]

    def _build_sqlite_db(self):
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        for table_name in TABLE_NAMES:
            csv_path = os.path.join(self.data_dir, f"{table_name}.csv")
            if not os.path.exists(csv_path):
                continue
            df = pd.read_csv(csv_path, encoding="utf-8-sig")
            df.to_sql(table_name, conn, if_exists="replace", index=False)
            log.info(f"[DB] {table_name} 已入库：{len(df)} 行")
        conn.close()

    def clarify_intent_if_needed(self, query: str) -> Optional[str]:
        """改进版：更智能的动态意图澄清 + 支持多轮意图记忆准备"""
        query_lower = query.lower().strip()

        # ==================== 1. 全局/排名类查询（不需要追问公司） ====================
        global_keywords = [
            "最高", "最低", "前", "排名", "最多", "最少", "最大", "最小",
            "所有公司", "哪家公司", "趋势", "变化", "近三年", "近两年",
            "近几年", "上半年", "下半年", "全年", "前5", "前10", "前三"
        ]
        if any(k in query_lower for k in global_keywords):
            return None

        # ==================== 2. 检测是否包含具体指标 ====================
        indicator_keywords = [
            "收益", "利润", "收入", "资产负债率", "eps", "roe",
            "净利润", "每股", "总资产", "总负债", "现金流",
            "毛利率", "净利率", "经营活动现金流", "每股收益"
        ]
        has_indicator = any(k in query_lower for k in indicator_keywords)

        # ==================== 3. 检测是否已有公司和年份 ====================
        has_company = any(company in query for company in self.company_list)
        has_year = bool(re.search(r'20\d{2}', query))  # 检测 202x 年份

        # ==================== 4. 决策逻辑 ====================
        if has_indicator and not has_company:
            # 根据指标类型生成更自然的动态追问
            if any(x in query_lower for x in ["每股收益", "eps"]):
                return "请问你要查询哪一家公司的每股收益？（例如：金花股份、华润三九）"
            elif any(x in query_lower for x in ["净利润", "利润"]):
                return "请问你要查询哪一家公司的净利润？（例如：金花股份、华润三九）"
            elif "资产负债率" in query_lower:
                return "请问你要查询哪一家公司的资产负债率？（例如：金花股份、华润三九）"
            elif "总资产" in query_lower:
                return "请问你要查询哪一家公司的总资产？（例如：金花股份、华润三九）"
            elif "现金流" in query_lower:
                return "请问你要查询哪一家公司的现金流数据？（例如：金花股份、华润三九）"
            else:
                return "请问你要查询哪一家公司的财务数据？（例如：金花股份、华润三九）"

        # 如果已有公司但缺少年份，且查询明显需要具体年份，可增加第二轮追问（可选）
        if has_company and not has_year and has_indicator and "年" in query_lower:
            return "请问你要查询哪一年的数据？（例如：2024年、2023年）"

        return None

    # ========================== RAG ==========================
    def retrieve_schemas(self, query: str) -> List[str]:
        query_emb = self.embedder.encode(query, convert_to_tensor=True)
        cos_scores = util.cos_sim(query_emb, self.schema_embeddings)[0]
        k = min(self.top_k, len(self.table_schemas))
        top_idx = torch.topk(cos_scores, k=k).indices.tolist()
        retrieved = [list(self.table_schemas.keys())[i] for i in top_idx]
        log.info(f"RAG 检索到表（top-{k}）: {retrieved}")
        return retrieved

    # ========================== Prompt ==========================
    def build_prompt(self, query: str, retrieved_tables: List[str]) -> str:
        p_base = (
            "你是一位精通财务报表的 SQL 专家。\n"
            "重要规则：\n"
            "1. 只输出一条 SQL 语句，不要任何解释。\n"
            "2. report_period 字段的值是 'FY' 表示全年，'Q1' 表示一季度，'HY' 表示半年，'Q3' 表示三季度。\n"
            "3. 如果用户提到具体年份（如 2022年），必须同时使用 report_period 和 report_year 两个字段。\n"
            "   示例：2022年 → report_period = 'FY' AND report_year = 2022\n"
            "4. 如果查询中没有明确的公司名称，且查询的是具体财务指标（如净利润、每股收益），请先生成追问，而不是随意选择一家公司。\n"
            "5. 严格使用 Schema 中真实存在的字段名。\n"
        )

        p_schema = "【数据库 Schema】\n" + "\n\n".join(
            self.table_schemas[t] for t in retrieved_tables
        )

        p_example = (
            "【示例】\n"
            "查询: 金花股份2022年的每股收益是多少\n"
            "SQL: SELECT eps FROM core_performance_indicators_sheet WHERE stock_abbr = '金花股份' AND report_period = '2022FY'\n\n"
            "查询: 2024年利润总额最高的前10家公司是哪些\n"
            "SQL: SELECT stock_abbr, total_profit FROM income_sheet WHERE report_period = '2024FY' ORDER BY total_profit DESC LIMIT 10\n"
        )

        history_str = ""
        if self.history:
            history_str = "【历史对话】\n" + "\n".join(f"用户: {h_q}\nSQL: {h_s}" for h_q, h_s in self.history[-3:])

        p_query = f"【当前查询】\n{query}\n\n请直接输出 SQL："

        return "\n\n".join(filter(None, [p_base, p_schema, p_example, history_str, p_query]))

    # ========================== SQL 生成 ==========================
    def generate_sql(self, prompt: str) -> str:
        messages = [
            {"role": "system", "content": "你是一个 SQL 生成助手，只输出一条 SQL 语句，不要任何解释。"},
            {"role": "user", "content": prompt}
        ]
        raw = self.llm.chat(messages, temperature=0.0, max_tokens=512)
        sql = self._extract_sql(raw)
        sql = self.correct_sql_columns(sql)
        log.info(f"生成 SQL: {sql}")
        return sql

    def correct_sql_columns(self, sql: str) -> str:
        """列名修正 + 报告期精准转换（同时使用 report_period 和 report_year）"""
        column_map = {
            "eps_basic": "eps",
            "operating_revenue": "total_operating_revenue",
            "net_profit": "net_profit",
            "net_profit_10k_yuan": "net_profit",
        }
        for wrong, correct in column_map.items():
            sql = re.sub(rf"\b{wrong}\b", correct, sql, flags=re.IGNORECASE)

        # ==================== 报告期精准转换 ====================
        # 2022年 → report_period = 'FY' AND report_year = 2022
        sql = re.sub(
            r"report_period\s*=\s*'(\d{4})FY'",
            r"report_period = 'FY' AND report_year = \1",
            sql, flags=re.IGNORECASE
        )

        # 2023Q3 → report_period = 'Q3' AND report_year = 2023
        sql = re.sub(
            r"report_period\s*=\s*'(\d{4})Q([1-3])'",
            r"report_period = 'Q\2' AND report_year = \1",
            sql, flags=re.IGNORECASE
        )

        # 2023HY → report_period = 'HY' AND report_year = 2023
        sql = re.sub(
            r"report_period\s*=\s*'(\d{4})HY'",
            r"report_period = 'HY' AND report_year = \1",
            sql, flags=re.IGNORECASE
        )

        # 处理 IN 查询（近三年等）
        sql = re.sub(
            r"report_period\s*IN\s*\(\s*'(\d{4})FY'(?:\s*,\s*'(\d{4})FY')*\s*\)",
            lambda m: "report_period = 'FY' AND report_year IN (" +
                      ",".join(m.groups()[:2]) + ")",
            sql, flags=re.IGNORECASE
        )

        # 资产负债率必须从 balance_sheet 取
        if "asset_liability_ratio" in sql.lower():
            sql = re.sub(r"FROM\s+\w+", "FROM balance_sheet", sql, flags=re.IGNORECASE)

        # 每股收益必须从 core_performance_indicators_sheet 取
        if "eps" in sql.lower() and "core_performance_indicators_sheet" not in sql.lower():
            sql = re.sub(r"FROM\s+\w+", "FROM core_performance_indicators_sheet", sql, flags=re.IGNORECASE)

        return sql

    @staticmethod
    def _extract_sql(text: str) -> str:
        text = text.strip()
        text = re.sub(r"^```sql\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"^```\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        m = re.search(r"(SELECT\b.*?)(?:;?\s*$)", text, flags=re.IGNORECASE | re.DOTALL)
        if m:
            sql = m.group(1).strip()
            return re.sub(r"\s+", " ", sql).rstrip(";")
        return text.rstrip(";")

    # ========================== SQL 修复（加强限制） ==========================
    def repair_sql_with_llm(self, query: str, bad_sql: str, error_msg: str, retrieved_tables: List[str]) -> str:
        schema_str = "\n\n".join(self.table_schemas[t] for t in retrieved_tables)

        messages = [
            {"role": "system",
             "content": "你是一个谨慎的 SQL 修复助手。只做最小必要修改，优先使用单表查询，绝不要自行计算或加入 JOIN。除非用户明确要求，否则不要引入新计算公式。"},
            {"role": "user", "content": f"""
【数据库 Schema】
{schema_str}

【用户问题】
{query}

【错误 SQL】
{bad_sql}

【报错信息】
{error_msg}

请只进行最小化修复，只输出一条修正后的 SQL：
"""}
        ]

        fixed_sql = self.llm.chat(messages, temperature=0.0, max_tokens=512)
        fixed_sql = self._extract_sql(fixed_sql)
        fixed_sql = self.correct_sql_columns(fixed_sql)
        log.info(f"修复后 SQL: {fixed_sql}")
        return fixed_sql

    def validate_and_fix_sql(self, sql: str, retrieved_tables: List[str]) -> str:
        sql_lower = sql.lower()
        for table in retrieved_tables:
            if table.lower() in sql_lower:
                return sql
        log.warning("SQL 校验失败，触发修复...")
        return self.repair_sql_with_llm("", sql, "列名或表名可能不存在", retrieved_tables)

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
        """完整查询流程 —— 优化多轮意图澄清 + 上下文记忆合并"""
        original_query = natural_language.strip()

        # Step 1: 先判断是否需要意图澄清（使用你最新的 clarify_intent_if_needed）
        clarify_msg = self.clarify_intent_if_needed(original_query)

        # Step 2: 处理多轮澄清 —— 如果上一轮是系统追问，且本次输入较短，则认为是补充信息
        if (self.history and
                len(self.history) >= 2 and
                "请问你要查询哪一" in str(self.history[-1][1]) and  # 上一轮是澄清消息
                len(original_query) < 40):  # 本轮输入较短（补充公司/年份）

            # 合并上一次的原始查询 + 本次补充的内容
            last_original = self.history[-2][0] if len(self.history) >= 2 else ""
            if last_original:
                natural_language = last_original + " " + original_query
                log.info(f"[多轮意图合并] 合并后查询: {natural_language}")
            else:
                natural_language = original_query
        else:
            natural_language = original_query

        # Step 3: 如果需要澄清，直接返回澄清消息
        if clarify_msg:
            self.history.append((original_query, clarify_msg))  # 保存原始查询和澄清消息
            return {
                "query": original_query,
                "sql": "",
                "retrieved": [],
                "result": pd.DataFrame({"message": [clarify_msg]}),
                "clarify": True,
                "original_query": original_query
            }

        # Step 4: 正常流程（无需澄清）
        retrieved = self.retrieve_schemas(natural_language)
        prompt = self.build_prompt(natural_language, retrieved)
        sql = self.generate_sql(prompt)

        # 修复：先修正列名和报告期，再执行
        sql = self.correct_sql_columns(sql)  # 假设你已有此方法
        result = self.execute_query(sql)

        # 保存对话历史（使用合并后的 natural_language）
        self.history.append((natural_language, sql))

        return {
            "query": original_query,  # 返回给用户看到的原始输入
            "sql": sql,
            "retrieved": retrieved,
            "result": result,
            "clarify": False
        }

    def reset_history(self):
        self.history.clear()

    def generate_chart(
            self, df: pd.DataFrame, query: str, question_id: str, turn_idx: int = 1) -> Tuple[List[str], str]:
        """自动生成图表（支持折线图、柱状图、饼图）"""
        if df is None or df.empty or "error" in df.columns:
            return [], "无"

        save_dir = "./result"
        os.makedirs(save_dir, exist_ok=True)

        images = []
        chart_type = "无"
        title = query[:50] + ("..." if len(query) > 50 else "")

        # 提取数值列和字符串列
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        str_cols = df.select_dtypes(exclude="number").columns.tolist()

        img_path = f"{save_dir}/{question_id}_{turn_idx}.jpg"
        img_rel = f"./result/{question_id}_{turn_idx}.jpg"

        # ==================== 1. 时间序列折线图（最常用） ====================
        if "report_period" in df.columns and numeric_cols and len(df) >= 2:
            try:
                fig, ax = plt.subplots(figsize=(11, 6))
                for col in numeric_cols[:3]:  # 最多画3条线
                    ax.plot(df["report_period"].astype(str), df[col],
                            marker="o", linewidth=2, label=col)

                ax.set_title(title, fontsize=14, pad=20)
                ax.set_xlabel("报告期", fontsize=12)
                ax.set_ylabel("数值", fontsize=12)
                ax.legend(fontsize=10)
                ax.grid(True, alpha=0.3)
                plt.xticks(rotation=45, ha='right')
                fig.tight_layout()

                fig.savefig(img_path, dpi=160, bbox_inches='tight')
                plt.close(fig)

                images.append(img_rel)
                chart_type = "折线图（趋势分析）"
                return images, chart_type
            except Exception as e:
                log.warning(f"折线图生成失败: {e}")

        # ==================== 2. 柱状图（排名、对比） ====================
        if len(str_cols) >= 1 and len(numeric_cols) >= 1 and 2 <= len(df) <= 15:
            try:
                fig, ax = plt.subplots(figsize=(12, 7))
                x_labels = df[str_cols[0]].astype(str)
                ax.bar(x_labels, df[numeric_cols[0]], color='#4a86e8')

                ax.set_title(title, fontsize=14, pad=20)
                ax.set_xlabel(str_cols[0], fontsize=12)
                ax.set_ylabel(numeric_cols[0], fontsize=12)
                plt.xticks(rotation=45, ha='right')
                ax.grid(axis='y', alpha=0.3)
                fig.tight_layout()

                fig.savefig(img_path, dpi=160, bbox_inches='tight')
                plt.close(fig)

                images.append(img_rel)
                chart_type = "柱状图（排名/对比）"
                return images, chart_type
            except Exception as e:
                log.warning(f"柱状图生成失败: {e}")

        # ==================== 3. 饼图（占比类，数据量少） ====================
        if len(numeric_cols) >= 1 and len(str_cols) >= 1 and 2 <= len(df) <= 8:
            try:
                fig, ax = plt.subplots(figsize=(9, 9))
                values = df[numeric_cols[0]].abs()  # 防止负数影响饼图
                labels = df[str_cols[0]].astype(str)

                ax.pie(values, labels=labels, autopct='%1.1f%%', startangle=90,
                       textprops={'fontsize': 10})
                ax.set_title(title, fontsize=14, pad=20)

                fig.savefig(img_path, dpi=160, bbox_inches='tight')
                plt.close(fig)

                images.append(img_rel)
                chart_type = "饼图（占比分析）"
                return images, chart_type
            except Exception as e:
                log.warning(f"饼图生成失败: {e}")

        # 如果以上都没生成，返回无图表
        return images, chart_type


# ══════════════════════════════════════════════════════════════════════════════
# 主入口
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["merge", "infer", "batch2"])
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--llm-api-key", default=os.getenv("DEEPSEEK_API_KEY", ""))
    args = parser.parse_args()

    if args.mode == "merge":
        merge_csvs(args.data_dir)
    elif args.mode == "infer":
        agent = SQLGPTInference(data_dir=args.data_dir, top_k=args.top_k, llm_api_key=args.llm_api_key)
        print("\nSQLGPT 已就绪（多轮对话）\n输入 'exit' 退出，'reset' 清空历史\n")
        while True:
            try:
                q = input("💬 查询: ").strip()
            except:
                break
            if q.lower() in ("exit", "quit"): break
            if q.lower() == "reset":
                agent.reset_history()
                continue
            out = agent.query(q)
            if out.get("clarify"):
                print(f"系统追问：{out['result']['message'].iloc[0]}\n")
                continue
            print(f"\nSQL：{out['sql']}")
            print(f"结果（前5行）：\n{out['result'].head().to_string(index=False)}\n")
    elif args.mode == "batch2":
        print("batch2 模式暂未完整实现，可后续补充")


if __name__ == "__main__":
    main()