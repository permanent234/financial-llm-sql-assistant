"""
SQLGPT 最终完整版（任务二专用）- 8GB RAM 低配适配版
已针对你的电脑（8GB RAM + i5-8250U + 940MX 2GB）进行极致优化
"""

import os
import re
import glob
import json
import sqlite3
import logging
import argparse
from typing import List, Dict, Tuple

import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

from sentence_transformers import SentenceTransformer, util
from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

AGG_OPS  = ["", "MAX", "MIN", "COUNT", "SUM", "AVG"]
COND_OPS = ["=", ">", "<"]

TABLE_NAMES = [
    "core_performance_indicators_sheet",
    "balance_sheet",
    "income_sheet",
    "cash_flow_sheet",
]


# ====================== 0. 合并 CSV ======================
def merge_csvs(data_dir: str):
    for table in TABLE_NAMES:
        pattern = os.path.join(data_dir, f"*_{table}.csv")
        files = glob.glob(pattern)
        global_csv = os.path.join(data_dir, f"{table}.csv")

        if not files:
            if os.path.exists(global_csv):
                log.info(f"[merge] {table} 全局表已存在，跳过")
            else:
                log.warning(f"[merge] 未找到 {table} 的 CSV 文件")
            continue

        dfs = [pd.read_csv(f, encoding="utf-8-sig") for f in files]
        merged = pd.concat(dfs, ignore_index=True)
        if "stock_code" in merged.columns and "report_period" in merged.columns:
            merged = merged.drop_duplicates(subset=["stock_code", "report_period"], keep="last")

        merged.to_csv(global_csv, index=False, encoding="utf-8-sig")
        log.info(f"[merge] {table}：合并 {len(files)} 个文件 → {len(merged)} 行")


# ====================== 1. WikiSQL 预处理 ======================
def build_sql_from_wikisql(sql_dict: dict, header: list) -> str:
    # （你的原逻辑，保持不变）
    agg_idx = sql_dict.get("agg", 0)
    agg_str = AGG_OPS[agg_idx] if 0 <= agg_idx < len(AGG_OPS) else ""
    col_idx = sql_dict.get("sel", sql_dict.get("columns", [0])[0] if sql_dict.get("columns") else 0)
    col_name = header[col_idx] if isinstance(header, (list, tuple)) and col_idx < len(header) else f"col{col_idx}"
    select_clause = f"SELECT {agg_str}({col_name})" if agg_str else f"SELECT {col_name}"
    sql = f"{select_clause} FROM table"

    conds_raw = sql_dict.get("conds", sql_dict.get("conditions", []))
    if isinstance(conds_raw, dict):
        col_idxs = conds_raw.get("column_index", [])
        op_idxs = conds_raw.get("operator_index", [])
        values = conds_raw.get("condition", [])
        conds_raw = list(zip(col_idxs, op_idxs, values))

    where_parts = []
    for cond in conds_raw:
        c_col_idx, c_op_idx, c_val = cond[0], cond[1], cond[2]
        c_col = header[c_col_idx] if isinstance(header, (list, tuple)) and c_col_idx < len(header) else f"col{c_col_idx}"
        c_op = COND_OPS[c_op_idx] if 0 <= c_op_idx < len(COND_OPS) else "="
        try:
            float(str(c_val))
            c_val_str = str(c_val)
        except ValueError:
            c_val_str = f"'{c_val}'"
        where_parts.append(f"{c_col} {c_op} {c_val_str}")

    if where_parts:
        sql += " WHERE " + " AND ".join(where_parts)
    return sql


def preprocess_wikisql_for_training(example: dict) -> dict:
    table_info = example.get("table", {})
    if isinstance(table_info, str):
        try:
            table_info = json.loads(table_info)
        except:
            table_info = {}

    header = table_info.get("header", [])
    page_title = table_info.get("page_title", table_info.get("name", "unknown_table"))
    types = table_info.get("types", ["text"] * len(header))
    question = example.get("question", "")
    sql_dict = example.get("sql", {})

    sql_target = build_sql_from_wikisql(sql_dict, header)

    schema_lines = [f"表名: {page_title}"]
    for i, col in enumerate(header):
        dtype = types[i] if i < len(types) else "text"
        schema_lines.append(f"- {col} ({dtype})")
    schema_str = "\n".join(schema_lines)

    prompt = (
        "你是一位精通数据库语言和开发的深度数据库工程师，"
        "请根据数据库信息和下面的自然语言需求写出对应的SQL查询语句。\n\n"
        f"数据库模式：\n{schema_str}\n\n"
        f"自然语言查询：{question}\n\n"
        "请直接输出对应的SQL语句（只输出SQL，不要任何解释）：\n"
    )
    return {"text": prompt + sql_target}


# ====================== 2. 训练阶段（8GB RAM 极致优化） ======================
def train_on_wikisql(output_dir: str, base_model: str = "Qwen/Qwen2-1.5B-Instruct"):
    from transformers import (
        AutoModelForCausalLM, AutoTokenizer,
        TrainingArguments, Trainer,
        DataCollatorForLanguageModeling,
    )
    from peft import LoraConfig, get_peft_model, TaskType

    log.info("加载 WikiSQL 数据集（htriedman/wikisql）...")
    dataset = load_dataset("htriedman/wikisql", split="train[:5%]")   # 进一步缩小数据集，适合 8GB 内存

    log.info(f"数据集加载完成，共 {len(dataset)} 条样本")

    dataset = dataset.map(preprocess_wikisql_for_training, remove_columns=dataset.column_names)

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize(example):
        enc = tokenizer(example["text"], truncation=True, max_length=128, padding="max_length")  # 极致压缩
        enc["labels"] = enc["input_ids"].copy()
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"], "labels": enc["labels"]}

    tokenized_dataset = dataset.map(tokenize, remove_columns=["text"])
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    # 8GB 内存关键设置
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float32,          # CPU 模式下更稳定
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )

    model.gradient_checkpointing_enable()   # 关键：节省激活值内存 ≈60%
    model.config.use_cache = False

    # LoRA 配置
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,                                # 进一步缩小 LoRA 秩，节省内存
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=target_modules,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=1,           # 必须为 1
        gradient_accumulation_steps=16,          # 等效 batch=16，但内存最低
        learning_rate=2e-4,
        fp16=False,                              # CPU 不支持
        logging_steps=20,                        # 更频繁打印，方便观察进度
        save_strategy="epoch",
        eval_strategy="no",
        report_to="none",
        dataloader_pin_memory=False,
        dataloader_num_workers=0,
        optim="adafactor",                       # 省内存优化器
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    log.info("开始 LoRA 微调（已针对 8GB RAM 优化，CPU 模式，速度较慢，请耐心等待）")
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    log.info(f"微调完成！模型保存至 {output_dir}")


# ====================== 3. 推理引擎（保持你的原代码 + 小幅优化） ======================
class SQLGPTInference:

    def __init__(
        self,
        data_dir:   str,
        model_path: str,
        base_model: str = "Qwen/Qwen2-1.5B-Instruct",
        top_k:      int = 2,
    ):
        self.data_dir = data_dir
        self.db_path  = os.path.join(data_dir, "financial_reports.db")
        self.top_k    = top_k
        self.history:  List[Tuple[str, str]] = []

        os.makedirs("./result", exist_ok=True)

        self._build_sqlite_db()
        self.table_schemas: Dict[str, str] = self._build_schemas_from_csv()

        log.info("加载 SBERT 编码器...")
        self.embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        self.schema_embeddings = self.embedder.encode(
            list(self.table_schemas.values()), convert_to_tensor=True, show_progress_bar=False
        )
        self.examples = self._load_financial_examples()
        self._load_model(model_path, base_model)
        log.info("SQLGPT 初始化完成")

    def _build_sqlite_db(self):
        conn   = sqlite3.connect(self.db_path)
        loaded = 0
        for table_name in TABLE_NAMES:
            csv_path = os.path.join(self.data_dir, f"{table_name}.csv")
            if not os.path.exists(csv_path):
                log.warning(f"[DB] {csv_path} 不存在，请先运行 --mode merge")
                continue
            df = pd.read_csv(csv_path, encoding="utf-8-sig")
            df.to_sql(table_name, conn, if_exists="replace", index=False)
            log.info(f"[DB] {table_name}：{len(df)} 行已入库")
            loaded += 1
        conn.close()
        if loaded == 0:
            raise FileNotFoundError("未找到任何全局 CSV，请先运行 --mode merge")

    def _build_schemas_from_csv(self) -> Dict[str, str]:
        cn_names = {
            "core_performance_indicators_sheet": "核心业绩指标表",
            "balance_sheet":   "资产负债表",
            "income_sheet":    "利润表",
            "cash_flow_sheet": "现金流量表",
        }
        schemas = {}
        for table_name in TABLE_NAMES:
            csv_path = os.path.join(self.data_dir, f"{table_name}.csv")
            if not os.path.exists(csv_path):
                schemas[table_name] = f"表名: {table_name}\n字段: (文件不存在)"
                continue
            df   = pd.read_csv(csv_path, encoding="utf-8-sig", nrows=0)
            cols = ", ".join(df.columns.tolist())
            cn   = cn_names.get(table_name, table_name)
            schemas[table_name] = f"表名: {table_name}（{cn}）\n字段: {cols}"
        return schemas

    def _load_financial_examples(self) -> List[Tuple[str, str]]:
        return [
            (
                "金花股份2022年的每股收益是多少",
                "SELECT eps_basic FROM core_performance_indicators_sheet "
                "WHERE stock_abbr = '金花股份' AND report_period = '2022FY'",
            ),
            (
                "查询所有公司2023财年的营业收入，按从高到低排列",
                "SELECT stock_abbr, operating_revenue FROM income_sheet "
                "WHERE report_period = '2023FY' ORDER BY operating_revenue DESC",
            ),
            (
                "华润三九近三年的主营业务收入",
                "SELECT report_period, operating_revenue FROM income_sheet "
                "WHERE stock_abbr = '华润三九' ORDER BY report_period",
            ),
            (
                "2024年利润总额最高的前10家公司",
                "SELECT stock_abbr, total_profit FROM income_sheet "
                "WHERE report_period = '2024FY' ORDER BY total_profit DESC LIMIT 10",
            ),
        ]

    def _load_model(self, model_path: str, base_model: str):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        log.info(f"加载基座模型：{base_model}")
        self.tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # [WARN-7修复] dtype 替代 torch_dtype
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map="auto",
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True,
        )
        log.info(f"叠加 LoRA adapter：{model_path}")
        self.model = PeftModel.from_pretrained(base, model_path)
        self.model.eval()

    def retrieve_schemas(self, query: str) -> List[str]:
        query_emb  = self.embedder.encode(query, convert_to_tensor=True)
        cos_scores = util.cos_sim(query_emb, self.schema_embeddings)[0]
        k          = min(self.top_k, len(self.table_schemas))
        top_idx    = torch.topk(cos_scores, k=k).indices.tolist()
        retrieved  = [list(self.table_schemas.keys())[i] for i in top_idx]
        log.info(f"RAG 检索到表（top-{k}）: {retrieved}")
        return retrieved

    def build_prompt(self, query: str, retrieved_tables: List[str]) -> str:
        p_base   = (
            "你是一位精通数据库语言和开发的深度数据库工程师，"
            "请根据数据库信息和下面的自然语言需求写出对应的SQL查询语句。"
        )
        p_schema  = "【数据库模式】\n" + "\n\n".join(
            self.table_schemas[t] for t in retrieved_tables
        )
        p_example = "【示例】\n" + "\n".join(
            f"查询: {q}\nSQL: {sql}" for q, sql in self.examples
        )
        history_str = ""
        if self.history:
            history_str = "【历史对话】\n" + "\n".join(
                f"用户: {h_q}\nSQL: {h_s}" for h_q, h_s in self.history[-3:]
            )
        p_query = f"【当前查询】\n{query}\n\n只输出SQL语句："
        return "\n\n".join(filter(None, [p_base, p_schema, p_example, history_str, p_query]))

    def generate_sql(self, prompt: str) -> str:
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=2048
        ).to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                repetition_penalty=1.1,
            )
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        raw        = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        sql        = self._extract_sql(raw)
        log.info(f"生成 SQL: {sql}")
        return sql

    @staticmethod
    def _extract_sql(text: str) -> str:
        """
        [BUG-5修复] 去掉 re.DOTALL，逐行优先匹配。
        原代码 DOTALL 会把 SELECT 之后所有换行内容（含模型解释文字）一并截入。
        """
        # 优先：逐行找 SELECT 开头的行（最精确）
        for line in text.splitlines():
            s = line.strip().rstrip(";")
            if s.upper().startswith("SELECT"):
                return s
        # 退而求其次：单行正则（不含 DOTALL）
        m = re.search(r"(SELECT\s+.+?)(?:;|\n|$)", text, re.IGNORECASE)
        if m:
            return m.group(1).strip().rstrip(";")
        return text.strip()

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
        retrieved = self.retrieve_schemas(natural_language)
        prompt    = self.build_prompt(natural_language, retrieved)
        sql       = self.generate_sql(prompt)
        result    = self.execute_query(sql)
        self.history.append((natural_language, sql))
        return {"query": natural_language, "sql": sql, "retrieved": retrieved, "result": result}

    def reset_history(self):
        self.history.clear()
        log.info("对话历史已清空")

    def generate_chart(
        self, df: pd.DataFrame, query: str, question_id: str, turn_idx: int = 1
    ) -> Tuple[List[str], str]:
        if df is None or df.empty or "error" in df.columns:
            return [], "无"

        save_dir     = "./result"
        images       = []
        chart_type   = "无"
        title        = query[:40] + ("..." if len(query) > 40 else "")
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        str_cols     = df.select_dtypes(exclude="number").columns.tolist()

        img_path = f"{save_dir}/{question_id}_{turn_idx}.jpg"
        img_rel  = f"./result/{question_id}_{turn_idx}.jpg"

        if "report_period" in df.columns and numeric_cols:
            fig, ax = plt.subplots(figsize=(10, 5))
            for col in numeric_cols[:2]:
                ax.plot(df["report_period"].astype(str), df[col], marker="o", label=col)
            ax.set_title(title); ax.set_xlabel("报告期"); ax.legend(); ax.grid(True, alpha=0.4)
            plt.xticks(rotation=45)
            fig.savefig(img_path, dpi=150, bbox_inches="tight"); plt.close(fig)
            images.append(img_rel); chart_type = "折线图"

        elif len(str_cols) >= 1 and len(numeric_cols) >= 1 and len(df) >= 2:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.bar(df[str_cols[0]].astype(str), df[numeric_cols[0]])
            ax.set_title(title); ax.set_xlabel(str_cols[0]); ax.set_ylabel(numeric_cols[0])
            plt.xticks(rotation=45)
            fig.savefig(img_path, dpi=150, bbox_inches="tight"); plt.close(fig)
            images.append(img_rel); chart_type = "柱状图"

        elif len(numeric_cols) >= 1 and len(str_cols) >= 1 and 2 <= len(df) <= 8:
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.pie(df[numeric_cols[0]].abs(), labels=df[str_cols[0]].astype(str), autopct="%1.1f%%")
            ax.set_title(title)
            fig.savefig(img_path, dpi=150, bbox_inches="tight"); plt.close(fig)
            images.append(img_rel); chart_type = "饼图"

        return images, chart_type
def _make_conclusion(query: str, df: pd.DataFrame, sql: str) -> str:
    if df is None or df.empty or "error" in df.columns:
        err = df["error"].iloc[0] if df is not None and "error" in df.columns else "无结果"
        return f"查询执行失败：{err}\nSQL：{sql}"
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    lines = [f"共查询到 {len(df)} 条记录。"]
    if numeric_cols:
        col = numeric_cols[0]
        lines.append(
            f"{col} 最大值：{df[col].max():,.2f}，"
            f"最小值：{df[col].min():,.2f}，"
            f"平均值：{df[col].mean():,.2f}。"
        )
        if len(df) >= 2:
            trend = "上升" if df[col].iloc[-1] > df[col].iloc[0] else "下降"
            lines.append(f"整体趋势{trend}。")
    lines.append(f"\n使用 SQL：{sql}")
    return "\n".join(lines)
# ====================== 4. 主入口 ======================
def main():
    parser = argparse.ArgumentParser(description="SQLGPT 低配适配版")
    parser.add_argument("--mode", required=True, choices=["merge", "train", "infer", "batch2"])
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--output-dir", default="./sqlgpt_model")
    parser.add_argument("--model-path", default="./sqlgpt_model")
    parser.add_argument("--base-model", default="Qwen/Qwen2-1.5B-Instruct")
    parser.add_argument("--top-k", type=int, default=2)
    args = parser.parse_args()

    if args.mode == "merge":
        merge_csvs(args.data_dir)
    elif args.mode == "train":
        train_on_wikisql(args.output_dir, args.base_model)
    elif args.mode == "infer":
        agent = SQLGPTInference(
            data_dir=args.data_dir, model_path=args.model_path,
            base_model=args.base_model, top_k=args.top_k,
        )
        print("\nSQLGPT 已就绪（多轮对话）\n输入 'exit' 退出，'reset' 清空历史\n")
        while True:
            try:
                q = input("💬 查询: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not q: continue
            if q.lower() in ("exit", "quit"): break
            if q.lower() == "reset": agent.reset_history(); continue
            out = agent.query(q)
            print(f"\nSQL：{out['sql']}")
            print(f"结果（前5行）：\n{out['result'].head().to_string(index=False)}\n")
            pass
    elif args.mode == "batch2":
        log.info("=== 任务二：批量处理附件4 ===")
        agent = SQLGPTInference(
            data_dir=args.data_dir, model_path=args.model_path,
            base_model=args.base_model, top_k=args.top_k,
        )

        excel_candidates = [
            os.path.join(args.data_dir, "附件4：问题汇总.xlsx"),
            os.path.join(args.data_dir, "问题汇总.xlsx"),
            "附件4：问题汇总.xlsx",
        ]
        excel_path = next((p for p in excel_candidates if os.path.exists(p)), None)
        if excel_path is None:
            raise FileNotFoundError("找不到附件4，请确认 --data-dir 路径")
        questions_df = pd.read_excel(excel_path)

        result_rows = []
        for _, row in questions_df.iterrows():
            q_id = str(row.get("编号", row.name)).strip()
            q_raw = row.get("问题", "")
            try:
                q_list = json.loads(q_raw) if isinstance(q_raw, str) else [{"Q": str(q_raw)}]
            except Exception:
                q_list = [{"Q": str(q_raw)}]

            log.info(f"[{q_id}] 共 {len(q_list)} 轮对话")
            agent.reset_history()

            conversation = []
            final_sql = ""
            final_images = []
            final_chart_tp = "无"

            for turn_idx, turn in enumerate(q_list, 1):
                q_text = turn.get("Q", str(turn))
                print(f"  [{q_id}] 轮{turn_idx}: {q_text}")
                out = agent.query(q_text)
                final_sql = out["sql"]
                images, chart_type = agent.generate_chart(out["result"], q_text, q_id, turn_idx)
                if images:
                    final_images.extend(images);
                    final_chart_tp = chart_type
                conclusion = _make_conclusion(q_text, out["result"], final_sql)
                answer_block = {"content": conclusion}
                if images:
                    answer_block["image"] = images
                conversation.append({"Q": q_text, "A": answer_block})

            result_rows.append({
                "编号": q_id,
                "问题": json.dumps(q_list, ensure_ascii=False),
                "SQL查询语句": final_sql,
                "图形格式": final_chart_tp,
                "回答": json.dumps(conversation, ensure_ascii=False),
            })

        pd.DataFrame(result_rows).to_excel("result_2.xlsx", index=False)
        log.info(f"✅ 任务二完成！result_2.xlsx 已生成，处理 {len(result_rows)} 个问题")
        pass

if __name__ == "__main__":
    main()