# 基于大语言模型的财务"智能问数"助手

> **泰迪杯数据挖掘挑战赛 B 题作品**

## 项目简介

本项目是一个**基于大语言模型（LLM）的财务数据智能问答系统**，能够将用户的自然语言查询自动转换为 SQL 语句，查询上市公司财务报表数据，并生成可视化图表和自然语言回答。

系统核心采用 **RAG（检索增强生成）+ LLM** 架构，结合语义检索、意图澄清、SQL 生成与修复、多轮对话管理等模块，实现高精度的财务智能问答。

## 主要功能

- **自然语言转 SQL**：用户用中文提问，系统自动生成 SQLite 查询语句
- **意图澄清**：当查询缺少关键信息（如未指定公司名）时，系统自动追问
- **多轮对话**：支持上下文继承，实现连贯的多轮交互
- **SQL 自修复**：执行失败时自动调用 LLM 修复 SQL
- **智能可视化**：根据查询内容自动生成折线图、柱状图、饼图、水平条形图等
- **批量处理**：支持 Excel 批量问题导入与结果导出
- **自然语言回答**：基于查询结果生成专业的财务分析回答

## 技术栈

| 模块 | 技术 |
|------|------|
| 大语言模型 | DeepSeek API |
| 语义检索 | Sentence-Transformers (paraphrase-multilingual-MiniLM-L12-v2) |
| 数据库 | SQLite |
| 数据可视化 | Matplotlib |
| 数据处理 | Pandas |
| 微调框架 | PEFT + LoRA (问题一模型微调) |

## 项目结构

```
├── src/                          # 核心代码
│   ├── question2.py              # 问题二：智能问数助手主程序（SQLGPT）
│   ├── 问题1.py                  # 问题一：财报 PDF 解析
│   ├── 问题2.py                  # 问题二：数据合并与预处理
│   ├── 1.py                      # 问题一：SQL 微调数据集生成
│   ├── 2.py                      # 问题二：SQL 生成实验脚本
│   └── 问题1/                    # 问题一：PDF 解析各表模块
│       ├── td_c_blance.py        # 资产负债表解析
│       ├── td_c_incomes.py       # 利润表解析
│       ├── td_c_CashFlowSheet.py # 现金流量表解析
│       ├── td_c_meigu.py         # 每股指标解析
│       ├── td_c_core_performance_indicators_sheet.py # 核心业绩指标解析
│       ├── 上交.py               # 上交所 PDF 处理
│       └── 深交.py               # 深交所 PDF 处理
├── data/                         # 数据集
│   ├── core_performance_indicators_sheet.csv  # 核心业绩指标表
│   ├── balance_sheet.csv         # 资产负债表
│   ├── income_sheet.csv          # 利润表
│   ├── cash_flow_sheet.csv       # 现金流量表
│   ├── financial_reports.db      # SQLite 数据库（自动构建）
│   └── 附件4：问题汇总.xlsx       # 批量测试问题
├── result/                       # 输出结果
│   ├── *.jpg                     # 可视化图表（问题二批量输出）
│   └── result_2.xlsx             # 批量处理结果
├── sqlgpt_model/                 # 微调模型（PEFT + LoRA）
│   ├── adapter_config.json       # 适配器配置
│   ├── adapter_model.safetensors # 模型权重
│   ├── tokenizer_config.json     # 分词器配置
│   └── README.md                 # 模型说明
├── assets/                       # 资源文件
│   └── 参赛论文.docx              # 泰迪杯参赛论文
├── README.md                     # 项目说明
└── .gitignore
```

## 核心模块说明

### SQLGPTInference（question2.py）

系统核心类，实现完整的智能问答流程：

1. **Schema 构建**：从 CSV 文件自动构建 SQLite 数据库及表结构描述
2. **RAG 检索**：使用 Sentence-Transformers 对查询进行语义检索，匹配最相关的数据表
3. **意图澄清**：检测查询是否缺少关键信息，自动触发追问
4. **Prompt 工程**：构建包含 Schema、示例、约束条件的结构化提示词
5. **SQL 生成**：调用 DeepSeek API 生成 SQL
6. **SQL 修复**：自动纠正常见错误（字段映射、日期格式、表名修正等）
7. **执行与修复**：执行失败时调用 LLM 进行最小化修复
8. **可视化生成**：根据数据特征自动选择图表类型（折线/柱状/水平条形/饼图）
9. **自然语言回答**：基于查询结果生成专业的财务分析回答

### 关键特性

- **指标护栏（Guardrails）**：预定义财务指标与字段的映射关系，防止 LLM 误映射（如将"经营活动现金流量净额"映射到每股指标）
- **字段约束**：自动修正常见错误字段映射（如 `eps_basic` → `eps`）
- **日期处理**：自动识别并规范化多种日期格式
- **多轮上下文继承**：支持省略公司名/年份的后续追问

## 运行方式

### 环境准备

```bash
pip install pandas torch matplotlib sentence-transformers openai pdfplumber
```

### 配置 API Key

```bash
export DEEPSEEK_API_KEY="your-api-key"
```

### 交互模式

```bash
python src/question2.py --mode infer --data-dir ./data --llm-api-key "your-api-key"
```

### 批量处理模式

```bash
python src/question2.py --mode batch2 --data-dir ./data --llm-api-key "your-api-key"
```

## 可视化效果

系统支持多种图表类型：
- **折线图**：趋势分析（如近三年净利润变化）
- **柱状图**：对比分析（如多家公司总资产对比）
- **水平条形图**：排名展示（如 Top10 公司利润总额）
- **饼图**：占比分析（如现金流结构分布）

## 数据说明

数据来源：泰迪杯 B 题提供的上市公司财务报表数据，包含 4 张核心财务表：
- 核心业绩指标表（EPS、ROE、毛利率等）
- 资产负债表（总资产、负债、资产负债率等）
- 利润表（营业收入、净利润、利润总额等）
- 现金流量表（经营/投资/融资现金流等）

涵盖 10 家中药行业上市公司：金花股份、华润三九、片仔癀、同仁堂、太极集团、中恒集团、白云山、东阿阿胶、羚锐制药、昆药集团。

## 个人贡献

本人主要负责：
- **问题二的模型构建与代码实现**：设计并实现了 SQLGPT 智能问答系统的核心架构，包括 RAG 检索、意图澄清、SQL 生成与修复、多轮对话管理、可视化生成等模块
- **论文撰写**：负责除问题三外的全部论文内容撰写
- **可视化实现**：设计并实现了系统的自动图表生成模块，支持多种图表类型的智能选择

## 论文

详见 `assets/参赛论文.docx`

## 许可证

本项目仅用于学术交流和比赛展示，数据来源于泰迪杯官方提供。

---

> 📌 **GitHub**: https://github.com/permanent234/financial-llm-sql-assistant
