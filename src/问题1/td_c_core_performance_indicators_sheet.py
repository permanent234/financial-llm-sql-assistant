import pandas as pd
import numpy as np
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


def calculate_performance_metrics(file_path):
    """
    计算核心财务指标：
    1. 营业总收入 同比增长率、季度环比增长率
    2. 净利润 同比增长率、季度环比增长率
    3. 每股经营现金流量
    4. 销售毛利率
    5. 销售净利率
    6. 每股净资产（新增）

    参数:
    file_path: Excel文件路径，包含核心绩效指标数据

    返回:
    df_result: 包含原始数据和计算结果的DataFrame

    匹配规则:
    - 同比：同股票代码 + 同报告期间 + 报告年份-1
    - 环比：同股票代码 + 报告期顺序(Q1→HY→Q3→FY)
    - 不进行数据排序，完全基于索引匹配
    """

    # 1. 读取Excel数据
    try:
        df = pd.read_excel(file_path, dtype={'stock_code': str})
        print(f"成功读取数据，共 {len(df)} 行，{len(df.columns)} 列")
    except Exception as e:
        print(f"读取文件失败: {str(e)}")
        raise

    # ===================== 去重逻辑 =====================
    # 按股票代码、报告期、报告年份去重，保留最后一行
    print("开始去重：按股票代码、报告期、报告年份去重，保留最后一行...")
    df = df.drop_duplicates(
        subset=['stock_code', 'report_period', 'report_year'],
        keep='last'
    ).reset_index(drop=True)
    print(f"去重后剩余数据：{len(df)} 行")
    # ====================================================

    # 复制数据
    df_result = df.copy()

    # 2. 验证必要列（新增：归属于上市公司股东的净资产列）
    required_columns = [
        'stock_code', 'report_year', 'report_period',
        'total_operating_revenue', 'net_profit_10k_yuan'
    ]   #'operating_cf_net_amount', 'guben', 'total_operating_expenses','归属于上市公司股东的净资产(万元)'

    missing_columns = [col for col in required_columns if col not in df_result.columns]
    if missing_columns:
        raise ValueError(f"数据缺少必要列: {', '.join(missing_columns)}")


    # 报告期顺序（用于环比匹配）
    period_order = {'Q1': 1, 'HY': 2, 'Q3': 3, 'FY': 4}
    df_result['period_seq'] = df_result['report_period'].map(period_order)

    # 获取股票列表
    unique_stocks = df_result['stock_code'].unique()
    print(f"共发现 {len(unique_stocks)} 个不同股票代码")

    # 按股票计算所有指标
    for stock_code in unique_stocks:
        stock_data = df_result[df_result['stock_code'] == stock_code]
        stock_indices = stock_data.index

        # ===================== 1. 计算 同比增长 =====================
        for idx in stock_indices:
            current_year = df_result.loc[idx, 'report_year']
            current_period = df_result.loc[idx, 'report_period']
            previous_year = current_year - 1

            # 匹配上年同期
            prev_data_mask = (
                (df_result['stock_code'] == stock_code) &
                (df_result['report_period'] == current_period) &
                (df_result['report_year'] == previous_year)
            )

            if prev_data_mask.sum() == 1:
                prev_idx = df_result[prev_data_mask].index[0]

                # 营业总收入 同比
                prev_rev = df_result.loc[prev_idx, 'total_operating_revenue']
                curr_rev = df_result.loc[idx, 'total_operating_revenue']
                if prev_rev != 0:
                    rev_yoy = ((curr_rev - prev_rev) / abs(prev_rev)) * 100
                    df_result.loc[idx, 'operating_revenue_yoy_growth'] = rev_yoy

                # 净利润 同比
                prev_np = df_result.loc[prev_idx, 'net_profit_10k_yuan']
                curr_np = df_result.loc[idx, 'net_profit_10k_yuan']
                if prev_np != 0:
                    np_yoy = ((curr_np - prev_np) / abs(prev_np)) * 100
                    df_result.loc[idx, 'net_profit_yoy_growth'] = np_yoy

        # ===================== 2. 计算 季度环比增长 =====================
        for idx in stock_indices:
            curr_seq = df_result.loc[idx, 'period_seq']
            curr_year = df_result.loc[idx, 'report_year']
            last_seq = curr_seq - 1

            # 匹配上一个报告期（同一年、上一个周期）
            last_period_mask = (
                (df_result['stock_code'] == stock_code) &
                (df_result['report_year'] == curr_year) &
                (df_result['period_seq'] == last_seq)
            )

            if last_period_mask.sum() == 1:
                last_idx = df_result[last_period_mask].index[0]

                # 营业总收入 环比
                last_rev = df_result.loc[last_idx, 'total_operating_revenue']
                curr_rev = df_result.loc[idx, 'total_operating_revenue']
                if last_rev != 0:
                    rev_qoq = ((curr_rev - last_rev) / abs(last_rev)) * 100
                    df_result.loc[idx, 'operating_revenue_qoq_growth'] = rev_qoq

                # 净利润 环比
                last_np = df_result.loc[last_idx, 'net_profit_10k_yuan']
                curr_np = df_result.loc[idx, 'net_profit_10k_yuan']
                if last_np != 0:
                    np_qoq = ((curr_np - last_np) / abs(last_np)) * 100
                    df_result.loc[idx, 'net_profit_qoq_growth'] = np_qoq

        # ===================== 3. 计算 静态财务指标（无需匹配） =====================
        for idx in stock_indices:
            # 每股经营现金流量 = 经营现金流净额 * 10000 / 总股本
            cf = df_result.loc[idx, 'operating_cf_net_amount']
            gb = df_result.loc[idx, 'guben']
            if pd.notna(cf) and pd.notna(gb) and gb != 0:
                df_result.loc[idx, 'operating_cf_per_share'] = (cf * 10000) / gb

            # 销售毛利率 = (营业总收入 - 营业成本) / 营业总收入 * 100%
            rev = df_result.loc[idx, 'total_operating_revenue']
            cost = df_result.loc[idx, 'total_operating_expenses']
            if pd.notna(rev) and pd.notna(cost) and rev != 0:
                df_result.loc[idx, 'gross_profit_margin'] = ((rev - cost) / rev) * 100

            # 销售净利率 = 净利润 / 营业总收入 * 100%
            np = df_result.loc[idx, 'net_profit_10k_yuan']
            if pd.notna(np) and pd.notna(rev) and rev != 0:
                df_result.loc[idx, 'net_profit_margin'] = (np / rev) * 100

            # 加权平均净资产收益率(扣非) =====================
            # 公式：roe_weighted_excl_non_recurring = ROE / 净利润 * 扣非净利润
            roe_val = df_result.loc[idx, 'roe']
            np = df_result.loc[idx, 'net_profit_10k_yuan']
            np_excl = df_result.loc[idx, 'net_profit_excl_non_recurring']

            if pd.notna(roe_val) and pd.notna(np) and pd.notna(np_excl) and np != 0:
                roe_excl = (roe_val / np) * np_excl
                df_result.loc[idx, 'roe_weighted_excl_non_recurring'] = roe_excl

            # ===================== 【新增】每股净资产 =====================
            # 公式：net_asset_per_share = 归属于上市公司股东的净资产(万元) * 10000 / 总股本
            #net_asset = df_result.loc[idx, '归属于上市公司股东的净资产(万元)']
            #guben = df_result.loc[idx, 'guben']
            #if pd.notna(net_asset) and pd.notna(guben) and guben != 0:
            #    df_result.loc[idx, 'net_asset_per_share'] = (net_asset * 10000) / guben

    # 删除辅助列
    df_result = df_result.drop(columns=['period_seq'])

    # 保留4位小数（新增列加入）
    numeric_cols = [
        'operating_revenue_yoy_growth', 'net_profit_yoy_growth',
        'operating_revenue_qoq_growth', 'net_profit_qoq_growth',
        'operating_cf_per_share', 'gross_profit_margin', 'net_profit_margin',
        'net_asset_per_share'  # 新增
    ]
    df_result[numeric_cols] = df_result[numeric_cols].round(4)

    return df_result


# 主函数
def main():
    # 你的文件路径
    input_file = r"D:\竞赛\泰迪杯\上交所财务报告\core_performance_indicators_sheet.xlsx"
    output_file = r"D:\竞赛\泰迪杯\上交所财务报告\core_performance_indicators_calculated_1.xlsx"

    try:
        print("开始计算核心绩效指标...")
        result_df = calculate_performance_metrics(input_file)

        print("\n=== 计算结果统计 ===")
        print(f"营业总收入同比有效数据: {result_df['operating_revenue_yoy_growth'].notna().sum()} 条")
        print(f"净利润同比有效数据: {result_df['net_profit_yoy_growth'].notna().sum()} 条")
        print(f"营业总收入环比有效数据: {result_df['operating_revenue_qoq_growth'].notna().sum()} 条")
        print(f"净利润环比有效数据: {result_df['net_profit_qoq_growth'].notna().sum()} 条")
        print(f"每股经营现金流有效数据: {result_df['operating_cf_per_share'].notna().sum()} 条")
        print(f"销售毛利率有效数据: {result_df['gross_profit_margin'].notna().sum()} 条")
        print(f"销售净利率有效数据: {result_df['net_profit_margin'].notna().sum()} 条")
        print(f"每股净资产有效数据: {result_df['net_asset_per_share'].notna().sum()} 条")  # 新增统计

        # 保存结果
        result_df.to_excel(output_file, index=False)
        print(f"\n计算完成！结果已保存至: {output_file}")

        # 预览结果（新增列加入预览）
        print("\n=== 结果预览（前5行）===")
        preview_cols = [
            'stock_code', 'report_year', 'report_period',
            'operating_revenue_yoy_growth', 'net_profit_yoy_growth',
            'operating_revenue_qoq_growth', 'net_profit_qoq_growth',
            'operating_cf_per_share', 'gross_profit_margin', 'net_profit_margin',
            'net_asset_per_share'  # 新增预览
        ]
        print(result_df[preview_cols].head().to_string(index=False))

        return result_df

    except Exception as e:
        print(f"执行出错: {str(e)}")
        raise


if __name__ == "__main__":
    main()