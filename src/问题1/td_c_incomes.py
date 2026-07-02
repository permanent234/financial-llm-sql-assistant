import pandas as pd
import numpy as np
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


def calculate_income_sheet_metrics(file_path):
    """
    计算净利润同比增长率和营业总收入同比增长率

    参数:
    file_path: Excel文件路径，包含股票利润表数据

    返回:
    df_result: 包含原始数据和计算结果的DataFrame

    计算说明:
    1. 净利润同比增长率：(本期净利润 - 上年同期净利润) / 上年同期净利润的绝对值 * 100%
    2. 营业总收入同比增长率：(本期营业总收入 - 上年同期营业总收入) / 上年同期营业总收入的绝对值 * 100%

    匹配规则:
    - 使用股票代码(stock_code)、报告期间(report_period)、报告年份(report_year)进行精确匹配
    - 上年同期 = 同股票代码 + 同报告期间 + 报告年份-1
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

    # 2. 验证必要列
    required_columns = ['stock_code', 'report_year', 'report_period',
                        'net_profit', 'total_operating_revenue']

    missing_columns = [col for col in required_columns if col not in df_result.columns]
    if missing_columns:
        raise ValueError(f"数据缺少必要列: {', '.join(missing_columns)}")

    # 3. 初始化同比列
    df_result['net_profit_yoy_growth'] = np.nan
    df_result['operating_revenue_yoy_growth'] = np.nan

    # 获取股票列表
    unique_stocks = df_result['stock_code'].unique()
    print(f"共发现 {len(unique_stocks)} 个不同股票代码")

    # 按股票计算同比
    for stock_code in unique_stocks:
        stock_indices = df_result[df_result['stock_code'] == stock_code].index

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

                # ========== 净利润同比 ==========
                prev_net_profit = df_result.loc[prev_idx, 'net_profit']
                current_net_profit = df_result.loc[idx, 'net_profit']
                if prev_net_profit != 0:
                    net_yoy = ((current_net_profit - prev_net_profit) / abs(prev_net_profit)) * 100
                    df_result.loc[idx, 'net_profit_yoy_growth'] = net_yoy

                # ========== 营业总收入同比 ==========
                prev_revenue = df_result.loc[prev_idx, 'total_operating_revenue']
                current_revenue = df_result.loc[idx, 'total_operating_revenue']
                if prev_revenue != 0:
                    rev_yoy = ((current_revenue - prev_revenue) / abs(prev_revenue)) * 100
                    df_result.loc[idx, 'operating_revenue_yoy_growth'] = rev_yoy

    # 保留2位小数
    numeric_cols = ['net_profit_yoy_growth', 'operating_revenue_yoy_growth']
    df_result[numeric_cols] = df_result[numeric_cols].round(2)

    return df_result


# 主函数
def main():
    input_file = "D:/竞赛/泰迪杯/上交所财务报告/income_sheet.xlsx"
    output_file = "D:/竞赛/泰迪杯/上交所财务报告/income_sheet_calculated_1.xlsx"

    try:
        print("开始计算利润表指标...")
        result_df = calculate_income_sheet_metrics(input_file)

        print("\n=== 计算结果统计 ===")
        print(f"净利润同比有效数据: {result_df['net_profit_yoy_growth'].notna().sum()} 条")
        print(f"营业总收入同比有效数据: {result_df['operating_revenue_yoy_growth'].notna().sum()} 条")

        result_df.to_excel(output_file, index=False)
        print(f"\n计算完成！结果已保存至: {output_file}")

        print("\n=== 结果预览（前5行）===")
        preview = ['stock_code', 'stock_abbr', 'report_year', 'report_period',
                   'net_profit', 'net_profit_yoy_growth',
                   'total_operating_revenue', 'operating_revenue_yoy_growth']
        print(result_df[preview].head().to_string(index=False))

        return result_df

    except Exception as e:
        print(f"执行出错: {str(e)}")
        raise


if __name__ == "__main__":
    main()