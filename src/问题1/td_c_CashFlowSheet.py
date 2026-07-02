import pandas as pd
import numpy as np
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


def calculate_cash_flow_metrics(file_path):
    """
    计算净现金流同比增长率和三个现金流净额占比

    参数:
    file_path: Excel文件路径，包含股票现金流数据

    返回:
    df_result: 包含原始数据和计算结果的DataFrame

    计算说明:
    1. 净现金流同比增长率：(本期净现金流 - 上年同期净现金流) / 上年同期净现金流的绝对值 * 100%
    2. 经营活动现金流净额占比：经营活动现金流净额 / 净现金流(修改为万元) * 100%
    3. 投资活动现金流净额占比：投资活动现金流净额 / 净现金流(修改为万元) * 100%
    4. 筹资活动现金流净额占比：筹资活动现金流净额 / 净现金流(修改为万元) * 100%

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

    # ===================== 新增：去重逻辑 =====================
    # 按股票代码、报告期、报告年份去重，保留最后一行
    print("开始去重：按股票代码、报告期、报告年份去重，保留最后一行...")
    df = df.drop_duplicates(
        subset=['stock_code', 'report_period', 'report_year'],
        keep='last'  # 保留最后一行
    ).reset_index(drop=True)
    print(f"去重后剩余数据：{len(df)} 行")
    # =========================================================

    # 复制数据以避免修改原始数据
    df_result = df.copy()

    # 2. 验证必要列是否存在
    required_columns = ['stock_code', 'report_year', 'report_period',
                        'net_cash_flow', 'operating_cf_net_amount',
                        'investing_cf_net_amount', 'financing_cf_net_amount']

    missing_columns = [col for col in required_columns if col not in df_result.columns]
    if missing_columns:
        raise ValueError(f"数据缺少必要列: {', '.join(missing_columns)}")

    # 3. 计算三个现金流净额占比
    # 处理净现金流为0的情况，避免除以零错误
    sum_NetCashFlow = df_result['operating_cf_net_amount']+df_result['investing_cf_net_amount']+df_result['financing_cf_net_amount']
    net_cf_zero_mask = df_result['net_cash_flow'] == 0
    # 经营活动现金流净额占比
    df_result['operating_cf_ratio_of_net_cf'] = np.where(
        ~net_cf_zero_mask,
        (df_result['operating_cf_net_amount'] / sum_NetCashFlow) * 100,
        np.nan
    )

    # 投资活动现金流净额占比
    df_result['investing_cf_ratio_of_net_cf'] = np.where(
        ~net_cf_zero_mask,
        (df_result['investing_cf_net_amount'] / sum_NetCashFlow) * 100,
        np.nan
    )

    # 筹资活动现金流净额占比
    df_result['financing_cf_ratio_of_net_cf'] = np.where(
        ~net_cf_zero_mask,
        (df_result['financing_cf_net_amount'] / sum_NetCashFlow) * 100,
        np.nan
    )

    # 4. 计算净现金流同比增长率（基于索引匹配，不排序）
    df_result['net_cash_flow_yoy_growth'] = np.nan

    # 获取所有唯一股票代码
    unique_stocks = df_result['stock_code'].unique()
    print(f"共发现 {len(unique_stocks)} 个不同股票代码")

    # 为每个股票单独计算同比增长
    for stock_code in unique_stocks:
        # 获取当前股票的所有数据行索引
        stock_indices = df_result[df_result['stock_code'] == stock_code].index

        for idx in stock_indices:
            # 获取当前行的年份和期间
            current_year = df_result.loc[idx, 'report_year']
            current_period = df_result.loc[idx, 'report_period']
            previous_year = current_year - 1

            # 查找上年同期数据的索引（同股票、同期间、年份-1）
            prev_data_mask = (
                    (df_result['stock_code'] == stock_code) &
                    (df_result['report_period'] == current_period) &
                    (df_result['report_year'] == previous_year)
            )

            # 确保只找到一条匹配数据
            if prev_data_mask.sum() == 1:
                prev_idx = df_result[prev_data_mask].index[0]
                prev_net_cf = df_result.loc[prev_idx, 'net_cash_flow']

                # 避免上年数据为0的情况
                if prev_net_cf != 0:
                    current_net_cf = df_result.loc[idx, 'net_cash_flow']
                    # 使用绝对值分母，确保增长率符号正确
                    yoy_growth = ((current_net_cf - prev_net_cf) / abs(prev_net_cf)) * 100
                    df_result.loc[idx, 'net_cash_flow_yoy_growth'] = yoy_growth

    # 5. 数据格式优化
    # 数值列保留2位小数
    numeric_columns = ['net_cash_flow_yoy_growth', 'operating_cf_ratio_of_net_cf',
                       'investing_cf_ratio_of_net_cf', 'financing_cf_ratio_of_net_cf']
    df_result[numeric_columns] = df_result[numeric_columns].round(2)

    return df_result


# 6. 主函数：执行计算并保存结果
def main():
    # 配置文件路径（可根据实际情况修改）
    input_file = "D:/竞赛/泰迪杯/上交所财务报告/cash_flow_sheet.xlsx"  # 输入文件路径
    output_file = "D:/竞赛/泰迪杯/上交所财务报告/cash_flow_sheet_calculated_1.xlsx"  # 输出文件路径

    try:
        # 执行计算
        print("开始计算现金流指标...")
        result_df = calculate_cash_flow_metrics(input_file)

        # 显示计算结果统计
        print("\n=== 计算结果统计 ===")
        print(f"净现金流同比增长率有效数据: {result_df['net_cash_flow_yoy_growth'].notna().sum()} 条")
        print(f"经营现金流占比有效数据: {result_df['operating_cf_ratio_of_net_cf'].notna().sum()} 条")
        print(f"投资现金流占比有效数据: {result_df['investing_cf_ratio_of_net_cf'].notna().sum()} 条")
        print(f"筹资现金流占比有效数据: {result_df['financing_cf_ratio_of_net_cf'].notna().sum()} 条")

        # 保存结果到Excel文件
        result_df.to_excel(output_file, index=False)
        print(f"\n计算完成！结果已保存至: {output_file}")

        # 显示前5行结果预览
        print("\n=== 结果预览（前5行）===")
        preview_columns = ['stock_code', 'stock_abbr', 'report_year', 'report_period',
                           'net_cash_flow', 'net_cash_flow_yoy_growth',
                           'operating_cf_ratio_of_net_cf', 'investing_cf_ratio_of_net_cf',
                           'financing_cf_ratio_of_net_cf']
        print(result_df[preview_columns].head().to_string(index=False))

        return result_df

    except Exception as e:
        print(f"执行过程中出现错误: {str(e)}")
        raise


# 7. 当脚本直接运行时执行主函数
if __name__ == "__main__":
    main()