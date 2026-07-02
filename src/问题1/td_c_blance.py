import pandas as pd
import numpy as np
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

def calculate_balance_sheet_metrics(file_path):
    """
    计算总资产同比增长率、总负债同比增长率、资产负债率

    参数:
    file_path: Excel文件路径，包含股票资产负债表数据

    返回:
    df_result: 包含原始数据和计算结果的DataFrame

    计算说明:
    1. 总资产同比增长率：(本期总资产 - 上年同期总资产) / 上年同期总资产的绝对值 * 100%
    2. 总负债同比增长率：(本期总负债 - 上年同期总负债) / 上年同期总负债的绝对值 * 100%
    3. 资产负债率：总负债 / 总资产 * 100%

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
        keep='last'  # 保留最后一行
    ).reset_index(drop=True)
    print(f"去重后剩余数据：{len(df)} 行")
    # ====================================================

    # 复制数据以避免修改原始数据
    df_result = df.copy()

    # 2. 验证必要列是否存在
    required_columns = ['stock_code', 'report_year', 'report_period',
                        'asset_total_assets', 'liability_total_liabilities']

    missing_columns = [col for col in required_columns if col not in df_result.columns]
    if missing_columns:
        raise ValueError(f"数据缺少必要列: {', '.join(missing_columns)}")

    # 3. 计算资产负债率
    # 处理总资产为0的情况，避免除以零错误
    asset_zero_mask = df_result['asset_total_assets'] == 0
    df_result['asset_liability_ratio'] = np.where(
        ~asset_zero_mask,
        (df_result['liability_total_liabilities'] / df_result['asset_total_assets']) * 100,
        np.nan
    )

    # 4. 计算总资产同比增长率（基于索引匹配，不排序）
    df_result['asset_total_assets_yoy_growth'] = np.nan

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
                prev_asset = df_result.loc[prev_idx, 'asset_total_assets']

                # 避免上年数据为0的情况
                if prev_asset != 0:
                    current_asset = df_result.loc[idx, 'asset_total_assets']
                    # 使用绝对值分母，确保增长率符号正确
                    yoy_growth = ((current_asset - prev_asset) / abs(prev_asset)) * 100
                    df_result.loc[idx, 'asset_total_assets_yoy_growth'] = yoy_growth

    # 5. 计算总负债同比增长率（基于索引匹配，不排序）
    df_result['liability_total_liabilities_yoy_growth'] = np.nan

    for stock_code in unique_stocks:
        stock_indices = df_result[df_result['stock_code'] == stock_code].index

        for idx in stock_indices:
            current_year = df_result.loc[idx, 'report_year']
            current_period = df_result.loc[idx, 'report_period']
            previous_year = current_year - 1

            # 查找上年同期数据
            prev_data_mask = (
                    (df_result['stock_code'] == stock_code) &
                    (df_result['report_period'] == current_period) &
                    (df_result['report_year'] == previous_year)
            )

            if prev_data_mask.sum() == 1:
                prev_idx = df_result[prev_data_mask].index[0]
                prev_liability = df_result.loc[prev_idx, 'liability_total_liabilities']

                if prev_liability != 0:
                    current_liability = df_result.loc[idx, 'liability_total_liabilities']
                    yoy_growth = ((current_liability - prev_liability) / abs(prev_liability)) * 100
                    df_result.loc[idx, 'liability_total_liabilities_yoy_growth'] = yoy_growth

    # 6. 数据格式优化
    # 数值列保留2位小数
    numeric_columns = ['asset_total_assets_yoy_growth',
                       'liability_total_liabilities_yoy_growth',
                       'asset_liability_ratio']
    df_result[numeric_columns] = df_result[numeric_columns].round(2)

    return df_result


# 6. 主函数：执行计算并保存结果
def main():
    # 配置文件路径（直接改成你的资产负债表路径）
    input_file = "D:/竞赛/泰迪杯/上交所财务报告/balance_sheet.xlsx"
    output_file = "D:/竞赛/泰迪杯/上交所财务报告/balance_sheet_calculated_1.xlsx"

    try:
        # 执行计算
        print("开始计算资产负债表指标...")
        result_df = calculate_balance_sheet_metrics(input_file)

        # 显示计算结果统计
        print("\n=== 计算结果统计 ===")
        print(f"总资产同比增长率有效数据: {result_df['asset_total_assets_yoy_growth'].notna().sum()} 条")
        print(f"总负债同比增长率有效数据: {result_df['liability_total_liabilities_yoy_growth'].notna().sum()} 条")
        print(f"资产负债率有效数据: {result_df['asset_liability_ratio'].notna().sum()} 条")

        # 保存结果到Excel文件
        result_df.to_excel(output_file, index=False)
        print(f"\n计算完成！结果已保存至: {output_file}")

        # 显示前5行结果预览
        print("\n=== 结果预览（前5行）===")
        preview_columns = ['stock_code', 'stock_abbr', 'report_year', 'report_period',
                           'asset_total_assets', 'asset_total_assets_yoy_growth',
                           'liability_total_liabilities', 'liability_total_liabilities_yoy_growth',
                           'asset_liability_ratio']
        print(result_df[preview_columns].head().to_string(index=False))

        return result_df

    except Exception as e:
        print(f"执行过程中出现错误: {str(e)}")
        raise


# 7. 当脚本直接运行时执行主函数
if __name__ == "__main__":
    main()