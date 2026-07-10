import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 1. 数据加载与初步探索
def load_and_explore(file_path):
    """加载数据并进行初步探索"""
    print("="*60)
    print("1. 数据加载与初步探索")
    print("="*60)
    
    # 加载数据
    df = pd.read_csv(file_path, encoding='utf-8')
    
    print(f"\n数据集形状: {df.shape}")
    print(f"\n数据列名:\n{df.columns.tolist()}")
    print(f"\n数据类型:\n{df.dtypes}")
    print(f"\n缺失值统计:\n{df.isnull().sum()}")
    
    # 查看数据样例
    print(f"\n数据前5行:\n{df.head()}")
    
    # 数据统计描述
    print(f"\n数值列统计描述:\n{df.describe()}")
    
    return df

# 2. 数据清洗与预处理
def clean_data(df):
    """数据清洗"""
    print("\n" + "="*60)
    print("2. 数据清洗")
    print("="*60)
    
    # 复制数据，避免修改原数据
    df_clean = df.copy()
    
    # 检查并删除金额为负数的行（可能是退货，但已经用return_flag标记了）
    # 注意：Quantity为负数的行也是退货
    print(f"\n清洗前数据量: {len(df_clean)}")
    
    # 删除CustomerID为空的行（无法识别用户）
    before = len(df_clean)
    df_clean = df_clean.dropna(subset=['CustomerID'])
    print(f"删除CustomerID为空的行: {before} -> {len(df_clean)}")
    
    # 删除Quantity <= 0的行（退货或无效记录）
    before = len(df_clean)
    df_clean = df_clean[df_clean['Quantity'] > 0]
    print(f"删除Quantity<=0的行: {before} -> {len(df_clean)}")
    
    # 删除UnitPrice <= 0的行
    before = len(df_clean)
    df_clean = df_clean[df_clean['UnitPrice'] > 0]
    print(f"删除UnitPrice<=0的行: {before} -> {len(df_clean)}")
    
    # 转换日期格式
    df_clean['InvoiceDate'] = pd.to_datetime(df_clean['InvoiceDate'])
    
    # 添加年月日便于分析
    df_clean['Date'] = df_clean['InvoiceDate'].dt.date
    df_clean['YearMonth'] = df_clean['InvoiceDate'].dt.to_period('M')
    
    print(f"\n清洗后数据量: {len(df_clean)}")
    print(f"时间范围: {df_clean['InvoiceDate'].min()} 到 {df_clean['InvoiceDate'].max()}")
    
    return df_clean

# 3. 销售额与订单量趋势分析
def sales_order_trend_analysis(df):
    """销售额与订单量趋势分析"""
    print("\n" + "="*60)
    print("3. 销售额与订单量趋势分析")
    print("="*60)
    
    # 按日期聚合
    daily_sales = df.groupby('Date').agg({
        'Amount': 'sum',           # 销售额
        'InvoiceNo': 'nunique'     # 订单数（去重）
    }).rename(columns={'Amount': 'Total_Sales', 'InvoiceNo': 'Order_Count'})
    
    print(f"\n每日销售统计:\n{daily_sales}")
    
    # 总体指标
    total_sales = df['Amount'].sum()
    total_orders = df['InvoiceNo'].nunique()
    total_customers = df['CustomerID'].nunique()
    total_items = df['Quantity'].sum()
    avg_order_value = total_sales / total_orders
    
    print(f"\n总体经营指标:")
    print(f"  总销售额: £{total_sales:,.2f}")
    print(f"  总订单数: {total_orders:,}")
    print(f"  总客户数: {total_customers:,}")
    print(f"  总销售件数: {total_items:,}")
    print(f"  平均客单价: £{avg_order_value:.2f}")
    
    # 绘制趋势图
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # 销售额趋势
    axes[0].plot(range(len(daily_sales)), daily_sales['Total_Sales'], 
                 marker='o', linewidth=2, markersize=8, color='#2E86AB')
    axes[0].set_title('每日销售额趋势', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('销售额 (英镑)', fontsize=12)
    axes[0].set_xticks(range(len(daily_sales)))
    axes[0].set_xticklabels([str(d) for d in daily_sales.index], rotation=45)
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(y=daily_sales['Total_Sales'].mean(), color='red', 
                    linestyle='--', label=f"均值: £{daily_sales['Total_Sales'].mean():,.0f}")
    axes[0].legend()
    
    # 订单量趋势
    axes[1].plot(range(len(daily_sales)), daily_sales['Order_Count'], 
                 marker='s', linewidth=2, markersize=8, color='#A23B72')
    axes[1].set_title('每日订单量趋势', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('订单数量', fontsize=12)
    axes[1].set_xlabel('日期', fontsize=12)
    axes[1].set_xticks(range(len(daily_sales)))
    axes[1].set_xticklabels([str(d) for d in daily_sales.index], rotation=45)
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=daily_sales['Order_Count'].mean(), color='red', 
                    linestyle='--', label=f"均值: {daily_sales['Order_Count'].mean():.0f}")
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig('sales_order_trend.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return daily_sales

# 4. 小时销售分析（业务时段洞察）
def hourly_sales_analysis(df):
    """按小时分析销售情况"""
    print("\n" + "="*60)
    print("4. 小时销售分析")
    print("="*60)
    
    # 提取小时
    df_hour = df.copy()
    df_hour['Hour'] = df_hour['InvoiceDate'].dt.hour
    
    hourly_stats = df_hour.groupby('Hour').agg({
        'Amount': ['sum', 'count', 'mean'],
        'InvoiceNo': 'nunique'
    }).round(2)
    
    hourly_stats.columns = ['Sales_Amount', 'Transaction_Count', 'Avg_Transaction_Value', 'Order_Count']
    
    print(f"\n各小时销售统计:\n{hourly_stats}")
    
    # 找出销售高峰时段
    peak_hour = hourly_stats['Sales_Amount'].idxmax()
    print(f"\n销售高峰时段: {peak_hour}:00 - {peak_hour+1}:00")
    print(f"  该时段销售额: £{hourly_stats.loc[peak_hour, 'Sales_Amount']:,.2f}")
    print(f"  该时段订单数: {hourly_stats.loc[peak_hour, 'Order_Count']}")
    
    # 绘制小时销售热力图
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 小时销售额柱状图
    axes[0].bar(hourly_stats.index, hourly_stats['Sales_Amount'], 
                color='#4CAF50', edgecolor='black')
    axes[0].set_title('各小时销售额分布', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('小时 (24小时制)', fontsize=12)
    axes[0].set_ylabel('销售额 (英镑)', fontsize=12)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # 小时订单量柱状图
    axes[1].bar(hourly_stats.index, hourly_stats['Order_Count'], 
                color='#FF9800', edgecolor='black')
    axes[1].set_title('各小时订单量分布', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('小时 (24小时制)', fontsize=12)
    axes[1].set_ylabel('订单数量', fontsize=12)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('hourly_sales_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return hourly_stats

# 5. 热门商品分析
def product_analysis(df):
    """商品分析"""
    print("\n" + "="*60)
    print("5. 热门商品分析")
    print("="*60)
    
    # 按销量排序
    top_by_quantity = df.groupby(['StockCode', 'Description']).agg({
        'Quantity': 'sum',
        'Amount': 'sum',
        'InvoiceNo': 'nunique'
    }).sort_values('Quantity', ascending=False).head(20)
    top_by_quantity.columns = ['总销量', '总销售额', '购买订单数']
    
    print(f"\n按销量排名 Top 20 商品:\n{top_by_quantity}")
    
    # 按销售额排序
    top_by_revenue = df.groupby(['StockCode', 'Description'])['Amount']\
        .sum().sort_values(ascending=False).head(20)
    
    print(f"\n按销售额排名 Top 20 商品:\n{top_by_revenue}")
    
    # 绘制图表
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # 销量Top 10
    top10_qty = top_by_quantity.head(10)
    desc_short = [d[:40] + '...' if len(str(d)) > 40 else str(d) for d in top10_qty.index.get_level_values(1)]
    
    axes[0].barh(range(len(desc_short)), top10_qty['总销量'], 
                 color='#2196F3', edgecolor='black')
    axes[0].set_yticks(range(len(desc_short)))
    axes[0].set_yticklabels(desc_short, fontsize=10)
    axes[0].set_xlabel('总销量 (件)', fontsize=12)
    axes[0].set_title('销量 Top 10 商品', fontsize=14, fontweight='bold')
    axes[0].invert_yaxis()
    
    # 销售额Top 10
    top10_rev = top_by_revenue.head(10)
    products_rev = [p[:40] + '...' if len(str(p)) > 40 else str(p) for p in top10_rev.index.get_level_values(1)]
    
    axes[1].barh(range(len(products_rev)), top10_rev.values, 
                 color='#E91E63', edgecolor='black')
    axes[1].set_yticks(range(len(products_rev)))
    axes[1].set_yticklabels(products_rev, fontsize=10)
    axes[1].set_xlabel('总销售额 (英镑)', fontsize=12)
    axes[1].set_title('销售额 Top 10 商品', fontsize=14, fontweight='bold')
    axes[1].invert_yaxis()
    
    plt.tight_layout()
    plt.savefig('top_products.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return top_by_quantity, top_by_revenue

# 6. 退货订单分析
def return_order_analysis(df):
    """退货订单分析"""
    print("\n" + "="*60)
    print("6. 退货订单分析")
    print("="*60)
    
    # 识别退货订单（Quantity为负数）
    returns = df[df['Quantity'] < 0].copy()
    # 识别退货订单（使用return_flag）
    returns_flag = df[df['return_flag'] == 1].copy()
    
    print(f"\n通过数量识别退货记录数: {len(returns)}")
    print(f"通过退货标记识别退货记录数: {len(returns_flag)}")
    
    # 退货订单统计
    if len(returns) > 0:
        # 获取唯一退货订单
        return_invoices = returns['InvoiceNo'].unique()
        # 注意：退货订单号通常以C开头
        c_invoices = df[df['InvoiceNo'].astype(str).str.startswith('C', na=False)]['InvoiceNo'].unique()
        
        print(f"\n退货订单统计:")
        print(f"  唯一退货订单数 ('C'开头): {len(c_invoices)}")
        print(f"  退货记录数: {len(returns)}")
        print(f"  退货总金额: £{abs(returns['Amount'].sum()):,.2f}")
        print(f"  退货总数量: {abs(returns['Quantity'].sum()):,.0f} 件")
        
        # 退货最多的商品
        top_return_products = returns.groupby(['StockCode', 'Description']).agg({
            'Quantity': 'sum',
            'Amount': 'sum'
        }).sort_values('Quantity').head(10)
        top_return_products.columns = ['退货数量', '退货金额']
        
        print(f"\n退货最多的商品 Top 10:\n{top_return_products}")
        
        # 绘图
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # 退货金额按日分布
        returns['Date'] = returns['InvoiceDate'].dt.date
        daily_returns = returns.groupby('Date')['Amount'].sum().abs()
        
        axes[0].bar(range(len(daily_returns)), daily_returns.values, 
                    color='#F44336', edgecolor='black')
        axes[0].set_title('每日退货金额分布', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('日期', fontsize=10)
        axes[0].set_ylabel('退货金额 (英镑)', fontsize=10)
        axes[0].set_xticks(range(len(daily_returns)))
        axes[0].set_xticklabels([str(d) for d in daily_returns.index], rotation=45, fontsize=8)
        
        # 退货Top 5商品
        top5_return = top_return_products.head(5)
        products_return = [p[:35] + '...' if len(str(p)) > 35 else str(p) 
                          for p in top5_return.index.get_level_values(1)]
        
        axes[1].barh(range(len(products_return)), abs(top5_return['退货数量'].values), 
                     color='#FF5722', edgecolor='black')
        axes[1].set_yticks(range(len(products_return)))
        axes[1].set_yticklabels(products_return, fontsize=9)
        axes[1].set_xlabel('退货数量 (件)', fontsize=10)
        axes[1].set_title('退货数量 Top 5 商品', fontsize=12, fontweight='bold')
        axes[1].invert_yaxis()
        
        plt.tight_layout()
        plt.savefig('return_analysis.png', dpi=150, bbox_inches='tight')
        plt.show()
    else:
        print("\n未发现退货记录")
    
    return returns

# 7. 客户行为分析
def customer_behavior_analysis(df):
    """客户行为分析"""
    print("\n" + "="*60)
    print("7. 客户行为分析")
    print("="*60)
    
    # 客户消费统计
    customer_stats = df.groupby('CustomerID').agg({
        'Amount': 'sum',           # 总消费金额
        'InvoiceNo': 'nunique',    # 订单数
        'Quantity': 'sum'          # 购买商品总数
    }).rename(columns={'Amount': 'Total_Spend', 'InvoiceNo': 'Order_Count', 'Quantity': 'Total_Items'})
    
    print(f"\n客户统计:")
    print(f"  总客户数: {len(customer_stats)}")
    print(f"  平均客户消费: £{customer_stats['Total_Spend'].mean():.2f}")
    print(f"  中位数客户消费: £{customer_stats['Total_Spend'].median():.2f}")
    print(f"  平均客户订单数: {customer_stats['Order_Count'].mean():.2f}")
    print(f"  平均客户购买件数: {customer_stats['Total_Items'].mean():.2f}")
    
    # 客户分层（基于消费金额）
    customer_stats['Customer_Segment'] = pd.cut(
        customer_stats['Total_Spend'],
        bins=[0, 50, 200, 500, float('inf')],
        labels=['低价值', '中价值', '高价值', 'VIP']
    )
    
    segment_dist = customer_stats['Customer_Segment'].value_counts()
    print(f"\n客户分层分布:")
    for segment, count in segment_dist.items():
        pct = count / len(customer_stats) * 100
        print(f"  {segment}: {count} 人 ({pct:.1f}%)")
    
    # 顶级客户
    top_customers = customer_stats.nlargest(10, 'Total_Spend')
    print(f"\n顶级客户 Top 10:\n{top_customers}")
    
    # 绘制客户消费分布
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 客户消费金额直方图
    axes[0].hist(customer_stats['Total_Spend'], bins=50, color='#9C27B0', 
                 edgecolor='black', alpha=0.7)
    axes[0].set_title('客户消费金额分布', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('消费金额 (英镑)', fontsize=10)
    axes[0].set_ylabel('客户数量', fontsize=10)
    axes[0].set_xlim(0, 1000)
    
    # 客户订单数分布
    axes[1].hist(customer_stats['Order_Count'], bins=30, color='#00BCD4', 
                 edgecolor='black', alpha=0.7)
    axes[1].set_title('客户订单数量分布', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('订单数量', fontsize=10)
    axes[1].set_ylabel('客户数量', fontsize=10)
    axes[1].set_xlim(0, 20)
    
    plt.tight_layout()
    plt.savefig('customer_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return customer_stats

# 8. 国家销售分析
def country_analysis(df):
    """国家销售分析"""
    print("\n" + "="*60)
    print("8. 国家销售分析")
    print("="*60)
    
    # 按国家聚合
    country_sales = df.groupby('Country').agg({
        'Amount': 'sum',
        'InvoiceNo': 'nunique',
        'CustomerID': 'nunique',
        'Quantity': 'sum'
    }).rename(columns={
        'Amount': 'Total_Sales', 
        'InvoiceNo': 'Order_Count',
        'CustomerID': 'Customer_Count',
        'Quantity': 'Items_Sold'
    }).sort_values('Total_Sales', ascending=False)
    
    print(f"\n各国销售统计 (Top 10):\n{country_sales.head(10)}")
    
    # 计算占比
    country_sales['Sales_Percentage'] = country_sales['Total_Sales'] / country_sales['Total_Sales'].sum() * 100
    
    # 绘制Top 10国家销售额
    top_countries = country_sales.head(10)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 销售额柱状图
    axes[0].barh(range(len(top_countries)), top_countries['Total_Sales'], 
                 color='#3F51B5', edgecolor='black')
    axes[0].set_yticks(range(len(top_countries)))
    axes[0].set_yticklabels(top_countries.index, fontsize=10)
    axes[0].set_xlabel('总销售额 (英镑)', fontsize=12)
    axes[0].set_title('各国销售额 Top 10', fontsize=14, fontweight='bold')
    axes[0].invert_yaxis()
    
    # 销售额占比饼图
    other_sales = country_sales.iloc[5:]['Total_Sales'].sum()
    pie_data = country_sales.head(5)['Total_Sales'].tolist()
    pie_labels = country_sales.head(5).index.tolist()
    pie_data.append(other_sales)
    pie_labels.append('其他')
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#D4A5A5']
    axes[1].pie(pie_data, labels=pie_labels, autopct='%1.1f%%', startangle=90, colors=colors)
    axes[1].set_title('各国销售额占比', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('country_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return country_sales

# 9. 异常交易检测
def anomaly_detection(df):
    """异常交易检测"""
    print("\n" + "="*60)
    print("9. 异常交易检测")
    print("="*60)
    
    # 按订单聚合
    order_stats = df.groupby('InvoiceNo').agg({
        'Amount': 'sum',
        'Quantity': 'sum',
        'CustomerID': 'first'
    }).reset_index()
    
    # 使用IQR方法检测异常
    Q1_amount = order_stats['Amount'].quantile(0.25)
    Q3_amount = order_stats['Amount'].quantile(0.75)
    IQR_amount = Q3_amount - Q1_amount
    outlier_threshold = Q3_amount + 3 * IQR_amount
    
    # 识别大额异常订单
    large_orders = order_stats[order_stats['Amount'] > outlier_threshold].sort_values('Amount', ascending=False)
    
    print(f"\n订单金额统计:")
    print(f"  平均订单金额: £{order_stats['Amount'].mean():.2f}")
    print(f"  中位数订单金额: £{order_stats['Amount'].median():.2f}")
    print(f"  最大订单金额: £{order_stats['Amount'].max():.2f}")
    print(f"  异常订单阈值 (>3*IQR): £{outlier_threshold:.2f}")
    
    if len(large_orders) > 0:
        print(f"\n超大额异常订单 (Top 10):")
        for _, row in large_orders.head(10).iterrows():
            print(f"  订单号: {row['InvoiceNo']}, 金额: £{row['Amount']:.2f}, "
                  f"客户ID: {int(row['CustomerID']) if pd.notna(row['CustomerID']) else 'N/A'}")
    
    # 检查异常商品（单价过高）
    high_price_items = df[df['UnitPrice'] > 100].sort_values('UnitPrice', ascending=False)
    print(f"\n高单价商品 (>£100): {len(high_price_items)} 条记录")
    if len(high_price_items) > 0:
        print(f"  单价最高的商品:\n{high_price_items[['StockCode', 'Description', 'UnitPrice', 'Quantity', 'Amount']].head(10)}")
    
    return large_orders, high_price_items

# 10. 主函数 - 执行所有分析
def main():
    """主函数"""
    file_path = 'OnlineRetail_Orders_Cleaned.csv'
    
    # 1. 加载数据
    df_raw = load_and_explore(file_path)
    
    # 2. 数据清洗
    df_clean = clean_data(df_raw)
    
    # 3. 销售额与订单量趋势分析
    daily_sales = sales_order_trend_analysis(df_clean)
    
    # 4. 小时销售分析
    hourly_stats = hourly_sales_analysis(df_clean)
    
    # 5. 热门商品分析
    top_products, top_revenue = product_analysis(df_clean)
    
    # 6. 退货订单分析
    returns = return_order_analysis(df_clean)
    
    # 7. 客户行为分析
    customer_stats = customer_behavior_analysis(df_clean)
    
    # 8. 国家销售分析
    country_sales = country_analysis(df_clean)
    
    # 9. 异常交易检测
    large_orders, high_price_items = anomaly_detection(df_clean)
    
    print("\n" + "="*60)
    print("分析完成！图表已保存至当前目录。")
    print("="*60)
    
    return {
        'df_clean': df_clean,
        'daily_sales': daily_sales,
        'hourly_stats': hourly_stats,
        'top_products': top_products,
        'customer_stats': customer_stats,
        'country_sales': country_sales
    }

# 执行主函数
if __name__ == "__main__":
    results = main()