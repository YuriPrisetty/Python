import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取 Excel 文件
df = pd.read_excel(r'D:\data1.xlsx')

# 定义变量分组
motivation_vars = [
    '减少孤独感', '提高效率', '降低成本', '拓展轻度人脉',
    '满足特定需求', '避免社交压力'
]

factor_vars = [
    '个人时间灵活性', '共同需求匹配度', '社交软件的便利性',
    '对浅社交的接受度', '过往搭子的社交体验', '生活工作环境'
]

all_vars = motivation_vars + factor_vars

# 1. 基本描述性统计分析
print("=" * 60)
print("描述性统计分析")
print("=" * 60)

desc_stats = df[all_vars].describe()
print(desc_stats.round(2))

# 2. 异常值检测和分析
print("\n" + "=" * 60)
print("异常值分析")
print("=" * 60)

def detect_outliers_iqr(data):
    """使用IQR方法检测异常值"""
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data < lower_bound) | (data > upper_bound)]
    return outliers, lower_bound, upper_bound

for var in all_vars:
    outliers, lower, upper = detect_outliers_iqr(df[var])
    print(f"{var}:")
    print(f"  异常值数量: {len(outliers)}")
    print(f"  异常值范围: {list(outliers.values)}")
    print(f"  正常值范围: [{lower:.2f}, {upper:.2f}]")
    print()

# 3. 偏度和峰度分析
print("=" * 60)
print("分布形态分析 (偏度和峰度)")
print("=" * 60)

distribution_analysis = []
for var in all_vars:
    skewness = stats.skew(df[var].dropna())
    kurtosis = stats.kurtosis(df[var].dropna())
    distribution_analysis.append({
        '变量': var,
        '偏度': skewness,
        '峰度': kurtosis,
        '分布形态': '正偏' if skewness > 0.5 else '负偏' if skewness < -0.5 else '近似对称',
        '峰度类型': '尖峰' if kurtosis > 0.5 else '低峰' if kurtosis < -0.5 else '常峰'
    })

dist_df = pd.DataFrame(distribution_analysis)
print(dist_df.round(3))

# 4. 可视化分析 - 创建综合分析图表
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('箱线图深度分析', fontsize=16, fontweight='bold')

# 4.1 所有变量的箱线图
sns.boxplot(data=df[all_vars], ax=axes[0,0])
axes[0,0].set_title('所有变量箱线图对比')
axes[0,0].set_xticklabels(all_vars, rotation=45, ha='right')

# 4.2 中位数比较
medians = df[all_vars].median().sort_values(ascending=False)
axes[0,1].bar(range(len(medians)), medians.values, color='skyblue')
axes[0,1].set_title('各变量中位数比较')
axes[0,1].set_xticks(range(len(medians)))
axes[0,1].set_xticklabels(medians.index, rotation=45, ha='right')
axes[0,1].set_ylabel('中位数')

# 4.3 数据分布范围（IQR）
iqr_data = []
for var in all_vars:
    Q1 = df[var].quantile(0.25)
    Q3 = df[var].quantile(0.75)
    iqr_data.append({'变量': var, 'IQR': Q3 - Q1, 'Q1': Q1, 'Q3': Q3})

iqr_df = pd.DataFrame(iqr_data).sort_values('IQR', ascending=False)
axes[0,2].bar(range(len(iqr_df)), iqr_df['IQR'], color='lightgreen')
axes[0,2].set_title('各变量IQR比较（数据离散程度）')
axes[0,2].set_xticks(range(len(iqr_df)))
axes[0,2].set_xticklabels(iqr_df['变量'], rotation=45, ha='right')
axes[0,2].set_ylabel('四分位距 (IQR)')

# 4.4 异常值比例
outlier_ratio = []
for var in all_vars:
    outliers, _, _ = detect_outliers_iqr(df[var])
    ratio = len(outliers) / len(df) * 100
    outlier_ratio.append({'变量': var, '异常值比例%': ratio})

outlier_df = pd.DataFrame(outlier_ratio).sort_values('异常值比例%', ascending=False)
axes[1,0].bar(range(len(outlier_df)), outlier_df['异常值比例%'], color='lightcoral')
axes[1,0].set_title('各变量异常值比例')
axes[1,0].set_xticks(range(len(outlier_df)))
axes[1,0].set_xticklabels(outlier_df['变量'], rotation=45, ha='right')
axes[1,0].set_ylabel('异常值比例 (%)')

# 4.5 偏度可视化
axes[1,1].bar(range(len(dist_df)), dist_df['偏度'], color='gold')
axes[1,1].set_title('各变量偏度比较')
axes[1,1].set_xticks(range(len(dist_df)))
axes[1,1].set_xticklabels(dist_df['变量'], rotation=45, ha='right')
axes[1,1].set_ylabel('偏度')
axes[1,1].axhline(y=0, color='red', linestyle='--', alpha=0.5)

# 4.6 变量相关性热图 plt.subplot(2, 3, 6)
correlation_matrix = df[all_vars].corr()
im = axes[1,2].imshow(correlation_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
axes[1,2].set_title('变量相关性热图')
axes[1,2].set_xticks(range(len(all_vars)))
axes[1,2].set_yticks(range(len(all_vars)))
axes[1,2].set_xticklabels(all_vars, rotation=45, ha='right')
axes[1,2].set_yticklabels(all_vars)

# 在每个方块中添加数值文本
for i in range(len(all_vars)):
    for j in range(len(all_vars)):
        # 获取相关系数值，并格式化为两位小数
        text = axes[1,2].text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                             ha="center", va="center", color="black", fontsize=8)

plt.colorbar(im, ax=axes[1,2])

plt.tight_layout()
plt.show()

# 5. 分组比较分析
print("\n" + "=" * 60)
print("动机变量 vs 影响因素变量 分组比较")
print("=" * 60)

motivation_stats = df[motivation_vars].describe()
factor_stats = df[factor_vars].describe()

print("动机变量组统计:")
print(f"平均中位数: {motivation_stats.loc['50%'].mean():.2f}")
print(f"平均标准差: {motivation_stats.loc['std'].mean():.2f}")

print("\n影响因素变量组统计:")
print(f"平均中位数: {factor_stats.loc['50%'].mean():.2f}")
print(f"平均标准差: {factor_stats.loc['std'].mean():.2f}")

# 6. 关键发现总结
print("\n" + "=" * 60)
print("关键发现总结")
print("=" * 60)

# 找出评分最高的变量
highest_median_var = df[all_vars].median().idxmax()
highest_median_value = df[all_vars].median().max()

# 找出评分最低的变量
lowest_median_var = df[all_vars].median().idxmin()
lowest_median_value = df[all_vars].median().min()

# 找出变异最大的变量
highest_std_var = df[all_vars].std().idxmax()
highest_std_value = df[all_vars].std().max()

print(f"评分最高的变量: {highest_median_var} (中位数: {highest_median_value:.2f})")
print(f"评分最低的变量: {lowest_median_var} (中位数: {lowest_median_value:.2f})")
print(f"变异最大的变量: {highest_std_var} (标准差: {highest_std_value:.2f})")

# 检查是否有明显的偏态分布
significant_skew = dist_df[abs(dist_df['偏度']) > 1]
if not significant_skew.empty:
    print(f"\n明显偏态的变量:")
    for _, row in significant_skew.iterrows():
        print(f"  {row['变量']}: 偏度 = {row['偏度']:.2f} ({row['分布形态']})")
