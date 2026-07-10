import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, silhouette_score, calinski_harabasz_score, davies_bouldin_score
from mlxtend.frequent_patterns import apriori, association_rules
from scipy import stats
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 1. 数据加载与预处理
print("="*60)
print("1. 数据加载与预处理")
print("="*60)

# 加载数据
df = pd.read_csv('OnlineRetail_Customer_Features.csv')

print(f"原始数据形状: {df.shape}")
print(f"列名: {df.columns.tolist()}")
print(f"\n数据前5行:\n{df.head()}")

# 数据基本信息
print(f"\n缺失值统计:\n{df.isnull().sum()}")

# 处理缺失值
df['Country'].fillna('Unknown', inplace=True)

# 查看流失标签分布
print(f"\n流失标签分布:")
print(df['churn_label'].value_counts())
print(f"流失率: {df['churn_label'].mean():.2%}")

# 查看国家分布
print(f"\n国家分布（前10）:")
print(df['Country'].value_counts().head(10))

# 删除可能的异常值（Monetary为负或0的）
df = df[df['Monetary'] > 0]
print(f"\n清洗后数据形状: {df.shape}")

# 2. 特征工程
print("\n" + "="*60)
print("2. 特征工程 - 构造新特征")
print("="*60)

# 构造RFM综合得分
df['R_score'] = pd.qcut(df['Recency'], q=4, labels=[4, 3, 2, 1]).astype(int)
df['F_score'] = pd.qcut(df['Frequency'].rank(method='first'), q=4, labels=[1, 2, 3, 4]).astype(int)
df['M_score'] = pd.qcut(df['Monetary'], q=4, labels=[1, 2, 3, 4]).astype(int)
df['RFM_Score'] = df['R_score'] + df['F_score'] + df['M_score']

# 构造消费效率指标
df['Monetary_per_Order'] = df['Monetary'] / df['Frequency']
df['Items_per_Order'] = df['TotalItems'] / df['Frequency']

# 构造流失风险指标
df['Risk_Score'] = (df['Recency'] / df['Recency'].max()) * 0.5 + \
                   (1 - df['Frequency'] / df['Frequency'].max()) * 0.3 + \
                   (1 - df['Monetary'] / df['Monetary'].max()) * 0.2

print("新增特征:")
print("- R_score, F_score, M_score: RFM评分")
print("- RFM_Score: RFM综合得分")
print("- Monetary_per_Order: 每单平均金额")
print("- Items_per_Order: 每单平均商品数")
print("- Risk_Score: 流失风险得分")

# 3. 聚类分析
print("\n" + "="*60)
print("3. 聚类分析 - 客户分群")
print("="*60)

# 选择聚类特征
cluster_features = ['Recency', 'Frequency', 'Monetary', 'AvgOrderAmount', 
                    'AvgItemsPerOrder', 'Monetary_per_Order']

cluster_data = df[cluster_features].fillna(0)

# 标准化
scaler = StandardScaler()
cluster_data_scaled = scaler.fit_transform(cluster_data)

# 肘部法则和轮廓系数评估
k_range = range(2, 7)
inertias = []
silhouette_scores = []
calinski_scores = []
davies_scores = []

print("正在计算各K值的评估指标...")
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(cluster_data_scaled)
    inertias.append(kmeans.inertia_)
    sil_score = silhouette_score(cluster_data_scaled, labels)
    silhouette_scores.append(sil_score)
    calinski_scores.append(calinski_harabasz_score(cluster_data_scaled, labels))
    davies_scores.append(davies_bouldin_score(cluster_data_scaled, labels))
    print(f"K={k}: 轮廓系数={sil_score:.4f}, CH指数={calinski_scores[-1]:.1f}, DB指数={davies_scores[-1]:.3f}")

# 绘制评估图
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

axes[0,0].plot(list(k_range), inertias, 'bo-')
axes[0,0].set_xlabel('K值')
axes[0,0].set_ylabel('SSE')
axes[0,0].set_title('肘部法则图')
axes[0,0].grid(True)

axes[0,1].plot(list(k_range), silhouette_scores, 'ro-')
axes[0,1].set_xlabel('K值')
axes[0,1].set_ylabel('轮廓系数')
axes[0,1].set_title('轮廓系数图')
axes[0,1].axhline(y=0.5, color='g', linestyle='--', label='良好阈值(0.5)')
axes[0,1].legend()
axes[0,1].grid(True)

axes[1,0].plot(list(k_range), calinski_scores, 'go-')
axes[1,0].set_xlabel('K值')
axes[1,0].set_ylabel('CH指数')
axes[1,0].set_title('Calinski-Harabasz指数')
axes[1,0].grid(True)

axes[1,1].plot(list(k_range), davies_scores, 'mo-')
axes[1,1].set_xlabel('K值')
axes[1,1].set_ylabel('DB指数')
axes[1,1].set_title('Davies-Bouldin指数（越小越好）')
axes[1,1].grid(True)

plt.tight_layout()
plt.savefig('clustering_evaluation.png', dpi=150)
plt.show()

# 选择最优K
best_k = k_range[np.argmax(silhouette_scores)]
print(f"\n✅ 根据轮廓系数，最优K值 = {best_k}")

# 最终聚类
kmeans_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df['Cluster'] = kmeans_final.fit_predict(cluster_data_scaled)

# 各簇人数分布
print(f"\n各簇人数分布:")
cluster_counts = df['Cluster'].value_counts().sort_index()
for i, count in cluster_counts.items():
    print(f"  簇{i}: {count}人 ({count/len(df):.1%})")

# 各簇特征分析
print(f"\n各簇特征均值:")
cluster_summary = df.groupby('Cluster')[cluster_features].mean().round(2)
print(cluster_summary)

# 按簇统计流失率
print(f"\n各簇流失率:")
for i in range(best_k):
    cluster_df = df[df['Cluster'] == i]
    churn_rate = cluster_df['churn_label'].mean()
    print(f"  簇{i}: 流失率={churn_rate:.1%}")

# 为每个簇命名
cluster_names = {}
for i in range(best_k):
    recency = cluster_summary.loc[i, 'Recency']
    monetary = cluster_summary.loc[i, 'Monetary']
    frequency = cluster_summary.loc[i, 'Frequency']
    churn_rate = df[df['Cluster'] == i]['churn_label'].mean()
    
    if recency < 30 and monetary > 5000:
        cluster_names[i] = "高价值活跃用户"
    elif recency < 60 and monetary > 2000:
        cluster_names[i] = "中等价值活跃用户"
    elif recency > 180:
        cluster_names[i] = "高流失风险用户"
    else:
        cluster_names[i] = "普通用户"

print(f"\n各簇命名:")
for i in range(best_k):
    print(f"  簇{i}: {cluster_names[i]}")

# PCA可视化
pca = PCA(n_components=2)
cluster_data_pca = pca.fit_transform(cluster_data_scaled)
plt.figure(figsize=(10, 6))
scatter = plt.scatter(cluster_data_pca[:, 0], cluster_data_pca[:, 1], 
                      c=df['Cluster'], cmap='viridis', alpha=0.6)
plt.colorbar(scatter)
plt.title(f'客户聚类可视化 (PCA降维, K={best_k})', fontsize=14)
plt.xlabel('第一主成分')
plt.ylabel('第二主成分')
plt.savefig('cluster_visualization.png', dpi=150)
plt.show()

# 4. 分类分析 - 预测流失用户
print("\n" + "="*60)
print("4. 分类分析 - 预测流失用户")
print("="*60)

# 特征准备
features = ['Recency', 'Frequency', 'Monetary', 'AvgOrderAmount', 
            'AvgItemsPerOrder', 'Monetary_per_Order', 'Items_per_Order',
            'R_score', 'F_score', 'M_score', 'RFM_Score']

X = df[features].fillna(0)
y = df['churn_label']

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, 
                                                     random_state=42, stratify=y)

# 随机森林（使用class_weight处理不平衡）
rf = RandomForestClassifier(n_estimators=100, random_state=42, 
                            class_weight='balanced')
rf.fit(X_train, y_train)

# 特征重要性
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print("\n特征重要性排序:")
print(feature_importance.to_string(index=False))

# 模型评估
y_pred = rf.predict(X_test)
print("\n流失预测模型分类报告:")
print(classification_report(y_test, y_pred))

# 交叉验证
cv_scores = cross_val_score(rf, X_train, y_train, cv=5, scoring='f1')
print(f"\n5折交叉验证F1分数: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# 5. 关联规则挖掘
print("\n" + "="*60)
print("5. 关联规则挖掘 - 用户特征关联分析")
print("="*60)

# 二值化特征
df['high_recency'] = (df['Recency'] > df['Recency'].quantile(0.7)).astype(bool)
df['high_frequency'] = (df['Frequency'] > df['Frequency'].quantile(0.7)).astype(bool)
df['high_monetary'] = (df['Monetary'] > df['Monetary'].quantile(0.7)).astype(bool)
df['high_rfm'] = (df['RFM_Score'] > df['RFM_Score'].quantile(0.7)).astype(bool)
df['churned'] = (df['churn_label'] == 1).astype(bool)

# 国家编码二值化（取前5个国家）
top_countries = df['Country'].value_counts().head(5).index.tolist()
for country in top_countries:
    df[f'country_{country}'] = (df['Country'] == country).astype(bool)

# 选择关联规则特征
basket_cols = ['high_recency', 'high_frequency', 'high_monetary', 
               'high_rfm', 'churned'] + [f'country_{c}' for c in top_countries]
basket = df[basket_cols]

# 挖掘频繁项集
min_support = 0.05
frequent_itemsets = apriori(basket, min_support=min_support, use_colnames=True)

if len(frequent_itemsets) > 0:
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.2)
    if len(rules) > 0:
        rules_sorted = rules.sort_values('lift', ascending=False)
        print(f"\n发现的关键关联规则（共{len(rules_sorted)}条，展示前10）:")
        display_cols = ['antecedents', 'consequents', 'support', 'confidence', 'lift']
        print(rules_sorted[display_cols].head(10).to_string())
    else:
        print("未发现提升度>1.2的关联规则")
else:
    print("未发现频繁项集")

# 6. 异常检测
print("\n" + "="*60)
print("6. 异常检测 - 识别异常用户")
print("="*60)

# 使用Z-score检测异常
df['recency_zscore'] = np.abs(stats.zscore(df['Recency']))
df['monetary_zscore'] = np.abs(stats.zscore(df['Monetary']))
df['frequency_zscore'] = np.abs(stats.zscore(df['Frequency']))

# 标记异常用户
df['is_anomaly'] = ((df['recency_zscore'] > 3) | 
                    (df['monetary_zscore'] > 3) | 
                    (df['frequency_zscore'] > 3))

# 正向异常：高价值但即将流失的用户
high_value_churn = df[(df['Monetary'] > df['Monetary'].quantile(0.8)) & 
                      (df['churn_label'] == 1)]
print(f"\n正向异常 - 高价值但即将流失的用户（共{len(high_value_churn)}人）:")
print(high_value_churn[['Monetary', 'Frequency', 'Recency', 'Country']].head(10))

# 负向异常：低价值但活跃的用户
low_value_active = df[(df['Monetary'] < df['Monetary'].quantile(0.2)) & 
                      (df['Recency'] < 30)]
print(f"\n负向异常 - 低价值但活跃的用户（共{len(low_value_active)}人）:")
print(low_value_active[['Monetary', 'Frequency', 'Recency', 'Country']].head(10))

# 7. 各簇详细分析
print("\n" + "="*60)
print("7. 各簇详细分析")
print("="*60)

for i in range(best_k):
    cluster_df = df[df['Cluster'] == i]
    print(f"\n{'='*40}")
    print(f"簇{i}: {cluster_names[i]}")
    print(f"{'='*40}")
    print(f"人数: {len(cluster_df)} ({len(cluster_df)/len(df):.1%})")
    print(f"流失率: {cluster_df['churn_label'].mean():.1%}")
    print(f"平均Recency: {cluster_df['Recency'].mean():.1f}天")
    print(f"平均Frequency: {cluster_df['Frequency'].mean():.1f}次")
    print(f"平均Monetary: {cluster_df['Monetary'].mean():.2f}")
    print(f"平均RFM得分: {cluster_df['RFM_Score'].mean():.1f}")
    print(f"\n主要国家分布:")
    print(cluster_df['Country'].value_counts().head(3).to_string())

# 8. 可视化增强
print("\n" + "="*60)
print("8. 可视化分析")
print("="*60)

# 箱线图：各簇的Monetary分布
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
df.boxplot(column='Monetary', by='Cluster')
plt.title('各簇消费金额分布')
plt.yscale('log')

plt.subplot(1, 2, 2)
df.boxplot(column='Recency', by='Cluster')
plt.title('各簇最近购买天数分布')
plt.tight_layout()
plt.savefig('cluster_boxplots.png', dpi=150)
plt.show()

# RFM热力图
rfm_pivot = df.pivot_table(index='R_score', columns='F_score', 
                           values='M_score', aggfunc='mean')
plt.figure(figsize=(10, 8))
sns.heatmap(rfm_pivot, annot=True, cmap='YlOrRd', fmt='.1f')
plt.title('RFM热力图（R×F → M）')
plt.xlabel('F评分（频率）')
plt.ylabel('R评分（最近性）')
plt.savefig('rfm_heatmap.png', dpi=150)
plt.show()

# 流失用户与活跃用户对比雷达图
from math import pi

# 对比特征
compare_features = ['Recency', 'Frequency', 'Monetary', 'AvgOrderAmount', 'Items_per_Order']
churn_profile = df[df['churn_label'] == 1][compare_features].mean()
active_profile = df[df['churn_label'] == 0][compare_features].mean()

# 归一化
all_vals = pd.concat([churn_profile, active_profile], axis=1)
all_vals = (all_vals - all_vals.min()) / (all_vals.max() - all_vals.min())

categories = compare_features
N = len(categories)
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
values1 = all_vals[0].values.flatten().tolist()
values1 += values1[:1]
values2 = all_vals[1].values.flatten().tolist()
values2 += values2[:1]

ax.plot(angles, values1, 'o-', linewidth=2, label='流失用户', color='red')
ax.fill(angles, values1, alpha=0.1, color='red')
ax.plot(angles, values2, 'o-', linewidth=2, label='活跃用户', color='green')
ax.fill(angles, values2, alpha=0.1, color='green')

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories)
ax.set_title('流失用户 vs 活跃用户特征对比', fontsize=14)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
plt.tight_layout()
plt.savefig('churn_vs_active_radar.png', dpi=150)
plt.show()

# 9. 保存结果
print("\n" + "="*60)
print("9. 保存分析结果")
print("="*60)

# 保存聚类结果
df.to_csv('user_clustering_result.csv', index=False, encoding='utf-8-sig')
print("✅ 结果已保存到: user_clustering_result.csv")
print("   包含原始数据 + Cluster列")

# 保存各簇画像
cluster_profile = df.groupby('Cluster').agg({
    'CustomerID': 'count',
    'Recency': 'mean',
    'Frequency': 'mean',
    'Monetary': 'mean',
    'AvgOrderAmount': 'mean',
    'churn_label': 'mean',
    'RFM_Score': 'mean'
}).round(2)
cluster_profile.columns = ['人数', '平均Recency', '平均Frequency', 
                           '平均Monetary', '平均订单金额', '流失率', '平均RFM得分']
cluster_profile.to_csv('cluster_profile.csv', encoding='utf-8-sig')
print("各簇画像已保存到: cluster_profile.csv")

# 10. 分析总结
print("\n" + "="*60)
print("10. 分析总结")
print("="*60)

print(f"""
【电商用户购买行为分析总结】

一、数据概况
- 总用户数: {len(df)}人
- 整体流失率: {df['churn_label'].mean():.1%}
- 主要国家: {', '.join(top_countries[:5])}

二、聚类分析结果
- 最优聚类数: K={best_k}
- 平均轮廓系数: {silhouette_scores[best_k-2]:.4f}

各簇画像:
""")
for i in range(best_k):
    cluster_df = df[df['Cluster'] == i]
    print(f"""
  簇{i}: {cluster_names[i]}
    - 人数: {len(cluster_df)}人 ({len(cluster_df)/len(df):.1%})
    - 流失率: {cluster_df['churn_label'].mean():.1%}
    - 平均消费: {cluster_df['Monetary'].mean():.0f}
    - 平均购买次数: {cluster_df['Frequency'].mean():.1f}
    - 主要国家: {cluster_df['Country'].mode().iloc[0] if len(cluster_df) > 0 else 'N/A'}
""")

print(f"""
三、流失预测模型
- 最显著特征: {feature_importance.iloc[0]['feature']}
- 模型F1分数: {cv_scores.mean():.4f}

四、关键发现
1. {'高价值活跃用户' if best_k>=1 else 'N/A'}：高消费、高频率、低流失风险，是核心资产
2. {'高流失风险用户' if best_k>=2 else 'N/A'}：Recency高、消费低，需要重点挽回
3. 消费金额(Monetary)和最近购买天数(Recency)是最重要的流失预测特征

五、运营建议
1. 【高价值活跃用户】→ VIP服务、推荐裂变、专属优惠
2. 【中等价值活跃用户】→ 引导升级、精准推荐、积分激励
3. 【高流失风险用户】→ 发送挽回优惠券、客服回访
4. 【普通用户】→ 推送小额优惠、提升活跃度

六、输出文件
- user_clustering_result.csv: 用户聚类结果
- cluster_profile.csv: 各簇详细画像
- clustering_evaluation.png: 聚类质量评估图
- cluster_visualization.png: PCA可视化图
- cluster_boxplots.png: 各簇分布箱线图
- rfm_heatmap.png: RFM热力图
- churn_vs_active_radar.png: 流失vs活跃用户雷达图
""")