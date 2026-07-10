import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 四、Streamlit可视化应用
st.set_page_config(layout="wide", page_title="专利选股策略回测")

# 4.1 侧边栏参数设置
st.sidebar.header("参数设置")

# 加载数据
@st.cache_data
def load_data():
    return pd.read_parquet('D:/Limmo/Python数分/实验四/return.parquet')

df = load_data()
df['交易日'] = pd.to_datetime(df['交易日'])
strategies = df['strategy'].unique().tolist()

# 策略选择
selected_strategies = st.sidebar.multiselect("选择策略", strategies, default=strategies)

# 日期范围
min_date = df['交易日'].min()
max_date = df['交易日'].max()
date_range = st.sidebar.date_input("日期范围", [min_date, max_date])

# 动态绘图
enable_animation = st.sidebar.checkbox("启用动态绘图", False)
speed = st.sidebar.slider("动画速度 (天/帧)", 1, 30, 5) if enable_animation else 1

# 无风险利率
risk_free = st.sidebar.slider("无风险利率 (年化)", 0.0, 5.0, 3.0) / 100

# 过滤数据
df_filtered = df[df['strategy'].isin(selected_strategies)]
df_filtered = df_filtered[(df_filtered['交易日'] >= pd.to_datetime(date_range[0])) & 
                         (df_filtered['交易日'] <= pd.to_datetime(date_range[1]))]

# 4.2 主图：累积收益率曲线
st.subheader("累积收益率对比")

fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                    row_heights=[0.7, 0.3], 
                    subplot_titles=("净值曲线", "超额收益曲线"))

# 基准数据
benchmark_data = df_filtered[df_filtered['strategy'] == df_filtered['strategy'].iloc[0]]
if not benchmark_data.empty:
    fig.add_trace(go.Scatter(
        x=benchmark_data['交易日'],
        y=benchmark_data['zz2000_nav'],
        mode='lines',
        name='中证2000基准',
        line=dict(color='gray', dash='dash')
    ), row=1, col=1)

# 各策略曲线
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
for i, strategy in enumerate(selected_strategies):
    strat_data = df_filtered[df_filtered['strategy'] == strategy]
    if not strat_data.empty:
        fig.add_trace(go.Scatter(
            x=strat_data['交易日'],
            y=strat_data['port_nav'],
            mode='lines',
            name=strategy,
            line=dict(color=colors[i % len(colors)])
        ), row=1, col=1)
        
        # 超额收益
        excess = strat_data['port_nav'] / strat_data['zz2000_nav']
        fig.add_trace(go.Scatter(
            x=strat_data['交易日'],
            y=excess,
            mode='lines',
            name=f'{strategy}超额',
            line=dict(color=colors[i % len(colors)], dash='dot')
        ), row=2, col=1)

fig.update_layout(height=600, showlegend=True)
st.plotly_chart(fig, use_container_width=True)

# 4.3 绩效指标展示
st.subheader("绩效指标")

metrics_data = []
for strategy in selected_strategies:
    strat_data = df_filtered[df_filtered['strategy'] == strategy]
    if not strat_data.empty:
        ret = strat_data['port_ret']
        ret_ann = (1 + ret.mean()) ** 252 - 1
        vol_ann = ret.std() * np.sqrt(252)
        sharpe = (ret.mean() - risk_free/252) / ret.std() * np.sqrt(252)
        nav = (1 + ret).cumprod()
        max_dd = (nav / nav.cummax() - 1).min()
        win_rate = (ret > 0).mean()
        
        metrics_data.append({
            '策略': strategy,
            '年化收益率': f'{ret_ann:.2%}',
            '年化波动率': f'{vol_ann:.2%}',
            '夏普比率': f'{sharpe:.4f}',
            '最大回撤': f'{max_dd:.2%}',
            '胜率': f'{win_rate:.2%}'
        })

metrics_df = pd.DataFrame(metrics_data)
st.dataframe(metrics_df, use_container_width=True)

# 4.4 风险分析面板
st.subheader("风险分析")

col1, col2 = st.columns(2)

with col1:
    # 回撤分析
    st.write("回撤分析")
    for strategy in selected_strategies:
        strat_data = df_filtered[df_filtered['strategy'] == strategy]
        if not strat_data.empty:
            ret = strat_data['port_ret']
            nav = (1 + ret).cumprod()
            peak = nav.cummax()
            dd = nav / peak - 1
            
            fig_dd = go.Figure()
            fig_dd.add_trace(go.Scatter(
                x=strat_data['交易日'],
                y=dd,
                fill='tozeroy',
                name=f'{strategy}回撤'
            ))
            max_dd_val = dd.min()
            max_dd_date = dd.idxmin()
            fig_dd.add_hline(y=max_dd_val, line_dash="dash", 
                           annotation_text=f"最大回撤: {max_dd_val:.2%}")
            st.plotly_chart(fig_dd, use_container_width=True)

with col2:
    # 收益分布
    st.write("收益分布")
    for strategy in selected_strategies:
        strat_data = df_filtered[df_filtered['strategy'] == strategy]
        if not strat_data.empty:
            ret = strat_data['port_ret']
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(
                x=ret,
                nbinsx=50,
                name=strategy,
                opacity=0.7
            ))
            # 添加正态分布曲线
            x_range = np.linspace(ret.min(), ret.max(), 100)
            mu = ret.mean()
            sigma = ret.std()
            y_norm = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_range - mu) / sigma) ** 2)
            fig_hist.add_trace(go.Scatter(
                x=x_range, y=y_norm * len(ret) * (ret.max()-ret.min())/50,
                mode='lines', name='正态分布', line=dict(color='red', dash='dash')
            ))
            st.plotly_chart(fig_hist, use_container_width=True)