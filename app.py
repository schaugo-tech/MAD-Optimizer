import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from mpl_toolkits.mplot3d import Axes3D

# 定义回归方程
def tmj_stress(mp, vo):
    """颞下颌关节盘应力回归方程"""
    return -11.108 + 0.106*mp + 0.360*vo

def pdl_stress(mp, vo):
    """牙周膜应力回归方程"""
    return -9.423 + 0.117*mp + 0.534*vo

# 页面配置
st.set_page_config(page_title="MAD参数优化系统", layout="wide")
st.title("下颌前移矫治器优化决策系统")

# 侧边栏参数设置
with st.sidebar:
    st.header("🛠️ 优化参数设置")
    col1, col2 = st.columns(2)
    with col1:
        max_tmj = st.slider("最大关节盘应力(MPa)", 3.0, 8.0, 5.0)
        min_mp = st.slider("最小前伸量(%)", 40, 70, 50)
        weight_tmj = st.slider("关节盘权重", 0.0, 1.0, 0.5)
    with col2:
        max_pdl = st.slider("最大牙周膜应力(kPa)", 3.0, 9.0, 6.0)
        min_vo = st.slider("最小开口量(mm)", 3, 7, 4)
        weight_pdl = st.slider("牙周膜权重", 0.0, 1.0, 0.5)

# 优化目标函数
def objective(x, w1, w2):
    mp, vo = x
    return w1*tmj_stress(mp, vo) + w2*pdl_stress(mp, vo)

# 约束条件
constraints = [
    {'type': 'ineq', 'fun': lambda x: max_tmj - tmj_stress(x, x)},
    {'type': 'ineq', 'fun': lambda x: max_pdl - pdl_stress(x, x)},
    {'type': 'ineq', 'fun': lambda x: x - min_mp},
    {'type': 'ineq', 'fun': lambda x: x - min_vo}
]

# 执行优化
result = minimize(
    objective,
    x0=[min_mp, min_vo],
    args=(weight_tmj, weight_pdl),
    method='SLSQP',
    bounds=[(min_mp, 70), (min_vo, 7)],
    constraints=constraints
)

# 结果显示
if result.success:
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("📊 优化结果")
        st.metric("最佳前伸量(MP)", f"{result.x:.1f}%")
        st.metric("最佳开口量(VO)", f"{result.x:.1f}mm")
        st.divider()
        st.metric("关节盘应力", f"{tmj_stress(*result.x):.2f} MPa", 
                 delta_color="off")
        st.metric("牙周膜应力", f"{pdl_stress(*result.x):.2f} kPa",
                 delta_color="off")

    with col2:
        st.header("🌐 三维应力分布")
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')
        
        # 生成网格数据
        mp_range = np.linspace(min_mp, 70, 30)
        vo_range = np.linspace(min_vo, 7, 30)
        MP, VO = np.meshgrid(mp_range, vo_range)
        
        # 计算综合应力
        Stress = weight_tmj*tmj_stress(MP, VO) + weight_pdl*pdl_stress(MP, VO)
        
        # 绘制曲面
        surf = ax.plot_surface(MP, VO, Stress, cmap='viridis', alpha=0.8)
        ax.scatter(result.x, result.x, result.fun, 
                  c='red', s=100, label='Optimal Point')
        
        # 设置坐标轴
        ax.set_xlabel('MP (%)', labelpad=12)
        ax.set_ylabel('VO (mm)', labelpad=12)
        ax.set_zlabel('综合应力', labelpad=12)
        ax.view_init(elev=30, azim=-45)
        
        # 添加颜色条
        cbar = fig.colorbar(surf, shrink=0.5, aspect=10)
        cbar.set_label('加权应力值', rotation=270, labelpad=15)
        
        st.pyplot(fig)

else:
    st.error("⚠️ 未找到可行解，请调整约束条件！")

# 临床建议
st.divider()
st.header("📋 临床建议")
if result.success:
    if result.x >= 65:
        st.warning("前伸量超过65%，建议密切监测关节健康")
    if result.x >= 6:
        st.warning("开口量超过6mm，建议检查牙周膜适应性")
    
    if tmj_stress(*result.x) < 4 and pdl_stress(*result.x) < 5:
        st.success("✅ 当前参数在安全范围内")
    else:
        st.info("ℹ️ 参数接近临界值，建议定期复查")

# 调试信息（可选）
# st.write("优化详细信息：", result)