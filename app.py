import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from io import BytesIO

# 初始化会话状态
if 'calculate' not in st.session_state:
    st.session_state.calculate = False

# 定义回归方程
def tmj_stress(mp, vo):
    """颞下颌关节盘应力回归方程（单位：MPa）"""
    return (9.978 + -0.479*mp + 0.840*vo + 
            0.006*mp**2 + -0.019*mp*vo + 0.021*vo**2)

def pdl_stress(mp, vo):
    """牙周膜应力回归方程（单位：kPa）"""
    return (4.034 + -0.193*mp + 0.091*vo + 
            0.003*mp**2 + 0.014*mp*vo + -0.061*vo**2)

# 页面配置
st.set_page_config(page_title="MAD智能决策系统", layout="wide")
st.title("下颌前移矫治器优化决策系统")

# ================= 侧边栏参数设置 =================
with st.sidebar:
    st.header("⚙️ 参数设置")
    
    # 安全阈值
    st.subheader("安全阈值")
    max_tmj = st.slider("最大关节盘应力(MPa)", 0.0, 20.0, 10.0, key='max_tmj')
    max_pdl = st.slider("最大牙周膜应力(kPa)", 0.0, 20.0, 10.0, key='max_pdl')
    
    # 最小范围
    st.subheader("最小允许值")
    min_mp = st.slider("最小前伸量(%)", 40, 70, 50, key='min_mp')
    min_vo = st.slider("最小开口量(mm)", 3, 7, 4, key='min_vo')
    
    # 权重设置
    st.subheader("权重分配")
    weight_tmj = st.slider("关节盘权重", 0.0, 1.0, 0.5, key='weight_tmj')
    weight_pdl = st.slider("牙周膜权重", 0.0, 1.0, 0.5, key='weight_pdl')
    
    # 专家选项
    with st.expander("🔧 专家设置"):
        algorithm = st.selectbox("优化算法", ["SLSQP", "COBYLA", "trust-constr"], index=0)
        tolerance = st.number_input("优化容差", 1e-8, 1e-2, 1e-6, format="%.0e")
        max_iter = st.number_input("最大迭代次数", 50, 500, 200)
    
    # 计算触发按钮
    if st.button("🚀 开始计算", use_container_width=True):
        st.session_state.calculate = True

# ================= 主内容区域 =================
if st.session_state.calculate:
    # 优化目标函数
    def objective(x):
        mp, vo = x
        return (st.session_state.weight_tmj * tmj_stress(mp, vo) 
                + st.session_state.weight_pdl * pdl_stress(mp, vo))

    # 约束条件
    constraints = [
        {'type': 'ineq', 'fun': lambda x: st.session_state.max_tmj - tmj_stress(x, x)},
        {'type': 'ineq', 'fun': lambda x: st.session_state.max_pdl - pdl_stress(x, x)},
        {'type': 'ineq', 'fun': lambda x: x - st.session_state.min_mp},
        {'type': 'ineq', 'fun': lambda x: x - st.session_state.min_vo}
    ]

    # 执行优化
    result = minimize(
        objective,
        x0=[st.session_state.min_mp, st.session_state.min_vo],
        method=algorithm,
        bounds=[(st.session_state.min_mp, 70), (st.session_state.min_vo, 7)],
        constraints=constraints,
        options={'maxiter': max_iter, 'ftol': tolerance}
    )

    if result.success:
        # ================= 结果显示 =================
        col1, col2 = st.columns(2)
        
        with col1:
            st.header("📊 优化结果")
            st.metric("最佳前伸量(MP)", f"{result.x:.1f}%", 
                     delta="关节盘安全阈值" if result.x < 65 else "注意：接近高风险区")
            st.metric("最佳开口量(VO)", f"{result.x:.1f}mm", 
                     delta="牙周膜安全阈值" if result.x < 6 else "注意：接近高风险区")
            st.divider()
            
            # 应力指标
            tmj_value = tmj_stress(*result.x)
            pdl_value = pdl_stress(*result.x)
            st.metric("关节盘应力", f"{tmj_value:.2f} MPa", 
                     delta="安全" if tmj_value < 10 else "注意：接近危险值", delta_color="normal")
            st.metric("牙周膜应力", f"{pdl_value:.2f} kPa",
                     delta="安全" if pdl_value < 10 else "注意：接近危险值", delta_color="normal")

        with col2:
            st.header("🌐 三维应力分布")
            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_subplot(111, projection='3d')
            
            # 生成网格数据
            mp_range = np.linspace(st.session_state.min_mp, 70, 30)
            vo_range = np.linspace(st.session_state.min_vo, 7, 30)
            MP, VO = np.meshgrid(mp_range, vo_range)
            
            # 计算综合应力
            Stress = (st.session_state.weight_tmj * tmj_stress(MP, VO) 
                    + st.session_state.weight_pdl * pdl_stress(MP, VO))
            
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

        # ================= 临床建议 =================
        st.divider()
        st.header("📋 临床建议")
        
        if result.x >= 65:
            st.warning("⚠️ 前伸量超过65%，建议密切监测关节健康")
        if result.x >= 6:
            st.warning("⚠️ 开口量超过6mm，建议检查牙周膜适应性")
        
        if tmj_value < 10 and pdl_value < 10:
            st.success("✅ 当前参数在安全范围内")
        else:
            st.info("ℹ️ 参数接近临界值，建议定期复查")

    else:
        st.error("⚠️ 未找到可行解，请调整约束条件！")

else:
    st.info("👆 请在侧边栏设置参数后点击【开始计算】")

# ================= 报告生成 =================
def generate_report():
    """生成PDF报告"""
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import A4
    
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    
    # 报告内容
    c.drawString(100, 750, "MAD矫治器优化报告")
    c.drawString(100, 700, f"最佳前伸量: {result.x:.1f}%")
    c.drawString(100, 680, f"最佳开口量: {result.x:.1f}mm")
    c.drawString(100, 660, f"关节盘应力: {tmj_value:.2f} MPa")
    c.drawString(100, 640, f"牙周膜应力: {pdl_value:.2f} kPa")
    
    c.save()
    buffer.seek(0)
    return buffer

# 下载按钮
if st.session_state.calculate and result.success:
    report = generate_report()
    st.download_button(
        label="📥 下载完整报告",
        data=report,
        file_name="MAD_优化报告.pdf",
        mime="application/pdf",
        use_container_width=True
    )
