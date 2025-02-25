import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from io import BytesIO

# 初始化会话状态
if 'calculate' not in st.session_state:
    st.session_state.calculate = False

# 定义回归方程
## def tmj_stress(mp, vo):
##     """颞下颌关节盘应力回归方程（单位：MPa）"""
##     return ( -7.036 + 0.179*mp + -0.106*vo)
## 
## def pdl_stress(mp, vo):
##     """牙周膜应力回归方程（单位：kPa）"""
##     return (-8.575 + 0.209*mp + 0.316*vo)

def tmj_stress(mp, vo):
    """颞下颌关节盘应力（MPa）"""
    return 9.978 -0.479*mp + 0.840*vo + 0.006*mp**2 -0.019*mp*vo + 0.021*vo**2

def pdl_stress(mp, vo):
    """牙周膜应力（kPa）"""
    return 4.034 -0.193*mp + 0.091*vo + 0.003*mp**2 + 0.014*mp*vo -0.061*vo**2

# 页面配置
st.set_page_config(page_title="MAD智能决策系统", layout="wide")
st.title("下颌前移矫治器优化决策系统")

# ================= 侧边栏参数设置 =================
with st.sidebar:
    st.header("⚙️ 参数设置")
    
    # 安全阈值
    st.subheader("安全阈值")
    max_tmj = st.slider("最大关节盘应力(MPa)", 5.0, 20.0, 10.0, key='max_tmj')
    max_pdl = st.slider("最大牙周膜应力(kPa)", 8.0, 20.0, 10.0, key='max_pdl')
    
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
        eps = st.number_input("调整步长参数", 1e-1, 1e0, 5e-1, format="%.0e")
    
    # 计算触发按钮
    # if st.button("🚀 开始计算", help="点击启动优化计算", use_container_width=True):
    if st.button("🚀 开始计算", help="点击启动优化计算"):
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
        {'type': 'ineq', 'fun': lambda x: st.session_state.max_tmj - tmj_stress(x[0], x[1])},
        {'type': 'ineq', 'fun': lambda x: st.session_state.max_pdl - pdl_stress(x[0], x[1])},
        {'type': 'ineq', 'fun': lambda x: x[0] - st.session_state.min_mp},
        {'type': 'ineq', 'fun': lambda x: x[1] - st.session_state.min_vo}
    ]

    # 执行优化
    result = minimize(
        objective,
        x0=[st.session_state.min_mp, st.session_state.min_vo],
        method=algorithm,
        bounds=[(st.session_state.min_mp, 70), (st.session_state.min_vo, 7)],
        constraints=constraints,
        options={'maxiter': max_iter, 'ftol': tolerance, 'eps': eps}
    )

    if result.success:
        # ================= 结果显示 =================
        col1, col2 = st.columns(2)
        
        with col1:
            st.header("📊 优化结果")
            st.metric("最佳前伸量(MP)", f"{result.x[0]:.1f}%", 
                     delta="关节盘安全阈值" if result.x[0] < 65 else "注意：接近高风险区")
            st.metric("最佳开口量(VO)", f"{result.x[1]:.1f}mm", 
                     delta="牙周膜安全阈值" if result.x[1] < 6 else "注意：接近高风险区")
            # st.divider()
            st.markdown("---") 
            
            # 应力指标
            tmj_value = tmj_stress(*result.x)
            pdl_value = pdl_stress(*result.x)
            st.metric("关节盘应力", f"{tmj_value:.2f} MPa", 
                     delta="安全" if tmj_value < 10 else "注意：接近危险值", delta_color="normal")
            st.metric("牙周膜应力", f"{pdl_value:.2f} kPa",
                     delta="安全" if pdl_value < 10 else "注意：接近危险值", delta_color="normal")

        with col2:
        
            # 患者最大前伸量和开口量（根据临床测量）
            max_mp = 10  # 示例值，需替换为实际测量值(mm)
            max_vo = 10  # 示例值，需替换为实际测量值(mm)
            
            # 正交实验设计参数
            mp_percentages = [0.5, 0.55, 0.6, 0.65, 0.7]  # 前伸量比例
            vo_values = [3, 4, 5, 6, 7]                   # 垂直开口量(mm)
            
            # 生成所有组合
            MP, VO = np.meshgrid(
                np.array(mp_percentages) * max_mp, 
                np.array(vo_values)
            )
       
        
            st.header("🌐 三维应力分布")
            
            # 创建包含两个子图的 Figure
            fig = plt.figure(figsize=(20, 8))
            
            # ================= 关节盘应力子图 =================
            ax1 = fig.add_subplot(121, projection='3d')
            surf1 = ax1.plot_surface(MP, VO, tmj_stress(MP, VO), cmap='viridis', alpha=0.8)
            ax1.scatter(max_mp*0.01*result.x[0], result.x[1], tmj_stress(result.x[0], result.x[1]), c='red', s=100, marker='*')
            ax1.set_title('颞下颌关节盘应力 (MPa)')
            
            # ================= 牙周膜应力子图 =================
            ax2 = fig.add_subplot(122, projection='3d')
            surf2 = ax2.plot_surface(MP, VO, pdl_stress(MP, VO), cmap='plasma', alpha=0.8)
            ax2.scatter(max_mp*0.01*result.x[0], result.x[1], pdl_stress(result.x[0], result.x[1]), c='blue', s=100, marker='^')
            ax2.set_title('牙周膜应力 (kPa)')
            
            # 统一设置子图属性
            for ax in [ax1, ax2]:
                ax.set_xlabel('MP (%)', labelpad=12)
                ax.set_ylabel('VO (mm)', labelpad=12)
                ax.view_init(elev=30, azim=-45)
            
            # 添加颜色条
            fig.colorbar(surf1, ax=ax1, shrink=0.5, label='应力值 (MPa)')
            fig.colorbar(surf2, ax=ax2, shrink=0.5, label='应力值 (kPa)')
            
            st.pyplot(fig)
                
        # ================= 临床建议 =================
        # st.divider()
        st.markdown("---") 
        st.header("📋 临床建议")
        
        if result.x[0] >= 65:
            st.warning("⚠️ 前伸量超过65%，建议密切监测关节健康")
        if result.x[1] >= 6:
            st.warning("⚠️ 开口量超过6mm，建议检查牙周膜适应性")
        
        if tmj_value < 10 and pdl_value < 10:
            st.success("✅ 当前参数在安全范围内")
        else:
            st.info("ℹ️ 参数接近临界值，建议定期复查")

    else:
        # st.error("⚠️ 未找到可行解，请调整约束条件！")
        st.error(f"""
            ⚠️ **未找到可行解，请调整约束条件！**  
            可能原因：  
            1. 约束过紧（当前阈值：{max_tmj}MPa/{max_pdl}kPa）  
            2. 最小范围过高（MP≥{min_mp}%, VO≥{min_vo}mm）  
            3. 权重失衡（当前：关节盘{weight_tmj:.1f}/牙周膜{weight_pdl:.1f}）  

            建议调整策略：  
            ▶ 放宽最大应力至{max_tmj+2}MPa/{max_pdl+2}kPa  
            ▶ 降低最小范围至MP≥{max(min_mp-5,40)}%, VO≥{max(min_vo-1,3)}mm  
            ▶ 调整权重分配（推荐：关节盘{weight_tmj-0.2:.1f}/牙周膜{weight_pdl+0.2:.1f}）
            """)

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
    c.drawString(100, 700, f"最佳前伸量: {result.x[0]:.1f}%")
    c.drawString(100, 680, f"最佳开口量: {result.x[1]:.1f}mm")
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
        # use_container_width=True
    )
    
# ================= 系统信息 =================
with st.sidebar:
    st.markdown("---") 
    st.sidebar.markdown("""
    **系统版本**  
    `v2.1.3 | 生物力学优化引擎  
    `©2024 空军军医大学数字医学中心
    """)
    # st.divider()
    
