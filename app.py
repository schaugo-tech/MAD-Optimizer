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
    return 3.562 + -0.183*mp + 0.153*vo + 0.003*mp**2 + 0.014*mp*vo + -0.062*vo**2

# 页面配置
st.set_page_config(page_title="MAD生物力学设计系统", layout="wide")
st.title("基于颞下颌关节及下前牙生物力学研究的下颌前移矫治器设计系统")

# ================= 侧边栏参数设置 =================
with st.sidebar:
    st.header("🛠️ 参数设置")
    st.subheader("⚙️ 临床诊断")
    max_mp_val = st.slider("下颌最大前伸量（mm）", 6.0, 18.0, 13.5, key='max_mp_val')
    max_vo_val = st.slider("最大允许开口量（mm）", 4.0, 10.0, 8.0, key='max_vo_val')
    # 权重设置
    st.subheader("⚖️ 治疗策略")
    # 滑动条组件
    # 初始化会话状态
    if 'weight_clinic' not in st.session_state:
        st.session_state.weight_clinic = 0.7  # 默认临床疗效权重70%
        st.session_state.weight_safety = 0.3  # 默认安全性权重30%
    weight_value = st.slider(
        label="治疗策略倾向",
        min_value=0.0,
        max_value=1.0,
        value=0.8,
        step=0.1,
        format="疗效 → %.1f ← 安全",
        help="向左滑动优先考虑治疗效果，向右滑动优先考虑治疗安全性"
    )
    # 动态更新权重
    st.session_state.weight_clinic = 1 - weight_value
    st.session_state.weight_safety = weight_value

    # st.markdown(f"""
    #     **实时权重分配**
    #     ```python
    #     临床疗效权重 = {st.session_state.weight_clinic:.2f}
    #     生物安全权重 = {st.session_state.weight_safety:.2f}
    #     ```
    #     """)
    # 专家选项
    with st.expander("🔧 专家设置"):
        # 安全阈值
        st.subheader("安全阈值")
        max_tmj = st.slider("最大关节盘应力(MPa)", 1.0, 20.0, 7.0, key='max_tmj')
        max_pdl = st.slider("最大牙周膜应力(kPa)", 1.0, 20.0, 10.0, key='max_pdl')

        # 最小范围
        st.subheader("最小允许值")
        min_mp = st.slider("最小前伸量(%)", 10, 80, 40, key='min_mp')
        min_vo = st.slider("最小开口量(mm)", 3, 8, 3, key='min_vo')

        st.subheader("权重分配")
        weight_tmj = st.slider("关节盘权重", 0.0, 1.0, 1.0, key='weight_tmj')
        weight_pdl = st.slider("牙周膜权重", 0.0, 1.0, 0.5, key='weight_pdl')

        st.subheader("算法参数")
        algorithm = st.selectbox("优化算法", ["SLSQP", "COBYLA"], index=0)
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
        return (st.session_state.weight_safety *
                (st.session_state.weight_tmj * tmj_stress(mp, vo) + st.session_state.weight_pdl * pdl_stress(mp, vo))
                + st.session_state.weight_clinic *
                (70-mp))

    # 约束条件
    constraints = [
        {'type': 'ineq', 'fun': lambda x: st.session_state.max_tmj - tmj_stress(x[0], x[1])},
        {'type': 'ineq', 'fun': lambda x: st.session_state.max_pdl - pdl_stress(x[0], x[1])},
        {'type': 'ineq', 'fun': lambda x: st.session_state.max_vo_val - x[1]},
        {'type': 'ineq', 'fun': lambda x: x[0] - st.session_state.min_mp},
        {'type': 'ineq', 'fun': lambda x: x[1] - st.session_state.min_vo}
    ]

    # 执行优化
    result = minimize(
        objective,
        x0 = [st.session_state.min_mp, st.session_state.min_vo],
        method=algorithm,
        bounds=[(st.session_state.min_mp, 70), (st.session_state.min_vo, st.session_state.max_vo_val)],
        constraints=constraints,
        options={'maxiter': max_iter, 'ftol': tolerance, 'eps': eps}
    )

    if result.success:
        # ================= 结果显示 =================
        col1, col2 = st.columns(2)
        
        with col1:
            st.header("📊 优化结果")
            st.metric("最佳前伸量(mm)", f"{st.session_state.max_mp_val * 0.01 * result.x[0]:.1f}",
                     delta="处于安全阈值范围内" if result.x[0] < 60 else "注意：接近高风险区")
            st.metric("最佳开口量(mm)", f"{result.x[1]:.1f}",
                     delta="处于安全阈值范围内" if result.x[1] < 5 else "注意：接近高风险区")
            # st.divider()
            st.markdown("---") 
            
            # 应力指标
            tmj_value = tmj_stress(*result.x)
            pdl_value = pdl_stress(*result.x)
            st.metric("关节盘应力", f"{tmj_value:.2f} MPa", 
                     delta="安全" if tmj_value < 5 else "注意：接近危险值", delta_color="normal")
            st.metric("牙周膜应力", f"{pdl_value:.2f} kPa",
                     delta="安全" if pdl_value < 4.7 else "注意：接近危险值", delta_color="normal")

        with col2:

            # 正交实验设计参数
            mp_percentages = [40, 45, 50, 55, 60, 65, 70]  # 前伸量比例
            vo_values = [3, 4, 5, 6, 7]            # 垂直开口量(mm)
            
            # 生成所有组合
            MP, VO = np.meshgrid(
                np.array(mp_percentages),
                np.array(vo_values)
            )

            st.header("🌐 生物力学应力分析")
            plt.rcParams.update({'font.size': 16, 'axes.titlesize': 16})
            # 创建包含两个子图的 Figure
            fig = plt.figure(figsize=(20, 8))
            
            # ================= 关节盘应力子图 =================
            ax1 = fig.add_subplot(121, projection='3d')
            surf1 = ax1.plot_surface(MP*0.01*st.session_state.max_mp_val, VO, tmj_stress(MP, VO), cmap='viridis', alpha=0.8)
            ax1.scatter(st.session_state.max_mp_val*0.01*result.x[0], result.x[1], tmj_stress(result.x[0], result.x[1]), c='red', s=300, marker='*')
            ax1.set_title('TMJ Disc')
            
            # ================= 牙周膜应力子图 =================
            ax2 = fig.add_subplot(122, projection='3d')
            surf2 = ax2.plot_surface(MP*0.01*st.session_state.max_mp_val, VO, pdl_stress(MP, VO), cmap='plasma', alpha=0.8)
            ax2.scatter(st.session_state.max_mp_val*0.01*result.x[0], result.x[1], pdl_stress(result.x[0], result.x[1]), c='blue', s=300, marker='^')
            ax2.set_title('Mandibular Anterior PDL')
            
            # 统一设置子图属性
            for ax in [ax1, ax2]:
                ax.set_xlabel('MP (mm)', labelpad=12)
                ax.set_ylabel('VO (mm)', labelpad=12)
                ax.view_init(elev=30, azim=-95)
            
            # 添加颜色条
            fig.colorbar(surf1, ax=ax1, shrink=0.5, label='Peak Stress (MPa)')
            fig.colorbar(surf2, ax=ax2, shrink=0.5, label='Peak Stress (kPa)')
            
            st.pyplot(fig)
                
        # ================= 临床建议 =================
        # st.divider()
        st.markdown("---") 
        st.header("📋 临床建议")
        
        if result.x[0] >= 60:
            st.warning("⚠️ 前伸量超过60%，建议密切监测关节健康")
        if result.x[1] >= 5:
            st.warning("⚠️ 开口量超过5mm，建议检查牙周膜适应性")
        
        if tmj_value < 5 and pdl_value < 4.7:
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
            3. 权重失衡（当前：治疗效果{st.session_state.weight_clinic:.1f}/治疗安全性{st.session_state.weight_safety:.1f}）  

            建议调整策略：  
            ▶ 放宽最大应力至{max_tmj+2}MPa/{max_pdl+2}kPa  
            ▶ 降低最小范围至MP≥{max(min_mp-5,40)}%, VO≥{max(min_vo-1,3)}mm  
            ▶ 调整权重分配（推荐：治疗效果{st.session_state.weight_clinic-0.2:.1f}/牙周膜{st.session_state.weight_safety+0.2:.1f}）
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
    c.drawString(100, 750, "MAD矫治器优化设计报告")
    c.drawString(100, 700, f"最佳前伸量: {st.session_state.max_mp_val*0.01*result.x[0]:.1f}mm")
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
        file_name="MAD_优化设计报告.pdf",
        mime="application/pdf",
        # use_container_width=True
    )
    
# ================= 系统信息 =================
with st.sidebar:
    st.markdown("---") 
    st.sidebar.markdown("""
    **系统版本**  
    `v2.1.3 | 生物力学优化引擎` \n ©2024 空军军医大学 甘淡
    """)
    # st.divider()
    
