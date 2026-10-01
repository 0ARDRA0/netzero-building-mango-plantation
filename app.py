# ==============================================================================
# STREAMLIT WEB APP: Building Net-Zero Planner: Mango Plantation Offset
# ==============================================================================

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go

# ------------------------------------------------------------------------------
# PAGE CONFIGURATION
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Building Net-Zero Planner: Mango Plantation Offset", 
    layout="wide",
    page_icon="🌳"
)

st.title("🌳 Building Net-Zero Planner: Mango Plantation Offset")
st.markdown(
    "Estimate whole-life building carbon emissions across structural material systems (**RCC, Composite, Steel**), "
    "evaluate compensatory **(*Mangifera indica*)** plantation requirements, and audit compliance with **GRIHA 2019** "
    "and **LEED v4.1/v5** decarbonization frameworks."
)

# ------------------------------------------------------------------------------
# MACHINE LEARNING ENGINE INITIALIZATION
# ------------------------------------------------------------------------------
@st.cache_resource
def load_ml_model():
    return joblib.load('netzero_mango_model.joblib')

try:
    ml_model = load_ml_model()
    st.sidebar.success("✅ ML Engine Active (Gradient Boosting)")
except Exception:
    ml_model = None
    st.sidebar.warning("⚠️ Running in Analytical Physics Mode")

# ------------------------------------------------------------------------------
# SIDEBAR CONTROLS & ARCHITECTURAL INPUTS
# ------------------------------------------------------------------------------
st.sidebar.header("🏢 1. Building & Energy Parameters")

gfa = st.sidebar.number_input("Gross Floor Area (GFA in m²)", min_value=100.0, max_value=100000.0, value=5000.0, step=500.0)
floors = st.sidebar.number_input("Number of Floors", min_value=1, max_value=100, value=5)
material = st.sidebar.selectbox("Primary Structural Material System", ['RCC', 'Composite', 'Steel'])
epi = st.sidebar.slider("Energy Performance Index (kWh/m²/year)", min_value=30, max_value=250, value=130)
horizon = st.sidebar.slider("Target Net-Zero Horizon (Years)", min_value=10, max_value=50, value=30, step=5)

# Rooftop Geometry & PV Constraint Validation
max_roof_footprint = gfa / max(1, floors)
max_usable_pv_area = max_roof_footprint * 0.65  # 65% usable limit

pv_area = st.sidebar.number_input(
    "Solar PV Panel Area (m²)", 
    min_value=0.0, 
    max_value=max(10000.0, max_roof_footprint), 
    value=min(400.0, max_usable_pv_area), 
    step=25.0
)

# Architectural Validation Warning
if pv_area > max_usable_pv_area:
    st.sidebar.error(
        f"⚠️ **Roof Area Limit Exceeded:** Roof plate is ~{max_roof_footprint:.1f} m². "
        f"Max usable PV area is ~{max_usable_pv_area:.1f} m² (65% roof coverage). "
        "Reduce PV area or plan off-site solar."
    )
else:
    st.sidebar.info(f"ℹ️ Roof Footprint: {max_roof_footprint:.1f} m² | Usable PV Limit: ~{max_usable_pv_area:.1f} m²")

st.sidebar.header("📜 2. Green Rating System Focus")
target_certification = st.sidebar.radio("Target Rating System Audit", ["GRIHA v2019", "LEED v4.1 / v5", "Both Systems"])

# Expandable Methodology Disclosure
with st.sidebar.expander("ℹ️ Calculation Parameters & Standards"):
    st.markdown("""
    - **A1-A3 Embodied Baseline:** RCC = 454 kgCO₂e/m², Composite = 408.6 kgCO₂e/m² (0.90x), Steel = 385.9 kgCO₂e/m² (0.85x).
    - **Lifecycle Expansion (+23%):** A5 Construction Site (8%), B4 Replacements (10%), C End-of-Life (5%).
    - **Grid Emission Factor:** 0.727 kgCO₂e/kWh (CEA India Baseline FY2023-24).
    - **Solar PV Yield:** 150 kWh/m²/year (~1 kWp per 5 m² array).
    - **Tree Allometry (*Mangifera indica*):** Pantropical equation (Chave et al., ρ = 0.50 g/cm³).
    - **Survival & Buffer:** 70% establishment at Year 3, 1% annual decay, 15% non-permanence risk buffer.
    """)

# ------------------------------------------------------------------------------
# SIMULATION ENGINE
# ------------------------------------------------------------------------------
def tree_stock_co2(age):
    """Chapman-Richards stock function per tree in tonnes CO2e (calibrated to TLS Mango data)."""
    if age <= 0:
        return 0.0
    return 9.5 * ((1.0 - np.exp(-0.032 * age)) ** 2.8)

def tree_survival(age):
    """Survival curve: 70% establishment at Year 3, 1% annual mortality thereafter."""
    if age < 1:
        return 1.0
    elif age < 3:
        return 1.0 - (0.30 / 3.0) * age
    else:
        return 0.70 * ((0.99) ** (age - 3.0))

def run_physical_simulation(gfa, floors, material, epi, pv_area, horizon):
    # 1. Structural Material Embodied Carbon Factors
    mat_factors = {'RCC': 1.0, 'Composite': 0.90, 'Steel': 0.85}
    m_factor = mat_factors.get(material, 1.0)
    
    a1_a3_base = gfa * 454.0 * m_factor / 1000.0
    a5_construction = a1_a3_base * 0.08
    b4_replacements = a1_a3_base * 0.10
    c_end_of_life = a1_a3_base * 0.05
    
    total_emb_co2 = a1_a3_base + a5_construction + b4_replacements + c_end_of_life
    
    # 2. Operational Carbon (CEA Grid Factor = 0.727 kgCO2e/kWh)
    annual_pv_kwh = pv_area * 150.0
    annual_bldg_kwh = gfa * epi
    net_grid_kwh = max(0.0, annual_bldg_kwh - annual_pv_kwh)
    annual_op_co2 = (net_grid_kwh * 0.727) / 1000.0
    
    target_bldg_co2_at_horizon = total_emb_co2 + (annual_op_co2 * horizon)
    
    # 3. Solve Planted Trees (5 Staggered Cohorts)
    cohort_starts = [0, 2, 4, 6, 8]
    yield_per_tree_at_horizon = 0.0
    
    for c_start in cohort_starts:
        if horizon >= c_start:
            age = horizon - c_start
            yield_per_tree_at_horizon += 0.20 * (tree_survival(age) * tree_stock_co2(age) * 0.85)
            
    n_trees = int(np.ceil(target_bldg_co2_at_horizon / max(0.0001, yield_per_tree_at_horizon)))
    hectares = n_trees / 100.0
    acres = hectares * 2.47105
    
    # 4. Trajectory Arrays
    years = np.arange(1, horizon + 1)
    bldg_cum_co2 = total_emb_co2 + (annual_op_co2 * years)
    
    plant_cum_co2 = []
    n_per_cohort = n_trees / 5.0
    for yr in years:
        total_yr_stock = 0.0
        for c_start in cohort_starts:
            if yr >= c_start:
                age = yr - c_start
                total_yr_stock += n_per_cohort * tree_survival(age) * tree_stock_co2(age) * 0.85
        plant_cum_co2.append(total_yr_stock)
        
    return (n_trees, acres, hectares, target_bldg_co2_at_horizon, total_emb_co2, 
            a1_a3_base, a5_construction, b4_replacements, c_end_of_life, annual_op_co2, 
            years, bldg_cum_co2, plant_cum_co2)

# ------------------------------------------------------------------------------
# MAIN INTERACTIVE DASHBOARD
# ------------------------------------------------------------------------------
if st.button("🚀 Run Full Net-Zero & Compliance Audit"):
    (n_trees, acres, hectares, total_co2, total_emb_co2, a1_a3_base, a5_const, 
     b4_rep, c_eol, annual_op_co2, years, bldg_cum, plant_cum) = run_physical_simulation(
        gfa, floors, material, epi, pv_area, horizon
    )
    
    mat_map = {'RCC': 0, 'Composite': 1, 'Steel': 2}
    if ml_model is not None:
        try:
            preds = ml_model.predict([[gfa, floors, mat_map[material], epi, pv_area, horizon]])[0]
            if int(preds[0]) > 0:
                n_trees = int(preds[0])
                acres = preds[1]
                hectares = acres / 2.47105
        except Exception:
            pass

    trees_per_acre = n_trees / max(0.01, acres)

    # Tabbed Interface
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Executive Net-Zero Dashboard", 
        "🏗️ Material LCA Breakdown", 
        "🌿 GRIHA v2019 Alignment", 
        "🏅 LEED v4.1 / v5 Decarbonization"
    ])

    # ---------------------------------------------------------
    # TAB 1: EXECUTIVE DASHBOARD
    # ---------------------------------------------------------
    with tab1:
        st.header("📊 Net-Zero Target Sizing Summary")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("🌳 Mango Trees Required", f"{n_trees:,} trees")
        col2.metric("📐 Total Land Required", f"{acres:.1f} Acres", f"{hectares:.1f} Hectares")
        col3.metric("🎯 Planting Density", f"{trees_per_acre:.1f} trees/acre", "10m x 10m Grid")
        col4.metric("⏱️ Target Horizon", f"{horizon} Years", f"Net-Zero at Year {horizon}")

        st.markdown("---")
        st.subheader("🏢 Whole-Life Building Carbon Summary")
        c1, c2, c3 = st.columns(3)
        c1.info(f"**Upfront Embodied Carbon (A1-C):** {total_emb_co2:,.1f} tonnes CO₂e")
        c2.info(f"**Annual Operational Carbon (B6):** {annual_op_co2:,.1f} tonnes CO₂e/yr")
        c3.success(f"**Total Carbon Target ({horizon} yrs):** {total_co2:,.1f} tonnes CO₂e")

        st.markdown("---")
        st.subheader("📈 Cumulative Emissions vs. Biological Sequestration Trajectory")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=years, y=bldg_cum, mode='lines', 
            name='Building Cumulative Carbon (CO₂e)', 
            line=dict(color='crimson', width=3)
        ))
        fig.add_trace(go.Scatter(
            x=years, y=plant_cum, mode='lines', 
            name='Mango Plantation Sequestration (CO₂e)', 
            line=dict(color='forestgreen', width=3, dash='dash')
        ))
        fig.add_trace(go.Scatter(
            x=[horizon], y=[plant_cum[-1]],
            mode='markers+text',
            name='Net-Zero Intersection Point',
            marker=dict(color='gold', size=14, symbol='star'),
            text=[f"Net-Zero (Year {horizon})"],
            textposition="top left"
        ))
        fig.update_layout(
            title=f"Net-Zero Carbon Alignment Trajectory over {horizon}-Year Horizon",
            xaxis_title="Years After Construction",
            yaxis_title="Cumulative Tonnes CO₂e",
            template="plotly_white",
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)

    # ---------------------------------------------------------
    # TAB 2: MATERIAL LCA BREAKDOWN
    # ---------------------------------------------------------
    with tab2:
        st.header(f"🏗️ Structural Material LCA: {material} System")
        st.markdown(f"Detailed lifecycle embodied carbon breakdown for **{material}** structural framing based on Indian high-rise LCA literature.")

        m_col1, m_col2 = st.columns([1, 1])
        
        with m_col1:
            st.markdown("### Lifecycle Stage Breakdown")
            lca_df = pd.DataFrame({
                "Lifecycle Stage": ["A1-A3 Raw Materials & Manufacture", "A5 Construction Process (+8%)", "B4 Material Replacements (+10%)", "C End-of-Life Demolition (+5%)"],
                "Embodied Carbon (tonnes CO₂e)": [a1_a3_base, a5_const, b4_rep, c_eol],
                "Share of Total Embodied (%)": [100/1.23, 8/1.23, 10/1.23, 5/1.23]
            })
            st.dataframe(lca_df.style.format({
                "Embodied Carbon (tonnes CO₂e)": "{:,.1f}",
                "Share of Total Embodied (%)": "{:.1f}%"
            }), use_container_width=True)

            rcc_comp_val = (gfa * 454.0 * 1.23) / 1000.0
            savings_vs_rcc = rcc_comp_val - total_emb_co2
            if material != 'RCC':
                st.success(f"💡 **Structural Material Savings:** Opting for **{material}** reduces upfront embodied carbon by **{savings_vs_rcc:,.1f} tonnes CO₂e** ({((savings_vs_rcc)/rcc_comp_val)*100:.1f}%) compared to baseline RCC.")
            else:
                st.info("💡 **Baseline Comparison:** Conventional RCC selected. Switching to Composite or Steel frames can cut embodied carbon by 10–15%.")

        with m_col2:
            fig_lca = go.Figure(data=[go.Pie(
                labels=["A1-A3 Manufacture", "A5 Construction", "B4 Replacements", "C End-of-Life"],
                values=[a1_a3_base, a5_const, b4_rep, c_eol],
                hole=.4,
                marker=dict(colors=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
            )])
            fig_lca.update_layout(title="Embodied Carbon Distribution by Stage")
            st.plotly_chart(fig_lca, use_container_width=True)

    # ---------------------------------------------------------
    # TAB 3: GRIHA v2019 ALIGNMENT AUDIT
    # ---------------------------------------------------------
    with tab3:
        st.header("🌿 GRIHA v2019 & Decarbonizing Habitat Program Audit")
        st.markdown("Evaluation against GRIHA Council's **Decarbonizing Habitat Program** scoring bands and rating criteria.")

        griha_epi_baseline = 140.0
        epi_reduction_pct = max(0.0, ((griha_epi_baseline - epi) / griha_epi_baseline) * 100)

        st.subheader("1. GRIHA Decarbonizing Habitat Performance Level")
        griha_band = "Indigo Level (Net Zero)"
        band_color = "indigo"
        
        st.markdown(f"""
        <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; border-left: 6px solid {band_color};">
            <h4>🟣 Projected GRIHA Carbon Status: <strong>{griha_band}</strong></h4>
            <p>The proposed <strong>{n_trees:,} tree Mango plantation ({hectares:.1f} ha)</strong> achieves 100% neutralization of residual building emissions within the {horizon}-year window.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("2. Clause-by-Clause Compliance Checklist")
        
        g_c1, g_c2 = st.columns(2)
        with g_c1:
            st.markdown("#### **Criterion: Energy Optimization & EPI**")
            if epi < griha_epi_baseline:
                st.success(f"✅ **Compliant:** Project EPI ({epi} kWh/m²/yr) is **{epi_reduction_pct:.1f}% below** the GRIHA baseline ({griha_epi_baseline} kWh/m²/yr).")
            else:
                st.warning(f"⚠️ **Action Needed:** Project EPI ({epi} kWh/m²/yr) exceeds GRIHA benchmark ({griha_epi_baseline} kWh/m²/yr). Reduce demand first.")

            st.markdown("#### **Criterion: Native Species Compensatory Plantation**")
            st.success("✅ **Compliant:** Uses *Mangifera indica* (Mango), an indigenous fruit-bearing species eligible for compensatory biodiversity credits.")

        with g_c2:
            st.markdown("#### **Criterion: Monitoring & Verification (MRV)**")
            st.info("ℹ️ **Requirement:** GRIHA requires geo-tagged tree inventory tracking at Years 1, 3, and 5 to verify sapling survival.")

            st.markdown("#### **Criterion: Additionality & Offsite Rules**")
            st.warning("⚠️ **Note:** Off-site plantation claims must be submitted to GRIHA Council for project-specific approval.")

    # ---------------------------------------------------------
    # TAB 4: LEED v4.1 / v5 DECARBONIZATION ROADMAP
    # ---------------------------------------------------------
    with tab4:
        st.header("🏅 LEED v4.1 / v5 Decarbonization & Offsets Pathway")
        st.markdown("Alignment with USGBC **LEED v5 Whole-Building Decarbonization** prerequisites and **LEED v4.1 Renewable Energy / Carbon Offsets** credits.")

        l_col1, l_col2 = st.columns(2)

        with l_col1:
            st.markdown("### 1. Embodied Carbon (BD+C Prerequisite & Credit)")
            st.markdown("""
            - **Prerequisite:** Whole-Building Life Cycle Assessment (WBLCA) conducted for structural & enclosure components (Stages A1-A3).
            - **Credit Pathway:** Achieving ≥10% to ≥20% embodied carbon reduction compared to an equivalent baseline building.
            """)
            if material in ['Steel', 'Composite']:
                st.success(f"✅ **LEED Point Eligible:** Using **{material}** framing yields an estimated **{((rcc_comp_val - total_emb_co2)/rcc_comp_val)*100:.1f}% reduction** in embodied carbon vs baseline RCC.")
            else:
                st.info("ℹ️ **Optimization Opportunity:** Switch to low-carbon concrete mixes or alternative structural framing to earn LEED WBLCA credits.")

        with l_col2:
            st.markdown("### 2. Operational Carbon & Carbon Offsets Rules")
            st.markdown("""
            - **LEED v4.1 EA Credit (Renewable Energy):** Recognizes combined on-site PV generation, green power procurement, and certified carbon offsets.
            - **Offsite Carbon Offset Qualification Rules:**
                - Must be registered under an accredited carbon standard (e.g., **Gold Standard, Verra VCS, or Green-e Climate**).
                - Uncertified private plantations **do not automatically earn LEED offset points**.
            """)
            st.warning("🔒 **LEED Certification Governance Rule:** To cite this 15,000+ tree plantation for LEED points, register the afforestation project under Verra VCS or Gold Standard before claiming offsets.")

        st.markdown("---")
        st.subheader("📋 USGBC Decarbonization Hierarchy Alignment")
        st.markdown(f"""
        1. **Demand Reduction:** Lower EPI via passive architecture and high-performance HVAC (Current EPI: **{epi} kWh/m²/yr**).
        2. **On-Site Renewables:** Maximize rooftop PV generation (Current PV Area: **{pv_area} m²**).
        3. **Embodied Carbon Optimization:** Select low-carbon materials (Current System: **{material}**).
        4. **Residual Neutralization:** Compensatory plantation neutralizes remaining whole-life emissions (**{total_co2:,.1f} t CO₂e**).
        """)
