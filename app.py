import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import shap
import matplotlib.pyplot as plt
import math

# ==========================================
# 1. INITIALIZATION & CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Living Expense Estimation Engine",
    page_icon="🏙️",
    layout="wide"
)

st.markdown("""
    <style>
    .stApp { background-color: #0E1117; color: #E2E8F0; }
    .stSidebar { background-color: #1A202C; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. CACHED RESOURCE LOADING
# ==========================================
@st.cache_resource(show_spinner=False)
def load_model(model_path: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at '{model_path}'.")
    return joblib.load(model_path)

@st.cache_data(show_spinner=False)
def load_data(csv_path: str):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found at '{csv_path}'.")
    df = pd.read_csv(csv_path)
    if 'zip_code' in df.columns:
        df['zip_code'] = df['zip_code'].astype(str).str.zfill(5)
    return df

# --- HELPER: ROBUST IMAGE LOADER FOR ML TAB ---
def render_image(filename: str, caption: str, base_dir: str):
    """Searches common directories for the image and renders it, or shows an explicit error."""
    search_paths = [
        os.path.join(base_dir, filename),                                    # Root dir
        os.path.join(base_dir, "Phase03_figures", filename),                 # Subfolder
        os.path.join(base_dir, "extracted_figures", "Phase03_figures", filename) # Zip extraction path
    ]
    
    for path in search_paths:
        if os.path.exists(path):
            st.image(path, caption=caption)
            return
            
    # If the loop finishes without returning, the image wasn't found anywhere.
    st.error(f"⚠️ **Image Missing:** Could not find `{filename}`. Please place this image in the same folder as `app.py` or inside a `Phase03_figures` folder.")

# ==========================================
# 3. SPATIAL ARBITRAGE MATH ENGINE
# ==========================================
def calc_geo_distance_miles(lat1, lon1, lat2, lon2):
    if pd.isna(lat1) or pd.isna(lat2): 
        return 999.0
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    a = math.sin((lat2 - lat1)/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1)/2)**2
    return 2 * math.asin(math.sqrt(a)) * 3958.8

def find_nearest_arbitrage_zip(df, target_row):
    target_zip = target_row['zip_code']
    target_twia = int(target_row.get('twia_mandatory', 0))
    target_mud = int(target_row.get('water_mud_flag', 0))
    target_flood = float(target_row.get('pct_area_100yr_flood', 0.0))
    target_lat = target_row.get('latitude', 0.0)
    target_lon = target_row.get('longitude', 0.0)

    candidates = df[df['zip_code'] != target_zip].copy()
    if candidates.empty: 
        return target_row, "none"

    candidates['geo_dist_miles'] = candidates.apply(
        lambda r: calc_geo_distance_miles(target_lat, target_lon, r.get('latitude', 0.0), r.get('longitude', 0.0)), axis=1
    )

    if target_twia == 1:
        ideal = candidates[(candidates.get('twia_mandatory', 0) == 0) & (candidates.get('water_mud_flag', 0) == 0)]
        if not ideal.empty: 
            return ideal.sort_values(by=['geo_dist_miles']).iloc[0], "cost_governance"
    if target_mud == 1:
        no_mud = candidates[(candidates.get('water_mud_flag', 0) == 0) & (candidates.get('twia_mandatory', 0) == 0)]
        if not no_mud.empty: 
            return no_mud.sort_values(by=['geo_dist_miles']).iloc[0], "mud_only"
    if target_flood > 5.0:
        lower_flood = candidates[(candidates.get('twia_mandatory', 0) == 0) & (candidates.get('pct_area_100yr_flood', target_flood) < target_flood)]
        if not lower_flood.empty: 
            return lower_flood.sort_values(by=['geo_dist_miles']).iloc[0], "flood_mitigation"

    return target_row, "optimal"

# ==========================================
# 4. DETERMINISTIC TWIA SURCHARGE ENGINE
# ==========================================
def calculate_twia_surcharge(twia_flag, multiplier):
    return 125.0 * multiplier if twia_flag == 1 else 0.0

# ==========================================
# 5. MAIN APPLICATION
# ==========================================
def main():
    st.title("🏙️ Non-Mortgage Living Expense Estimation Engine")
    st.markdown("### Harris, Fort Bend, and Galveston Counties")

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_FILENAME = os.path.join(BASE_DIR, "xgboost_cost_model.joblib")
    CSV_FILENAME = os.path.join(BASE_DIR, "master_living_costs_clustered.csv")

    try:
        model = load_model(MODEL_FILENAME)
        df = load_data(CSV_FILENAME)
    except Exception as e:
        st.error(f"Critical Initialization Error: {e}")
        st.stop()

    # --- UI SIDEBAR ---
    st.sidebar.header("1. Property Parameters")
    zip_query = st.sidebar.text_input("Enter 5-Digit ZIP Code", value="77598", max_chars=5).strip()
    sqft_input = st.sidebar.number_input("Property Square Footage", min_value=500, max_value=10000, value=1500, step=50)
    occupants_input = st.sidebar.slider("Household Occupants", min_value=1, max_value=6, value=2)

    st.sidebar.markdown("---")
    st.sidebar.header("2. Housing Governance & HOA")
    include_hoa = st.sidebar.toggle("Enable Subdivision / HOA Fees", value=False, help="Toggle on if targeting housing within a managed HOA community.")

    st.sidebar.markdown("---")
    st.sidebar.header("3. Household Budget Profiles")

    tier_display_names = [
        "Thrifty (SNAP Baseline / Maximum Savings)",
        "Low-Cost (Budget-Conscious / Sales & Coupons)",
        "Moderate-Cost (Standard Middle-Class Average)",
        "Liberal (Premium / Specialty & Organic Focus)"
    ]
    selected_tier_name = st.sidebar.selectbox("USDA Grocery Plan Tier", tier_display_names, index=2)
    view_mode = st.sidebar.radio("Display Horizon", ["Monthly ($/mo)", "Annualized ($/yr)"], horizontal=True)
    time_multiplier = 12 if view_mode == "Annualized ($/yr)" else 1

    # --- LOOKUP PIPELINE ---
    selected_row = df[df['zip_code'] == zip_query]
    if selected_row.empty:
        st.error("❌ ZIP Code not found in the regional dataset.")
        st.stop()
    data = selected_row.iloc[0]

    city_name = data.get('city_name', 'Houston Area')
    county_name = data.get('county_name', 'Unknown')
    archetype = data.get('cluster_archetype_name', 'Unclassified Region')
    pop_growth_pct = float(data.get('five_year_pop_growth_pct', 0.0))

    st.sidebar.markdown("---")
    st.sidebar.header("4. ZIP Classification Profile")
    st.sidebar.markdown(f"""
    <div style="font-size: 14px; line-height: 1.5;">
    <b>City:</b> {city_name}<br>
    <b>Archetype:</b> {archetype}<br><br>
    <b>5-Year Population Growth Rate:</b> {pop_growth_pct:+.1f}%<br>
    <i>This ZIP code exhibits stable municipal infrastructure with high-density urban core characteristics.</i>
    </div>
    """, unsafe_allow_html=True)

    # --- ML DATA PREPARATION ---
    features = {
        'sqft': sqft_input,
        'occupants': occupants_input,
        'pct_area_100yr_flood': float(data.get('pct_area_100yr_flood', 0.0)),
        'twia_mandatory': int(data.get('twia_mandatory', 0)),
        'water_mud_flag': int(data.get('water_mud_flag', 0)),
        'houston_cpi_food_scalar': float(data.get('houston_cpi_food_scalar', 1.0)),
        'effective_tax_rate_pct': float(data.get('effective_tax_rate_pct', 2.2)),
        'five_year_pop_growth_pct': pop_growth_pct
    }

    expected_features = ['sqft', 'occupants', 'pct_area_100yr_flood', 'twia_mandatory',
                         'water_mud_flag', 'houston_cpi_food_scalar', 'effective_tax_rate_pct',
                         'five_year_pop_growth_pct']
    input_df = pd.DataFrame([features])[expected_features]

    # --- PREDICTION EXECUTION ---
    with st.spinner("Running XGBoost Inference..."):
        predictions = model.predict(input_df)[0]

    ml_food = predictions[0]
    ml_elec = predictions[1]
    ml_water = predictions[2]
    ml_gas = predictions[3]
    ml_hoa = predictions[4]

    tier_scalars = {
        "Thrifty (SNAP Baseline / Maximum Savings)": 0.75,
        "Low-Cost (Budget-Conscious / Sales & Coupons)": 0.88,
        "Moderate-Cost (Standard Middle-Class Average)": 1.00,
        "Liberal (Premium / Specialty & Organic Focus)": 1.20
    }
    grocery_multiplier = tier_scalars[selected_tier_name]

    pred_food = (ml_food * grocery_multiplier) * time_multiplier
    pred_utils = (ml_elec + ml_water + ml_gas) * time_multiplier
    pred_hoa = (ml_hoa * time_multiplier) if include_hoa else 0.0
    pred_twia = calculate_twia_surcharge(features['twia_mandatory'], time_multiplier)

    total_cost = pred_food + pred_utils + pred_hoa + pred_twia

    # --- ARBITRAGE EXECUTION ---
    best_alt_data, arbitrage_type = find_nearest_arbitrage_zip(df, data)
    alt_zip = best_alt_data['zip_code']
    alt_city = best_alt_data.get('city_name', 'Nearby')
    dist_miles = calc_geo_distance_miles(data.get('latitude'), data.get('longitude'), best_alt_data.get('latitude'), best_alt_data.get('longitude'))

    is_optimal_target = (alt_zip == zip_query) or (arbitrage_type == "optimal")

    alt_features = features.copy()
    alt_features.update({
        'pct_area_100yr_flood': float(best_alt_data.get('pct_area_100yr_flood', 0.0)),
        'twia_mandatory': int(best_alt_data.get('twia_mandatory', 0)),
        'water_mud_flag': int(best_alt_data.get('water_mud_flag', 0)),
        'houston_cpi_food_scalar': float(best_alt_data.get('houston_cpi_food_scalar', 1.0)),
        'effective_tax_rate_pct': float(best_alt_data.get('effective_tax_rate_pct', features['effective_tax_rate_pct']))
    })
    alt_df = pd.DataFrame([alt_features])[expected_features]
    alt_preds = model.predict(alt_df)[0]

    alt_pred_food = (alt_preds[0] * grocery_multiplier) * time_multiplier
    alt_pred_utils = (alt_preds[1] + alt_preds[2] + alt_preds[3]) * time_multiplier
    alt_pred_hoa = (alt_preds[4] * time_multiplier) if include_hoa else 0.0
    alt_pred_twia = calculate_twia_surcharge(int(alt_features['twia_mandatory']), time_multiplier)
    
    alt_total_cost = alt_pred_food + alt_pred_utils + alt_pred_hoa + alt_pred_twia

    # --- RENDER DASHBOARD HEADER ---
    if features['twia_mandatory'] == 1:
        st.warning(f"🌊 **Coastal Zone Designation ({city_name}):** Located within TWIA coastal policy eligibility boundary.")
    if features['pct_area_100yr_flood'] > 20.0:
        st.error(f"⚠️ **High Flood Hazard Alert ({city_name}):** {features['pct_area_100yr_flood']:.1f}% of this ZIP area lies within the 100-year floodplain.")

    st.subheader(f"Cost Breakdown for ZIP Code: {zip_query} | {city_name} ({archetype})")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Occupancy Cost", f"${total_cost:,.2f}")
    with col2:
        st.metric(f"Food ({selected_tier_name.split(' ')[0]})", f"${pred_food:,.2f}")
    with col3:
        st.metric("Utilities", f"${pred_utils:,.2f}")
    with col4:
        st.metric("TWIA Surcharge", f"${pred_twia:,.2f}" if features['twia_mandatory'] == 1 else "$0.00")
    with col5:
        st.metric("5-Yr Pop Growth", f"{pop_growth_pct:+.1f}%")

    st.markdown("---")

    # --- DEEP DIVE TABS ---
    tab1, tab2, tab3, tab6, tab4, tab5 = st.tabs([
        "💰 Itemized Breakdown & Comparison",
        "🌊 Flood & MUD Risk",
        "⚡ Utility Grid Detail",
        "🎯 Budget Recommender",
        "📊 ML Explainability & Diagnostics",
        "🗺️ Phase 03 Project Roadmap"
    ])

    with tab1:
        st.subheader("Target vs. Alternative Component Comparison")
        if is_optimal_target:
            st.success(f"🌟 **Optimal Non-Mortgage Governance:** ZIP {zip_query} already operates on optimal municipal utilities without MUD taxes or coastal surcharges. No alternative migration required.")
            comp_rows = ["Groceries & Food", "Utilities (Power/Water/Gas)"]
            target_vals = [f"${pred_food:,.2f}", f"${pred_utils:,.2f}"]
            if features['twia_mandatory'] == 1:
                comp_rows.append("Mandatory TWIA Coastal Surcharge")
                target_vals.append(f"${pred_twia:,.2f}")
            if include_hoa:
                comp_rows.append("Optional Subdivision Dues / HOA")
                target_vals.append(f"${pred_hoa:,.2f}")
            comp_rows.extend(["Total Estimated Cost", "5-Year Pop Growth Rate", "100-Yr Flood Exposure", "MUD Governance"])
            target_vals.extend([f"${total_cost:,.2f}", f"{pop_growth_pct:+.1f}%", f"{features['pct_area_100yr_flood']:.1f}%", "Suburban MUD" if features['water_mud_flag']==1 else "Standard Municipal"])

            comp_df = pd.DataFrame({
                "Cost / Metric": comp_rows,
                f"Target & Optimal: {zip_query} ({city_name})": target_vals
            })
            st.table(comp_df)
        else:
            st.info(f"💡 **Arbitrage Found:** Evaluating nearby **ZIP {alt_zip} ({alt_city})** — ~{dist_miles:.1f} miles away. Total Cost: ${alt_total_cost:,.2f}")
            comp_rows = ["Groceries & Food", "Utilities (Power/Water/Gas)"]
            target_vals = [f"${pred_food:,.2f}", f"${pred_utils:,.2f}"]
            alt_vals = [f"${alt_pred_food:,.2f}", f"${alt_pred_utils:,.2f}"]

            if features['twia_mandatory'] == 1 or alt_features['twia_mandatory'] == 1:
                comp_rows.append("Mandatory TWIA Coastal Surcharge")
                target_vals.append(f"${pred_twia:,.2f}")
                alt_vals.append(f"${alt_pred_twia:,.2f}")

            if include_hoa:
                comp_rows.append("Optional Subdivision Dues / HOA")
                target_vals.append(f"${pred_hoa:,.2f}")
                alt_vals.append(f"${alt_pred_hoa:,.2f}")

            comp_rows.extend(["Total Estimated Cost", "5-Year Pop Growth Rate", "100-Yr Flood Exposure", "MUD Governance"])
            target_vals.extend([f"${total_cost:,.2f}", f"{pop_growth_pct:+.1f}%", f"{features['pct_area_100yr_flood']:.1f}%", "Suburban MUD" if features['water_mud_flag']==1 else "Standard Municipal"])
            alt_vals.extend([f"${alt_total_cost:,.2f}", f"{float(best_alt_data.get('five_year_pop_growth_pct', 0.0)):+.1f}%", f"{alt_features['pct_area_100yr_flood']:.1f}%", "Suburban MUD" if alt_features['water_mud_flag']==1 else "Standard Municipal"])

            comp_df = pd.DataFrame({
                "Cost / Metric": comp_rows,
                f"Target: {zip_query} ({city_name})": target_vals,
                f"Alternative: {alt_zip} ({alt_city})": alt_vals
            })
            st.table(comp_df)

    with tab2:
        st.subheader("Actuarial Flood Hazard & Municipal Water Governance")
        st.markdown("""
        This tab evaluates structural flood exposure and water service governance for the selected ZIP code. 
        It details historical National Flood Insurance Program (NFIP) claim payouts, 100-year floodplain area percentages, 
        and Municipal Utility District (MUD) unincorporated tax governance rules to assess long-term infrastructure reliability.
        """)
        st.markdown("---")
        
        f_col1, f_col2, f_col3 = st.columns(3)
        with f_col1:
            st.metric("100-Yr Flood Zone Area", f"{float(data.get('pct_area_100yr_flood', 0.0)):.1f}%")
        with f_col2:
            st.metric("Water Base Monthly Fee", f"${float(data.get('water_base_monthly_fee', 0.0)):,.2f}")
        with f_col3:
            st.metric("Water Rate per 1k Gal", f"${float(data.get('water_rate_per_1k_gal', 0.0)):,.2f}")

        st.markdown("---")
        st.markdown(f"**Historical NFIP Flood Claims:** {data.get('nfip_total_claims', 'N/A')}")
        st.markdown(f"**Average Building Payout:** ${float(data.get('nfip_avg_building_claim', 0.0)):,.2f}")
        st.markdown(f"**Average Contents Payout:** ${float(data.get('nfip_avg_contents_claim', 0.0)):,.2f}")
        st.markdown("---")
        if features['water_mud_flag'] == 1:
            st.warning("⚠️ **Suburban MUD Territory Notice:** This ZIP code is predominantly located in unincorporated territory served by Suburban Municipal Utility Districts (MUDs), resulting in additional property tax levies.")
        else:
            st.success("✅ **Standard Municipal Jurisdiction:** Serviced directly by municipal infrastructure. No additional MUD property tax levies apply.")

    with tab3:
        st.subheader("Utility Delivery Grid Operators")
        g_col1, g_col2 = st.columns(2)
        with g_col1:
            st.markdown(f"### ⚡ Electricity")
            st.markdown(f"- **Transmission & Delivery Utility:** `{data.get('elec_provider', 'Unknown TDU')}`")
            st.markdown(f"- **Base Monthly Fee:** `${float(data.get('elec_base_monthly_fee', 0.0)):.2f}`")
            st.markdown(f"- **Delivery Surcharge per kWh:** `${float(data.get('elec_delivery_rate_per_kwh', 0.0)):.4f}`")
        with g_col2:
            st.markdown(f"### 🔥 Natural Gas")
            st.markdown(f"- **Gas Utility Distribution Provider:** `{data.get('gas_provider', 'Unknown TDU')}`")
            st.markdown(f"- **Base Monthly Fee:** `${float(data.get('gas_base_monthly_fee', 0.0)):.2f}`")
            st.markdown(f"- **Distribution Charge per CCF:** `${float(data.get('gas_distribution_rate_per_ccf', 0.0)):.4f}`")

    with tab6:
        st.subheader("Regional Budget Recommender")
        st.markdown(f"Find alternative ZIP codes within a specific radius of **{zip_query} ({city_name})** that fit your maximum occupancy budget.")
        
        with st.form("budget_recommender_form"):
            r_col1, r_col2 = st.columns(2)
            with r_col1:
                max_budget = st.number_input("Maximum Monthly Budget ($)", min_value=500.0, max_value=10000.0, value=1500.0, step=100.0)
            with r_col2:
                search_radius = st.slider("Search Radius (Miles)", min_value=1, max_value=50, value=10)
                
            submit_search = st.form_submit_button("Search Affordable Areas")

        if submit_search:
            with st.spinner("Scanning regional ZIP codes..."):
                target_lat = data.get('latitude', 0.0)
                target_lon = data.get('longitude', 0.0)
                df_search = df.copy()
                df_search['dist_to_target'] = df_search.apply(
                    lambda r: calc_geo_distance_miles(target_lat, target_lon, r.get('latitude', 0.0), r.get('longitude', 0.0)), axis=1
                )
                nearby_df = df_search[(df_search['dist_to_target'] <= search_radius) & (df_search['zip_code'] != zip_query)].copy()

                if nearby_df.empty:
                    st.warning(f"No alternative ZIP codes found within {search_radius} miles of {zip_query}.")
                else:
                    batch_input = pd.DataFrame({
                        'sqft': sqft_input,
                        'occupants': occupants_input,
                        'pct_area_100yr_flood': nearby_df['pct_area_100yr_flood'].fillna(0.0),
                        'twia_mandatory': nearby_df['twia_mandatory'].fillna(0),
                        'water_mud_flag': nearby_df['water_mud_flag'].fillna(0),
                        'houston_cpi_food_scalar': nearby_df['houston_cpi_food_scalar'].fillna(1.0),
                        'effective_tax_rate_pct': nearby_df['effective_tax_rate_pct'].fillna(2.2),
                        'five_year_pop_growth_pct': nearby_df['five_year_pop_growth_pct'].fillna(0.0)
                    })

                    batch_preds = model.predict(batch_input[expected_features])
                    
                    food_costs = (batch_preds[:, 0] * grocery_multiplier) * time_multiplier
                    utils_costs = (batch_preds[:, 1] + batch_preds[:, 2] + batch_preds[:, 3]) * time_multiplier
                    hoa_costs = (batch_preds[:, 4] * time_multiplier) if include_hoa else 0.0
                    twia_costs = batch_input['twia_mandatory'].apply(lambda x: calculate_twia_surcharge(int(x), time_multiplier))

                    nearby_df['Estimated Total'] = food_costs + utils_costs + hoa_costs + twia_costs
                    
                    affordable_df = nearby_df[nearby_df['Estimated Total'] <= (max_budget * time_multiplier)]
                    
                    if affordable_df.empty:
                        st.error(f"No ZIP codes within {search_radius} miles met the budget of ${max_budget:,.2f}.")
                    else:
                        st.success(f"Found {len(affordable_df)} affordable ZIP codes within {search_radius} miles!")
                        display_cols = ['zip_code', 'city_name', 'cluster_archetype_name', 'dist_to_target', 'Estimated Total']
                        res_df = affordable_df[display_cols].sort_values('Estimated Total').rename(columns={
                            'zip_code': 'ZIP',
                            'city_name': 'City',
                            'cluster_archetype_name': 'Archetype',
                            'dist_to_target': 'Distance (Miles)',
                            'Estimated Total': 'Total Monthly Cost'
                        })
                        
                        res_df['Total Monthly Cost'] = res_df['Total Monthly Cost'].apply(lambda x: f"${x:,.2f}")
                        res_df['Distance (Miles)'] = res_df['Distance (Miles)'].apply(lambda x: f"{x:.1f}")
                        st.dataframe(res_df, use_container_width=True, hide_index=True)

    with tab4:
        st.header("📊 ML Explainability & Diagnostics")
        st.markdown("""
        Welcome to the **"Under the Hood"** section of the engine. Here, we decode the XGBoost model's decision-making 
        process using SHapley Additive exPlanations (SHAP) and evaluate its statistical accuracy against our simulated testing dataset.
        """)
        st.divider()

        st.subheader("1. Household Evaluation Matrix & Baseline Analysis")
        st.markdown("""
        **Household Evaluation Matrix (Scenario Breakdown)**
        * **What it tracks:** A testing dataset of 2,790 simulated household scenarios combining 279 unique ZIP codes with random variations of home sizes (800 to 4,000 sqft) and household sizes (1 to 6 occupants). 
        * **What it means:** This provides a statistical breakdown that confirms the simulated parameters worked exactly as intended. It demonstrates how the engine behaves across a wide spectrum of realistic living situations to ensure accurate testing.

        **Baseline Cost Analysis (Global Interpretability)**
        * **What it tracks:** The starting baseline values calculated by the engine for all five core expenses: Food Cost, Electricity, Water, Gas, and HOA / Maintenance.
        * **What it means:** This decodes the model's underlying logic by establishing the baseline expected monthly costs before specific household features adjust the final prediction.
        
        *Baseline Intercepts established by the model:*
        * **Food Cost:** 1,314.46 USD
        * **HOA / Maintenance:** 207.06 USD
        * **Electricity:** 85.66 USD
        * **Water:** 75.25 USD
        * **Natural Gas:** 24.25 USD
        """)
        st.divider()

        st.subheader("2. Feature Impact Chart (Beeswarm Plot)")
        st.markdown("""
        * **What it tracks:** An explainability matrix that calculates exactly how much each individual feature influences the predicted expenses.
        * **What it means:** These charts provide a clear visual guide to how the model makes decisions. It shows exactly how much a specific feature (like a high number of occupants) adds or subtracts from the baseline monthly costs to generate your final custom estimate.
        """)
        render_image("shap_importance_all_targets.png", "Global Feature Importance Across All 5 Cost Targets", BASE_DIR)
        
        col_b1, col_b2 = st.columns(2)
        with col_b1:
            render_image("shap_beeswarm_food.png", "Food Cost SHAP Beeswarm", BASE_DIR)
            render_image("shap_beeswarm_electricity.png", "Electricity SHAP Beeswarm", BASE_DIR)
            render_image("shap_beeswarm_water.png", "Water SHAP Beeswarm", BASE_DIR)
        with col_b2:
            render_image("shap_beeswarm_gas.png", "Gas SHAP Beeswarm", BASE_DIR)
            render_image("shap_beeswarm_hoa.png", "HOA / Maintenance SHAP Beeswarm", BASE_DIR)
        st.divider()

        st.subheader("3. Regression Metrics by Cost Target")
        st.markdown("""
        * **What it plots:** These three side-by-side bar charts act as our model's report card. They map out three key performance scores across all five of our cost categories (Food, Electricity, Water, Gas, and HOA): R² (overall accuracy), MAE (our average dollar mistake), and RMSE (our average mistake, but penalized more for larger errors).
        * **What it means:** This shows the class at a glance what our model is good at. It visually proves that we are predicting Food, Water, Gas, and Electricity with incredible precision (tall R² bars near 1.0, and very short MAE/RMSE dollar bars). It also highlights that our HOA estimates are our weakest link, showing the highest average dollar error.
        """)
        render_image("metric_summary.png", "Regression Metrics by Cost Target (R², MAE, RMSE)", BASE_DIR)
        st.divider()

        st.subheader("4. Error Distribution Charts: Residual Distributions")
        st.markdown("""
        * **What it plots:** These five charts (histograms) look like bell curves and show the distribution of our errors. The horizontal axis represents the dollar amount we were off by, and the vertical axis (how tall the bars are) shows how many times we made that specific mistake.
        * **What it means:** This shows the spread of our mistakes. A tall, skinny spike right at the zero mark means we are consistently accurate with very few outliers (which is exactly what we see for Food and Water). The HOA chart has a wider, flatter spread, showing the class that when we miss on an HOA prediction, we tend to miss by a wider dollar range.
        """)
        render_image("residual_histograms.png", "Residual Distributions Across All 5 Cost Targets", BASE_DIR)
        st.divider()

        st.subheader("5. Error & Residual Diagnostics")
        st.markdown("""
        **Actual vs. Predicted (The Blue Scatter Plots)**
        * **What it plots:** For each cost category, we plotted the actual historical cost on the horizontal axis and our model's predicted cost on the vertical axis. We also drew a red dashed diagonal line straight through the middle, which represents a "perfect fit" where the prediction exactly matches the actual cost.
        * **What it means:** We want our blue dots to hug that red line as tightly as possible. When presenting this, point out how the dots for Food and Water are essentially perfectly lined up. For Electricity and Gas, you can point out a slight "staircase" effect, this is just a quirk of how our tree-based model groups square footage. For HOA, the dots are grouped in distinct streaks because actual HOA costs in our data are locked into three specific price tiers: 45 USD, 85 USD, and 275 USD.
        
        **Homoscedasticity Check (The Red Scatter Plots)**
        * **What it plots:** This plots our predicted cost on the horizontal axis against our "residuals" (the exact dollar amount we were off by) on the vertical axis. A black dashed line runs across zero, representing a prediction with zero error.
        * **What it means:** This chart checks if our model gets confused by more expensive homes. Ideally, we want the red dots scattered evenly both above and below the zero line, no matter how high the predicted cost gets. For Gas and Electricity, the errors are steady and flat across the board. However, for HOA, the class will notice the errors spread out and get larger as the predictions increase, showing exactly where our model struggles with those higher-tier ZIP codes.
        """)
        
        diag_tabs = st.tabs(["Groceries & Food", "Electricity", "Water", "Natural Gas", "HOA / Maintenance"])
        with diag_tabs[0]:
            render_image("diagnostics_food.png", "Food Cost Target Validation & Homoscedasticity Check", BASE_DIR)
        with diag_tabs[1]:
            render_image("diagnostics_electricity.png", "Electricity Target Validation & Homoscedasticity Check", BASE_DIR)
        with diag_tabs[2]:
            render_image("diagnostics_water.png", "Water Target Validation & Homoscedasticity Check", BASE_DIR)
        with diag_tabs[3]:
            render_image("diagnostics_gas.png", "Gas Target Validation & Homoscedasticity Check", BASE_DIR)
        with diag_tabs[4]:
            render_image("diagnostics_hoa.png", "HOA / Maintenance Target Validation & Homoscedasticity Check", BASE_DIR)
        st.divider()

        st.subheader("6. Local Household Interpretability (Live SHAP)")
        st.markdown("Understand exactly how your current **sidebar parameters dynamically influence** the ML-predicted components for the active ZIP code scenario.")
        
        if hasattr(model, 'estimators_'):
            target_names = ["Groceries & Food", "Electricity", "Water", "Natural Gas", "Subdivision Maintenance / HOA"]
            for i, estimator in enumerate(model.estimators_):
                target_name = target_names[i]
                with st.expander(f"Live Feature Impact: {target_name}", expanded=(i==0)):
                    try:
                        explainer = shap.TreeExplainer(estimator)
                        shap_values = explainer.shap_values(input_df)
                        fig, ax = plt.subplots(figsize=(8, 4))
                        shap.summary_plot(shap_values, input_df, plot_type="bar", show=False, axis_color="#2B6CB0")
                        plt.title(f"Local Household Breakdown for {target_name}")
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close(fig)
                    except Exception as e:
                        st.error(f"Failed to generate SHAP tree graph: {e}")
        else:
            st.warning("The loaded model lacks underlying tree estimators for live local SHAP extraction.")

    with tab5:
        st.subheader("Project Roadmap: Phase 03 Capstone Completion")
        with st.container():
            col1, col2 = st.columns([1, 3])
            with col1:
                st.markdown("#### Week 7")
                st.caption("Pipeline & Environment")
            with col2:
                st.markdown("""
                - **Achieved:** Mounted Google Drive working directories, loaded the core `DataPipeline`, and successfully initialized the XGBoost multi-output regression model for all 5 utility estimators.
                - **Contributors:** Brandon Matias & Taki Eddine Boubekri
                """)
            st.divider()

        with st.container():
            col1, col2 = st.columns([1, 3])
            with col1:
                st.markdown("#### Week 8")
                st.caption("Evaluation Matrix")
            with col2:
                st.markdown("""
                - **Achieved:** Generated the testing dataset of 2,790 simulated household scenarios across 279 ZIP codes to establish a statistical breakdown of model behavior.
                - **Contributors:** Jonah Joseph & Khadijah Patience
                """)
            st.divider()

        with st.container():
            col1, col2 = st.columns([1, 3])
            with col1:
                st.markdown("#### Week 9")
                st.caption("Global Interpretability")
            with col2:
                st.markdown("""
                - **Achieved:** Initialized the SHAP library and ran a `TreeExplainer` across all 5 regressors to calculate feature influence and establish baseline expected monthly costs.
                - **Contributors:** Brandon Matias & Jonah Joseph
                """)

if __name__ == "__main__":
    main()
