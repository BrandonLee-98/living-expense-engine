Python
# import os
# import joblib
# import pandas as pd
# import numpy as np
# import streamlit as st
# import shap
# import matplotlib.pyplot as plt
# import math
# 
# # ==========================================
# # 1. INITIALIZATION & CONFIGURATION
# # ==========================================
# st.set_page_config(
#     page_title="Living Expense Estimation Engine",
#     page_icon="🏙️",
#     layout="wide"
# )
# 
# st.markdown("""
#     <style>
#     .stApp { background-color: #0E1117; color: #E2E8F0; }
#     .stSidebar { background-color: #1A202C; }
#     </style>
# """, unsafe_allow_html=True)
# 
# # ==========================================
# # 2. CACHED RESOURCE LOADING
# # ==========================================
# @st.cache_resource(show_spinner=False)
# def load_model(model_path: str):
#     if not os.path.exists(model_path):
#         raise FileNotFoundError(f"Model not found at '{model_path}'.")
#     return joblib.load(model_path)
# 
# @st.cache_data(show_spinner=False)
# def load_data(csv_path: str):
#     if not os.path.exists(csv_path):
#         raise FileNotFoundError(f"CSV not found at '{csv_path}'.")
#     df = pd.read_csv(csv_path)
#     if 'zip_code' in df.columns:
#         df['zip_code'] = df['zip_code'].astype(str).str.zfill(5)
#     return df
# 
# # ==========================================
# # 3. SPATIAL ARBITRAGE MATH ENGINE
# # ==========================================
# def calc_geo_distance_miles(lat1, lon1, lat2, lon2):
#     if pd.isna(lat1) or pd.isna(lat2): return 999.0
#     lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
#     a = math.sin((lat2 - lat1)/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1)/2)**2
#     return 2 * math.asin(math.sqrt(a)) * 3958.8
# 
# def find_nearest_arbitrage_zip(df, target_row):
#     target_zip = target_row['zip_code']
#     target_twia = int(target_row.get('twia_mandatory', 0))
#     target_mud = int(target_row.get('water_mud_flag', 0))
#     target_flood = float(target_row.get('pct_area_100yr_flood', 0.0))
#     target_lat = target_row.get('latitude', 0.0)
#     target_lon = target_row.get('longitude', 0.0)
# 
#     candidates = df[df['zip_code'] != target_zip].copy()
#     if candidates.empty: return target_row, "none"
# 
#     candidates['geo_dist_miles'] = candidates.apply(
#         lambda r: calc_geo_distance_miles(target_lat, target_lon, r.get('latitude', 0.0), r.get('longitude', 0.0)), axis=1
#     )
# 
#     if target_twia == 1:
#         ideal = candidates[(candidates.get('twia_mandatory', 0) == 0) & (candidates.get('water_mud_flag', 0) == 0)]
#         if not ideal.empty: return ideal.sort_values(by=['geo_dist_miles']).iloc[0], "cost_governance"
#     if target_mud == 1:
#         no_mud = candidates[(candidates.get('water_mud_flag', 0) == 0) & (candidates.get('twia_mandatory', 0) == 0)]
#         if not no_mud.empty: return no_mud.sort_values(by=['geo_dist_miles']).iloc[0], "mud_only"
#     if target_flood > 5.0:
#         lower_flood = candidates[(candidates.get('twia_mandatory', 0) == 0) & (candidates.get('pct_area_100yr_flood', target_flood) < target_flood)]
#         if not lower_flood.empty: return lower_flood.sort_values(by=['geo_dist_miles']).iloc[0], "flood_mitigation"
# 
#     return target_row, "optimal"
# 
# # ==========================================
# # 4. MAIN APPLICATION
# # ==========================================
# def main():
#     st.title("🏙️ Non-Mortgage Living Expense Estimation Engine")
#     st.markdown("### Harris, Fort Bend, Galveston, and Brazoria Counties")
# 
#     MODEL_FILENAME = "xgboost_cost_model.joblib"
#     CSV_FILENAME = "master_living_costs_clustered.csv"
# 
#     try:
#         model = load_model(MODEL_FILENAME)
#         df = load_data(CSV_FILENAME)
#     except Exception as e:
#         st.error(f"Critical Initialization Error: {e}")
#         st.stop()
# 
#     # --- UI SIDEBAR ---
#     st.sidebar.header("1. Property Parameters")
#     zip_query = st.sidebar.text_input("Enter 5-Digit ZIP Code", value="77598", max_chars=5).strip()
#     sqft_input = st.sidebar.number_input("Property Square Footage", min_value=500, max_value=10000, value=1500)
# 
#     # FIXED: Occupants max_value capped at 6 to align with XGBoost training data logic
#     occupants_input = st.sidebar.slider("Household Occupants", min_value=1, max_value=6, value=2)
# 
#     st.sidebar.markdown("---")
#     st.sidebar.header("2. Household Budget Profiles")
# 
#     tier_display_names = [
#         "Thrifty (SNAP Baseline / Maximum Savings)",
#         "Low-Cost (Budget-Conscious / Sales & Coupons)",
#         "Moderate-Cost (Standard Middle-Class Average)",
#         "Liberal (Premium / Specialty & Organic Focus)"
#     ]
#     selected_tier_name = st.sidebar.selectbox("USDA Grocery Plan Tier", tier_display_names, index=2)
#     view_mode = st.sidebar.radio("Display Horizon", ["Monthly ($/mo)", "Annualized ($/yr)"], horizontal=True)
#     time_multiplier = 12 if view_mode == "Annualized (\$/yr)" else 1
# 
#     # --- LOOKUP PIPELINE ---
#     selected_row = df[df['zip_code'] == zip_query]
#     if selected_row.empty:
#         st.error("❌ ZIP Code not found in the regional dataset.")
#         st.stop()
#     data = selected_row.iloc[0]
# 
#     city_name = data.get('city_name', 'Houston Area')
#     county_name = data.get('county_name', 'Unknown')
#     archetype = data.get('cluster_archetype_name', 'Unclassified Region')
# 
#     # FIXED: ZIP Classification Profile Formatting in Sidebar
#     st.sidebar.markdown("---")
#     st.sidebar.header("3. ZIP Classification Profile")
#     col1, col2 = st.sidebar.columns(2)
#     col1.metric(label="City", value=city_name)
#     col2.metric(label="Archetype", value=archetype)
#     st.sidebar.info("This ZIP code exhibits stable municipal infrastructure with high-density urban core characteristics, influencing baseline utility demands.")
# 
#     # --- ML DATA PREPARATION ---
#     features = {
#         'sqft': sqft_input,
#         'occupants': occupants_input,
#         'pct_area_100yr_flood': float(data.get('pct_area_100yr_flood', 0.0)),
#         'twia_mandatory': int(data.get('twia_mandatory', 0)),
#         'water_mud_flag': int(data.get('water_mud_flag', 0)),
#         'houston_cpi_food_scalar': float(data.get('houston_cpi_food_scalar', 1.0)),
#         'effective_tax_rate_pct': float(data.get('effective_tax_rate_pct', 2.2)),
#         'five_year_pop_growth_pct': float(data.get('five_year_pop_growth_pct', 5.0))
#     }
# 
#     expected_features = ['sqft', 'occupants', 'pct_area_100yr_flood', 'twia_mandatory',
#                          'water_mud_flag', 'houston_cpi_food_scalar', 'effective_tax_rate_pct',
#                          'five_year_pop_growth_pct']
#     input_df = pd.DataFrame([features])[expected_features]
# 
#     # --- PREDICTION EXECUTION ---
#     with st.spinner("Running XGBoost Inference..."):
#         predictions = model.predict(input_df)[0]
# 
#     # Target outputs: ["Groceries/Food", "Utilities", "Property Tax", "Home Insurance"]
#     tier_scalars = {
#         "Thrifty (SNAP Baseline / Maximum Savings)": 0.75,
#         "Low-Cost (Budget-Conscious / Sales & Coupons)": 0.88,
#         "Moderate-Cost (Standard Middle-Class Average)": 1.00,
#         "Liberal (Premium / Specialty & Organic Focus)": 1.20
#     }
#     grocery_multiplier = tier_scalars[selected_tier_name]
# 
#     # FIXED: Syntax errors in math operations
#     pred_food = (predictions[0] * grocery_multiplier) * time_multiplier
#     pred_utils = predictions[1] * time_multiplier
#     pred_tax = predictions[2] * time_multiplier
#     pred_ins = predictions[3] * time_multiplier
#     total_cost = pred_food + pred_utils + pred_tax + pred_ins
# 
#     # --- ARBITRAGE EXECUTION ---
#     best_alt_data, arbitrage_type = find_nearest_arbitrage_zip(df, data)
#     alt_zip = best_alt_data['zip_code']
#     alt_city = best_alt_data.get('city_name', 'Nearby')
#     dist_miles = calc_geo_distance_miles(data.get('latitude'), data.get('longitude'), best_alt_data.get('latitude'), best_alt_data.get('longitude'))
# 
#     # Calculate Alt Costs via ML
#     alt_features = features.copy()
#     alt_features.update({
#         'pct_area_100yr_flood': float(best_alt_data.get('pct_area_100yr_flood', 0.0)),
#         'twia_mandatory': int(best_alt_data.get('twia_mandatory', 0)),
#         'water_mud_flag': int(best_alt_data.get('water_mud_flag', 0)),
#         'houston_cpi_food_scalar': float(best_alt_data.get('houston_cpi_food_scalar', 1.0)),
#         'effective_tax_rate_pct': float(best_alt_data.get('effective_tax_rate_pct', features['effective_tax_rate_pct']))
#     })
#     alt_df = pd.DataFrame([alt_features])[expected_features]
#     alt_preds = model.predict(alt_df)[0]
# 
#     # FIXED: Syntax errors in alt math operations
#     alt_pred_food = (alt_preds[0] * grocery_multiplier) * time_multiplier
#     alt_pred_utils = alt_preds[1] * time_multiplier
#     alt_pred_tax = alt_preds[2] * time_multiplier
#     alt_pred_ins = alt_preds[3] * time_multiplier
#     alt_total_cost = alt_pred_food + alt_pred_utils + alt_pred_tax + alt_pred_ins
# 
#     # --- RENDER DASHBOARD HEADER ---
#     if features['twia_mandatory'] == 1:
#         st.warning(f"🌊 **Coastal Surcharge Notice ({city_name}):** Mandatory Texas Windstorm Insurance Association (TWIA) coverage required.")
#     if features['pct_area_100yr_flood'] > 20.0:
#         st.error(f"⚠️ **High Flood Hazard Alert ({city_name}):** {features['pct_area_100yr_flood']:.1f}% of this ZIP area lies within the 100-year floodplain.")
# 
#     # FIXED: Syntax error in subheader
#     st.subheader(f"Cost Breakdown for ZIP Code: {zip_query} | {city_name} ({archetype})")
#     col1, col2, col3, col4 = st.columns(4)
#     col1.metric("Total Occupancy Cost", f"\${total_cost:,.2f}")
#     col2.metric(f"Food ({selected_tier_name.split(' ')[0]})", f"\${pred_food:,.2f}")
#     col3.metric("Utilities", f"\${pred_utils:,.2f}")
#     col4.metric("Property Tax + Ins.", f"\${(pred_tax + pred_ins):,.2f}")
# 
#     st.markdown("---")
# 
#     # --- DEEP DIVE TABS ---
#     tab1, tab2, tab3, tab4, tab5 = st.tabs([
#         "💰 Itemized Breakdown & Comparison",
#         "🌊 Flood & MUD Risk",
#         "⚡ Utility Grid Detail",
#         "🤖 ML Explainability (SHAP)",
#         "🗺️ Phase 03 Project Roadmap"
#     ])
# 
#     with tab1:
#         st.subheader("Target vs. Alternative Component Comparison")
#         if arbitrage_type == "optimal":
#             st.success(f"🌟 **Optimal Non-Mortgage Governance:** ZIP {zip_query} operates on standard municipal utilities without MUD taxes or coastal surcharges.")
#         else:
#             st.info(f"💡 **Arbitrage Found:** Evaluating nearby **ZIP {alt_zip} ({alt_city})** — ~{dist_miles:.1f} miles away. Total Cost: \${alt_total_cost:,.2f}")
# 
#         comp_df = pd.DataFrame({
#             "Cost / Metric": [
#                 "Groceries & Food", "Utilities (Power/Water/Gas)",
#                 "Property Tax", "Home/TWIA Insurance",
#                 "Total Estimated Cost", "100-Yr Flood Exposure", "MUD Governance"
#             ],
#             f"Target: {zip_query} ({city_name})": [
#                 f"${pred_food:,.2f}", f"${pred_utils:,.2f}",
#                 f"${pred_tax:,.2f}", f"${pred_ins:,.2f}",
#                 f"\${total_cost:,.2f}", f"{features['pct_area_100yr_flood']:.1f}%",
#                 "Suburban MUD" if features['water_mud_flag']==1 else "Standard Municipal"
#             ],
#             f"Alternative: {alt_zip} ({alt_city})": [
#                 f"${alt_pred_food:,.2f}", f"${alt_pred_utils:,.2f}",
#                 f"${alt_pred_tax:,.2f}", f"${alt_pred_ins:,.2f}",
#                 f"\${alt_total_cost:,.2f}", f"{alt_features['pct_area_100yr_flood']:.1f}%",
#                 "Suburban MUD" if alt_features['water_mud_flag']==1 else "Standard Municipal"
#             ]
#         })
#         st.table(comp_df)
# 
#     with tab2:
#         st.subheader("Actuarial Flood Hazard & Municipal Water Governance")
#         st.markdown(f"**Historical NFIP Flood Claims:** {data.get('nfip_total_claims', 'N/A')}")
#         st.markdown(f"**Average Building Payout:** \${float(data.get('nfip_avg_building_claim', 0.0)):,.2f}")
#         st.markdown(f"**Average Contents Payout:** \${float(data.get('nfip_avg_contents_claim', 0.0)):,.2f}")
#         st.markdown("---")
#         if features['water_mud_flag'] == 1:
#             st.warning("⚠️ **Suburban MUD Territory Notice:** This ZIP code is predominantly located in unincorporated territory served by Suburban Municipal Utility Districts (MUDs), resulting in additional property tax levies.")
#         else:
#             st.success("✅ **Standard Municipal Jurisdiction:** Serviced directly by municipal infrastructure. No additional MUD property tax levies apply.")
# 
#     with tab3:
#         st.subheader("Utility Delivery Grid Operators")
#         g_col1, g_col2 = st.columns(2)
#         with g_col1:
#             st.markdown(f"### ⚡ Electricity")
#             st.markdown(f"- **Transmission & Delivery Utility:** `{data.get('elec_provider', 'Unknown TDU')}`")
#             st.markdown(f"- **Base Monthly Fee:** `${float(data.get('elec_base_monthly_fee', 0.0)):.2f}`")
#             st.markdown(f"- **Delivery Surcharge per kWh:** `${float(data.get('elec_delivery_rate_per_kwh', 0.0)):.4f}`")
#         with g_col2:
#             st.markdown(f"### 🔥 Natural Gas")
#             st.markdown(f"- **Gas Utility Distribution Provider:** `{data.get('gas_provider', 'Unknown TDU')}`")
#             st.markdown(f"- **Base Monthly Fee:** `${float(data.get('gas_base_monthly_fee', 0.0)):.2f}`")
#             st.markdown(f"- **Distribution Charge per CCF:** `${float(data.get('gas_distribution_rate_per_ccf', 0.0)):.4f}`")
# 
#     with tab4:
#         # FIXED: Robust SHAP Interpretability Tab
#         st.subheader("Model Logic & Interpretability (SHAP)")
#         st.markdown("Understand exactly how household parameters influence the final estimated living expenses.")
# 
#         with st.expander("How to read these charts"):
#             st.markdown("""
#             **What it tracks:** These charts break down the exact math for this specific household prediction.
#             - It starts from the **Baseline Value** (the average cost for all households in the region).
#             - Each bar shows exactly how many dollars a specific feature added or subtracted from that baseline to reach the final estimated cost.
#             """)
# 
#         if hasattr(model, 'estimators_'):
#             for i, estimator in enumerate(model.estimators_):
#                 target_name = ["Groceries/Food", "Utilities", "Property Tax", "Home Insurance"][i] if i < 4 else f"Target {i+1}"
#                 with st.expander(f"Feature Impact: {target_name}", expanded=(i==0)):
#                     try:
#                         explainer = shap.TreeExplainer(estimator)
#                         shap_values = explainer.shap_values(input_df)
#                         fig, ax = plt.subplots(figsize=(8, 4))
#                         shap.summary_plot(shap_values, input_df, plot_type="bar", show=False, axis_color="#2B6CB0")
#                         plt.title(f"Local Household Breakdown for {target_name}")
#                         plt.tight_layout()
#                         st.pyplot(fig)
#                         plt.close(fig)
#                     except Exception as e:
#                         st.error(f"Failed to generate SHAP tree graph: {e}")
#         else:
#             st.warning("The loaded model lacks underlying tree estimators for SHAP extraction.")
# 
#     with tab5:
#         # FIXED: Phase 03 Roadmap layout via columns and containers
#         st.subheader("Project Roadmap: Phase 03 Capstone Completion")
#         st.markdown("Detailed breakdown of pipeline initialization, model evaluation, and UI integration.")
# 
#         with st.container():
#             col1, col2 = st.columns([1, 3])
#             with col1:
#                 st.markdown("#### Week 7")
#                 st.caption("Pipeline & Environment")
#             with col2:
#                 st.markdown("""
#                 - **Achieved:** Mounted Google Drive working directories, loaded the core `DataPipeline`, and successfully initialized the XGBoost multi-output regression model for all 5 utility estimators.
#                 - **Contributors:** Brandon Matias & Taki Eddine Boubekri
#                 """)
#             st.divider()
# 
#         with st.container():
#             col1, col2 = st.columns([1, 3])
#             with col1:
#                 st.markdown("#### Week 8")
#                 st.caption("Evaluation Matrix")
#             with col2:
#                 st.markdown("""
#                 - **Achieved:** Generated the testing dataset of 2,790 simulated household scenarios across 279 ZIP codes to establish a statistical breakdown of model behavior.
#                 - **Contributors:** Jonah Joseph & Khadijah Patience
#                 """)
#             st.divider()
# 
#         with st.container():
#             col1, col2 = st.columns([1, 3])
#             with col1:
#                 st.markdown("#### Week 9")
#                 st.caption("Global Interpretability")
#             with col2:
#                 st.markdown("""
#                 - **Achieved:** Initialized the SHAP library and ran a `TreeExplainer` across all 5 regressors to calculate feature influence and establish baseline expected monthly costs.
#                 - **Contributors:** Brandon Matias & Jonah Joseph
#                 """)
#             st.divider()
# 
#         with st.container():
#             col1, col2 = st.columns([1, 3])
