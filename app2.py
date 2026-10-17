# ============================================================
# ICU CLINICAL PREDICTION SYSTEM
# Unified Streamlit app: Mortality Risk OR ICU Length of Stay
# Developed by Mubarik Meka
# ============================================================
import os
import warnings
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.utils.validation import check_is_fitted

######################################################
# PAGE CONFIGURATION — MUST COME BEFORE OTHER STREAMLIT COMMANDS
# ============================================================

st.set_page_config(
    page_title="ICU Clinical Prediction System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

warnings.filterwarnings("ignore", category=UserWarning)


# ============================================================
# SIDEBAR CONTACT INFORMATION — VISIBLE BEFORE AND AFTER LOGIN
# ============================================================

with st.sidebar:
    st.markdown("## 📩 Access & Information")
    st.markdown("---")

    st.write(
        "If you want access to log in or need more information, "
        "please contact us:"
    )

    st.markdown("### 👨‍💻 Contact")
    st.markdown("📧 **Email:**")
    st.markdown(
        "[mubarikmeka0@gmail.com](mailto:mubarikmeka0@gmail.com)"
    )

    st.markdown("📞 **Phone:**")
    st.markdown("[+251 932 998 635](tel:+251932998635)")

    st.markdown("---")
    st.caption("ICU Clinical Prediction System")


# ============================================================
# LOGIN / PASSWORD PROTECTION
# ============================================================

def login():
    VALID_USERNAME = "mubarik"
    VALID_PASSWORD = "8100"

    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        st.title("🔐 ICU Clinical Prediction System")
        st.subheader("Please log in to continue")

        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input(
                "Password",
                type="password"
            )

            submitted = st.form_submit_button("Login")

            if submitted:
                if (
                    username == VALID_USERNAME
                    and password == VALID_PASSWORD
                ):
                    st.session_state.authenticated = True
                    st.rerun()
                else:
                    st.error("❌ Incorrect username or password.")

        st.stop()


# ============================================================
# REQUIRE LOGIN BEFORE ACCESSING THE PREDICTION APPLICATION
# ============================================================

login()




######################################################

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MORTALITY_MODEL_PATH = os.path.join(BASE_DIR, "icu_mortality_xgboost.pkl")
MORTALITY_PREPROCESSOR_PATH = os.path.join(BASE_DIR, "icu_mortality_preprocessor.pkl")
LOS_MODEL_PATH = os.path.join(BASE_DIR, "icu_los_random_forest.pkl")
LOS_PREPROCESSOR_PATH = os.path.join(BASE_DIR, "preprocessor.pkl")

st.markdown("""
<style>
.main-title {font-size:32px;font-weight:800;text-align:center;margin-bottom:4px;}
.sub-title {text-align:center;font-size:15px;opacity:.82;margin-bottom:18px;}
.footer {text-align:center;font-size:12px;margin-top:30px;opacity:.75;line-height:1.8;}
div[data-testid="stMetric"] {border:1px solid rgba(128,128,128,.25);padding:12px;border-radius:10px;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🏥 ICU Clinical Prediction System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">ICU Mortality Risk and Length-of-Stay Prediction • Research prototype</div>', unsafe_allow_html=True)

# The user chooses a task first. Only the selected task's artifacts are loaded.
task = st.radio(
    "Select prediction task",
    ["Mortality Risk", "ICU Length of Stay (LOS)"],
    horizontal=True,
    key="prediction_task",
)
st.divider()

@st.cache_resource
def load_artifacts(model_path, preprocessor_path):
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found:\n{model_path}")
    if not os.path.isfile(preprocessor_path):
        raise FileNotFoundError(f"Fitted preprocessor file not found:\n{preprocessor_path}")
    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)
    check_is_fitted(preprocessor)
    if not hasattr(model, "predict_proba"):
        raise TypeError("The selected model does not support predict_proba().")
    return model, preprocessor


def parse_optional(raw, name):
    raw = str(raw).strip()
    if raw == "":
        return np.nan
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(f"{name}: enter a valid number or leave it blank.")
    if not np.isfinite(value):
        raise ValueError(f"{name}: enter a finite number or leave it blank.")
    return value


def get_categories(preprocessor, column, fallback=None):
    try:
        cat_pipe = preprocessor.named_transformers_["categorical"]
        encoder = cat_pipe.named_steps["onehot"]
        cat_cols = list(preprocessor.transformers_[2][2])
        idx = cat_cols.index(column)
        return [str(x) for x in encoder.categories_[idx]]
    except Exception:
        return list(fallback or [])


def category_select(label, options, key, allow_missing=True):
    options = list(options)
    if not options:
        st.error(f"No fitted training categories could be read for {label}. Check the saved preprocessor.")
        st.stop()
    choices = (["Not recorded / missing"] if allow_missing else []) + options
    selected = st.selectbox(label, choices, key=key)
    return np.nan if selected == "Not recorded / missing" else selected

# Broad data-entry guardrails; these are not diagnostic thresholds.
RANGES = {
    "Age": (0, 120), "Heart rate": (20, 300), "Systolic blood pressure": (30, 350),
    "Diastolic blood pressure": (10, 250), "Respiratory rate": (1, 100),
    "SpO2": (0, 100), "Temperature": (25, 45), "RBS": (1, 2000),
    "Pain score": (0, 10), "Sodium": (80, 220), "Potassium": (1, 12),
    "Creatinine": (0, 50), "Neutrophil": (0, 100), "Lymphocyte": (0, 100),
    "Monocyte": (0, 100), "Platelet count": (1, 3000), "Hemoglobin": (1, 30),
    "AST/SGOT": (0, 20000), "WBC": (0, 500),
}

def check_ranges(values, names=None):
    errors = []
    for name, value in values.items():
        if pd.isna(value):
            continue
        rule = RANGES.get(name)
        if rule and not rule[0] <= value <= rule[1]:
            errors.append(f"{name} must be between {rule[0]:g} and {rule[1]:g}.")
    return errors


def prepare_patient(preprocessor, patient_data, model):
    processed = preprocessor.transform(patient_data)
    if hasattr(processed, "toarray"):
        processed = processed.toarray()
    processed = np.asarray(processed, dtype=float)
    if processed.ndim != 2 or processed.shape[0] != 1:
        raise ValueError("Preprocessed patient data must contain exactly one row.")
    if not np.isfinite(processed).all():
        raise ValueError("Missing/non-finite values remain after preprocessing. Check the saved fitted preprocessor.")
    expected = getattr(model, "n_features_in_", None)
    if expected is not None and processed.shape[1] != expected:
        raise ValueError(f"Preprocessor generated {processed.shape[1]} features, but model expects {expected}. Confirm model/preprocessor files match.")
    return processed


if task == "Mortality Risk":
    st.header("ICU Mortality Risk Prediction")
    st.caption("XGBoost classifier • 0 = survived, 1 = died")
    try:
        model, preprocessor = load_artifacts(MORTALITY_MODEL_PATH, MORTALITY_PREPROCESSOR_PATH)
    except Exception as e:
        st.error("Unable to load the mortality model and fitted preprocessor.")
        st.exception(e)
        st.stop()

    fallback_case = ["Cardiovascular Disease", "Respiratory Disease", "Neurological Disease", "Sepsis / Infection",
                     "Gastrointestinal / Hepatic Disease", "Trauma / Toxicology", "Endocrine / Metabolic Disease",
                     "Renal Disease", "Thromboembolism", "Malignancy", "Multi-organ Failure",
                     "Obstetric / Gynecology", "Autoimmune / Systemic Disease", "Other"]
    # Mortality preprocessor may have a different transformer layout; use original app's category options if needed.
    case_options = get_categories(preprocessor, "case_category", fallback_case)
    oxy_options = get_categories(preprocessor, "oxygen_support", ["Off oxygen", "Intranasal_O2_support", "Facemask O2_support", "Mechanical ventilation"])

    with st.sidebar:
        st.subheader("Mortality model information")
        st.write("**Algorithm:** XGBoost classifier")
        st.write("**Class 0:** Survived / lower predicted mortality probability")
        st.write("**Class 1:** Died / higher predicted mortality probability")
        st.caption("Research prototype; not a standalone clinical decision tool.")
        with st.expander("Model checks"):
            st.write("Expected processed features:", getattr(model, "n_features_in_", "Not available"))
            st.write("Model classes:", list(getattr(model, "classes_", [])))

    st.info("Enter the available patient measurements. Blank numeric fields are imputed by the saved mortality preprocessor. Values outside the input guardrails are rejected.")
    with st.form("mortality_form"):
        st.subheader("1. Demographic information")
        c1, c2 = st.columns(2)
        with c1:
            age_raw = st.text_input("Age (years)", placeholder="Leave blank if unavailable", key="m_age")
        with c2:
            case_category = category_select("Diagnosis category", case_options, "m_case", allow_missing=False)

        st.subheader("2. Vital signs")
        c1, c2, c3 = st.columns(3)
        with c1:
            hr_raw = st.text_input("Heart rate (beats/min)", placeholder="Leave blank if unavailable", key="m_hr")
            sbp_raw = st.text_input("Systolic blood pressure (mmHg)", placeholder="Leave blank if unavailable", key="m_sbp")
        with c2:
            dbp_raw = st.text_input("Diastolic blood pressure (mmHg)", placeholder="Leave blank if unavailable", key="m_dbp")
            rr_raw = st.text_input("Respiratory rate (breaths/min)", placeholder="Leave blank if unavailable", key="m_rr")
        with c3:
            temp_raw = st.text_input("Temperature (°C)", placeholder="Leave blank if unavailable", key="m_temp")
            pain_raw = st.text_input("Pain score (0–10)", placeholder="Leave blank if unavailable", key="m_pain")

        st.subheader("3. Laboratory measurements")
        c1, c2, c3 = st.columns(3)
        with c1:
            sodium_raw = st.text_input("Sodium", placeholder="Leave blank if unavailable", key="m_sodium")
            potassium_raw = st.text_input("Potassium", placeholder="Leave blank if unavailable", key="m_potassium")
        with c2:
            wbc_raw = st.text_input("White blood cell count (WBC)", placeholder="Leave blank if unavailable", key="m_wbc")
            neut_raw = st.text_input("Neutrophil (%)", placeholder="Leave blank if unavailable", key="m_neut")
        with c3:
            lymph_raw = st.text_input("Lymphocyte (%)", placeholder="Leave blank if unavailable", key="m_lymph")
            plt_raw = st.text_input("Platelet count (PLT)", placeholder="Leave blank if unavailable", key="m_plt")
        c1, c2 = st.columns(2)
        with c1:
            hb_raw = st.text_input("Hemoglobin", placeholder="Leave blank if unavailable", key="m_hb")
        with c2:
            ast_raw = st.text_input("AST/SGOT", placeholder="Leave blank if unavailable", key="m_ast")

        oxy_support = category_select("Oxygen support category", oxy_options, "m_oxy", allow_missing=False)
        st.caption("Select oxygen support as a category; no oxygen flow/consumption value is required.")
        submitted = st.form_submit_button("Predict mortality risk", type="primary", use_container_width=True)

    if submitted:
        raw = {"Age": age_raw, "Heart rate": hr_raw, "Systolic blood pressure": sbp_raw,
               "Diastolic blood pressure": dbp_raw, "Respiratory rate": rr_raw, "Temperature": temp_raw,
               "Pain score": pain_raw, "Sodium": sodium_raw, "Potassium": potassium_raw, "WBC": wbc_raw,
               "Neutrophil": neut_raw, "Lymphocyte": lymph_raw, "Platelet count": plt_raw,
               "Hemoglobin": hb_raw, "AST/SGOT": ast_raw}
        try:
            v = {k: parse_optional(val, k) for k, val in raw.items()}
        except ValueError as e:
            st.error(str(e)); st.stop()
        errors = check_ranges(v)
        if errors:
            st.error("Correct these values before prediction:")
            for e in errors: st.markdown(f"- {e}")
            st.stop()

        age, hr, sbp, dbp = v["Age"], v["Heart rate"], v["Systolic blood pressure"], v["Diastolic blood pressure"]
        rr, temp, pain = v["Respiratory rate"], v["Temperature"], v["Pain score"]
        sodium, potassium, wbc = v["Sodium"], v["Potassium"], v["WBC"]
        neut, lymph, plt = v["Neutrophil"], v["Lymphocyte"], v["Platelet count"]
        hb, ast = v["Hemoglobin"], v["AST/SGOT"]
        shock = hr / (sbp + 1e-6) if pd.notna(hr) and pd.notna(sbp) else np.nan
        interaction = wbc * temp if pd.notna(wbc) and pd.notna(temp) else np.nan
        patient = pd.DataFrame([{
            "dbp": dbp, "Shock_Index": shock, "sodium": sodium, "potassium": potassium,
            "neutrophil": neut, "lymphocyte": lymph, "plt": plt, "rr": rr,
            "case_category": case_category, "Age_years": age, "oxygen_support": oxy_support,
            "ast_sgot": ast, "hemoglobin": hb, "Pain_score": pain,
            "WBC_Temperature_Interaction": interaction,
        }])
        mortality_cols = ["dbp", "Shock_Index", "sodium", "potassium", "neutrophil", "lymphocyte", "plt", "rr", "case_category", "Age_years", "oxygen_support", "ast_sgot", "hemoglobin", "Pain_score", "WBC_Temperature_Interaction"]
        patient = patient[mortality_cols]
        try:
            processed = prepare_patient(preprocessor, patient, model)
            prediction = int(model.predict(processed)[0])
            classes = list(model.classes_)
            if 1 not in classes: raise ValueError("Mortality model does not contain class 1 (died).")
            pdeath = float(model.predict_proba(processed)[0, classes.index(1)])
        except Exception as e:
            st.error("Mortality prediction could not be generated."); st.exception(e); st.stop()

        st.divider(); st.subheader("Mortality prediction result")
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Predicted class", "Died / higher predicted risk" if prediction == 1 else "Survived / lower predicted risk")
        with c2:
            st.metric("Model-estimated probability of death", f"{pdeath:.1%}")
        gauge = go.Figure(go.Indicator(mode="gauge+number", value=pdeath*100, number={"suffix":"%"},
            title={"text":"Estimated probability of death"}, gauge={"axis":{"range":[0,100]},"bar":{"color":"#345995"}}))
        gauge.update_layout(height=300, margin=dict(l=20,r=20,t=55,b=10))
        st.plotly_chart(gauge, use_container_width=True)
        prob_df = pd.DataFrame({"Outcome":["Survived", "Died"], "Predicted probability (%)":[(1-pdeath)*100, pdeath*100]})
        fig = px.bar(prob_df, x="Outcome", y="Predicted probability (%)", text="Predicted probability (%)", range_y=[0,100], title="Predicted outcome probabilities")
        fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
        fig.update_layout(height=320, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        with st.expander("Review values sent to the model"):
            review = patient.T.reset_index(); review.columns = ["Predictor", "Input value before imputation"]
            st.dataframe(review, use_container_width=True, hide_index=True)
        result = pd.DataFrame({"Outcome":["Predicted class", "Probability of death (%)", "Probability of survival (%)", "Shock Index", "WBC × Temperature"],
                               "Value":["Died" if prediction == 1 else "Survived", round(pdeath*100,2), round((1-pdeath)*100,2), shock, interaction]})
        st.download_button("Download mortality result (CSV)", result.to_csv(index=False).encode("utf-8"), "icu_mortality_prediction.csv", "text/csv", use_container_width=True)
        st.warning("Research prototype only. This model output is not a definitive diagnosis or an independent clinical decision. External validation and clinical evaluation are required.")

else:
    st.header("ICU Length-of-Stay Prediction")
    st.caption("Random Forest classifier • 0 = LOS ≤ 7 days, 1 = LOS > 7 days")
    try:
        model, preprocessor = load_artifacts(LOS_MODEL_PATH, LOS_PREPROCESSOR_PATH)
    except Exception as e:
        st.error("Unable to load the LOS model and fitted preprocessor.")
        st.exception(e)
        st.stop()

    admission_options = get_categories(preprocessor, "Admission_source", [])
    case_options = get_categories(preprocessor, "case_category", [])
    oxy_options = get_categories(preprocessor, "oxygen_support", ["Off oxygen", "Intranasal_O2_support", "Facemask O2_support", "Mechanical ventilation"])
    with st.sidebar:
        st.subheader("LOS model information")
        st.write("**Algorithm:** Random Forest classifier")
        st.write("**Class 0:** LOS ≤ 7 days")
        st.write("**Class 1:** LOS > 7 days")
        st.caption("Missing values are imputed by the fitted training preprocessor.")
        with st.expander("Model checks"):
            st.write("Expected processed features:", getattr(model, "n_features_in_", "Not available"))
            try: st.write("Preprocessor output features:", len(preprocessor.get_feature_names_out()))
            except Exception: pass

    st.info("Enter available patient information. Leave unknown numeric fields blank; the fitted LOS preprocessor handles missing values using training-data means, medians, or most-frequent categories.")
    with st.form("los_form"):
        st.subheader("1. Demographic and admission information")
        c1, c2, c3 = st.columns(3)
        with c1: age_raw = st.text_input("Age (years)", placeholder="Leave blank if unavailable", key="l_age")
        with c2: admission = category_select("Admission source", admission_options, "l_admission")
        with c3: case_category = category_select("Case / diagnosis category", case_options, "l_case")

        st.subheader("2. Vital signs")
        c1, c2, c3 = st.columns(3)
        with c1:
            hr_raw = st.text_input("Heart rate (beats/min)", placeholder="Leave blank if unavailable", key="l_hr")
            sbp_raw = st.text_input("Systolic blood pressure (mmHg)", placeholder="Leave blank if unavailable", key="l_sbp")
            dbp_raw = st.text_input("Diastolic blood pressure (mmHg)", placeholder="Leave blank if unavailable", key="l_dbp")
        with c2:
            rr_raw = st.text_input("Respiratory rate (breaths/min)", placeholder="Leave blank if unavailable", key="l_rr")
            spo2_raw = st.text_input("SpO₂ (%)", placeholder="Leave blank if unavailable", key="l_spo2")
            temp_raw = st.text_input("Temperature (°C; for interaction)", placeholder="Leave blank if unavailable", key="l_temp")
        with c3:
            rbs_raw = st.text_input("Random blood sugar (RBS)", placeholder="Leave blank if unavailable", key="l_rbs")
            pain_raw = st.text_input("Pain score (0–10)", placeholder="Leave blank if unavailable", key="l_pain")
            oxy_support = category_select("Oxygen support category", oxy_options, "l_oxy")

        st.subheader("3. Laboratory measurements")
        c1, c2, c3 = st.columns(3)
        with c1:
            sodium_raw = st.text_input("Sodium", placeholder="Leave blank if unavailable", key="l_sodium")
            potassium_raw = st.text_input("Potassium", placeholder="Leave blank if unavailable", key="l_potassium")
            creatinine_raw = st.text_input("Creatinine", placeholder="Leave blank if unavailable", key="l_creatinine")
        with c2:
            neut_raw = st.text_input("Neutrophil (%)", placeholder="Leave blank if unavailable", key="l_neut")
            lymph_raw = st.text_input("Lymphocyte (%)", placeholder="Leave blank if unavailable", key="l_lymph")
            mono_raw = st.text_input("Monocyte (%)", placeholder="Leave blank if unavailable", key="l_mono")
        with c3:
            plt_raw = st.text_input("Platelet count (PLT)", placeholder="Leave blank if unavailable", key="l_plt")
            hb_raw = st.text_input("Hemoglobin", placeholder="Leave blank if unavailable", key="l_hb")
            ast_raw = st.text_input("AST/SGOT", placeholder="Leave blank if unavailable", key="l_ast")
            wbc_raw = st.text_input("White blood cell count (WBC; for interaction)", placeholder="Leave blank if unavailable", key="l_wbc")
        st.caption("Oxygen support is selected directly as a category; no oxygen flow/consumption value is required.")
        submitted = st.form_submit_button("Predict ICU length of stay", type="primary", use_container_width=True)

    if submitted:
        raw = {"Age":age_raw,"Heart rate":hr_raw,"Systolic blood pressure":sbp_raw,"Diastolic blood pressure":dbp_raw,
               "Respiratory rate":rr_raw,"SpO2":spo2_raw,"Temperature":temp_raw,"RBS":rbs_raw,"Pain score":pain_raw,
               "Sodium":sodium_raw,"Potassium":potassium_raw,"Creatinine":creatinine_raw,"Neutrophil":neut_raw,
               "Lymphocyte":lymph_raw,"Monocyte":mono_raw,"Platelet count":plt_raw,"Hemoglobin":hb_raw,"AST/SGOT":ast_raw,"WBC":wbc_raw}
        try: v = {k:parse_optional(val,k) for k,val in raw.items()}
        except ValueError as e: st.error(str(e)); st.stop()
        errors = check_ranges(v)
        if errors:
            st.error("Correct these values before prediction:")
            for e in errors: st.markdown(f"- {e}")
            st.stop()
        shock = v["Heart rate"]/(v["Systolic blood pressure"]+1e-6) if pd.notna(v["Heart rate"]) and pd.notna(v["Systolic blood pressure"]) else np.nan
        interaction = v["WBC"]*v["Temperature"] if pd.notna(v["WBC"]) and pd.notna(v["Temperature"]) else np.nan
        patient = pd.DataFrame([{
            "dbp":v["Diastolic blood pressure"], "Shock_Index":shock, "spo2":v["SpO2"], "rbs":v["RBS"],
            "sodium":v["Sodium"], "potassium":v["Potassium"], "neutrophil":v["Neutrophil"], "lymphocyte":v["Lymphocyte"],
            "plt":v["Platelet count"], "Admission_source":admission, "case_category":case_category, "Age_years":v["Age"],
            "oxygen_support":oxy_support, "creatinine":v["Creatinine"], "rr":v["Respiratory rate"], "ast_sgot":v["AST/SGOT"],
            "hemoglobin":v["Hemoglobin"], "monocyte":v["Monocyte"], "Pain_score":v["Pain score"],
            "WBC_Temperature_Interaction":interaction,
        }])
        los_cols = ["dbp","Shock_Index","spo2","rbs","sodium","potassium","neutrophil","lymphocyte","plt","Admission_source","case_category","Age_years","oxygen_support","creatinine","rr","ast_sgot","hemoglobin","monocyte","Pain_score","WBC_Temperature_Interaction"]
        patient = patient[los_cols]
        try:
            processed = prepare_patient(preprocessor, patient, model)
            prediction = int(model.predict(processed)[0])
            probs = model.predict_proba(processed)[0]
            classes = list(getattr(model, "classes_", [0,1]))
            p_by_class = {int(c):float(p) for c,p in zip(classes,probs)}
            p_short, p_long = p_by_class.get(0,0.0), p_by_class.get(1,0.0)
        except Exception as e:
            st.error("LOS prediction could not be generated."); st.exception(e); st.stop()

        st.divider(); st.subheader("Length-of-stay prediction result")
        if prediction == 1: st.warning("**Predicted class: ICU length of stay > 7 days**")
        else: st.success("**Predicted class: ICU length of stay ≤ 7 days**")
        c1,c2 = st.columns(2)
        c1.metric("Probability of LOS ≤ 7 days", f"{p_short:.1%}")
        c2.metric("Probability of LOS > 7 days", f"{p_long:.1%}")
        gauge = go.Figure(go.Indicator(mode="gauge+number", value=p_long*100, number={"suffix":"%"}, title={"text":"Estimated probability of ICU stay > 7 days"}, gauge={"axis":{"range":[0,100]},"bar":{"color":"#2563eb"}}))
        gauge.update_layout(height=300, margin=dict(l=20,r=20,t=55,b=10))
        st.plotly_chart(gauge, use_container_width=True)
        with st.expander("Review values sent to the model"):
            review = patient.T.reset_index(); review.columns=["Feature","Input value before imputation"]
            st.dataframe(review,use_container_width=True,hide_index=True)
        result = pd.DataFrame({"Outcome":["Predicted class","Probability of LOS ≤ 7 days (%)","Probability of LOS > 7 days (%)","Shock Index","WBC × Temperature"],
                               "Value":["LOS > 7 days" if prediction==1 else "LOS ≤ 7 days",round(p_short*100,2),round(p_long*100,2),shock,interaction]})
        st.download_button("Download LOS result (CSV)",result.to_csv(index=False).encode("utf-8"),"icu_los_prediction.csv","text/csv",use_container_width=True)
        st.warning("Research prototype only. This output is not a clinical diagnosis or a substitute for clinician judgment. External validation and clinical evaluation are required.")

st.divider()
st.markdown('<div class="footer">ICU Mortality Risk and Length-of-Stay Prediction • Research prototype<br>Developed by Mubarik Meka<br>For research and educational demonstration; not for independent clinical decision-making.</div>', unsafe_allow_html=True)
