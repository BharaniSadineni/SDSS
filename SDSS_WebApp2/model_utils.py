import pandas as pd
import numpy as np
import joblib

MODEL_DIR = "models/"

# ===============================
# SCENARIO 1: MORPHOLOGY
# ===============================
def predict_scenario1(df):
    model = joblib.load(MODEL_DIR + "SDSS_S1_USC_LRC.pkl")

    df["u_g"] = df["u"] - df["g"]
    df["g_r"] = df["g"] - df["r"]
    df["r_i"] = df["r"] - df["i"]
    df["i_z"] = df["i"] - df["z"]

    features = [
        "u_g", "g_r", "r_i", "i_z",
        "petroR50_g", "petroR50_r", "petroR50_i",
        "expAB_g", "expAB_r", "expAB_i"
    ]

    preds = model.predict(df[features])

    label_map = {0: "Spiral", 1: "Elliptical", 2: "Irregular"}
    return [label_map[p] for p in preds]

# ===============================
# SCENARIO 2: REDSHIFT
# ===============================
def predict_scenario2(df):
    model = joblib.load(MODEL_DIR + "SDSS_S2_Reg_XGB.pkl")
    medians = joblib.load(MODEL_DIR + "SDSS_feature_medians.pkl")
    columns = joblib.load(MODEL_DIR + "SDSS_feature_columns.pkl")

    X = pd.DataFrame()

    for col in columns:
        if col in df.columns:
            X[col] = df[col]
        else:
            X[col] = medians[col]

    return model.predict(X).tolist()

# ===============================
# SCENARIO 3: AGN
# ===============================
def predict_scenario3(df):
    model = joblib.load(MODEL_DIR + "SDSS_S3_USC_XGB.pkl")

    df["u_g"] = df["u"] - df["g"]
    df["g_r"] = df["g"] - df["r"]
    df["r_i"] = df["r"] - df["i"]
    df["i_z"] = df["i"] - df["z"]

    for b in ["u", "g", "r", "i", "z"]:
        df[f"compact_{b}"] = df[f"psfMag_{b}"] - df[f"modelFlux_{b}"]

    df["flux_u_g"] = df["modelFlux_u"] / (df["modelFlux_g"] + 1e-5)
    df["flux_g_r"] = df["modelFlux_g"] / (df["modelFlux_r"] + 1e-5)
    df["flux_r_i"] = df["modelFlux_r"] / (df["modelFlux_i"] + 1e-5)
    df["flux_i_z"] = df["modelFlux_i"] / (df["modelFlux_z"] + 1e-5)

    features = [
        "u", "g", "r", "i", "z",
        "psfMag_u", "psfMag_g", "psfMag_r", "psfMag_i", "psfMag_z",
        "modelFlux_u", "modelFlux_g", "modelFlux_r", "modelFlux_i", "modelFlux_z",
        "u_g", "g_r", "r_i", "i_z",
        "compact_u", "compact_g", "compact_r", "compact_i", "compact_z",
        "flux_u_g", "flux_g_r", "flux_r_i", "flux_i_z"
    ]

    preds = model.predict(df[features])
    return ["AGN" if p == 1 else "Non-AGN" for p in preds]
