# Importing Libraries
import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Streamlit page config
st.set_page_config(page_title="Laptop Price Predictor", page_icon="💻", layout="wide")

# Title
st.title("Laptop Price Predictor 💻")

# Load pipeline and DataFrame
pipe = pickle.load(open("pipe.pkl", "rb"))
df = pickle.load(open("df.pkl", "rb"))

# OS mapping to match training
def map_os_category(op_sys):
    if "mac" in op_sys.lower():
        return "Mac"
    elif "windows" in op_sys.lower():
        return "Windows"
    else:
        return "Others"

# 1st row of inputs
left_column, middle_column, right_column = st.columns(3)
with left_column:
    company = st.selectbox("Brand", df["Company"].unique())

with middle_column:
    type_ = st.selectbox("Type", df["TypeName"].unique())

with right_column:
    ram = st.selectbox("Ram (in GB)", sorted(df["Ram"].unique()))

# 2nd row of inputs
left_column, middle_column, right_column = st.columns(3)
with left_column:
    weight = st.number_input("Weight of laptop in kg", min_value=0.0, format="%.2f")

with middle_column:
    touchscreen = st.selectbox("Touchscreen", ["No", "Yes"])

with right_column:
    ips = st.selectbox("IPS Display", ["No", "Yes"])

# 3rd row of inputs
left_column, middle_column, right_column = st.columns(3)
with left_column:
    screen_size = st.number_input("Screen Size (in Inches)", min_value=1.0, format="%.1f")

with middle_column:
    resolution = st.selectbox(
        "Screen Resolution",
        ['1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800',
         '2880x1800', '2560x1600', '2560x1440', '2304x1440']
    )

with right_column:
    cpu = st.selectbox("CPU Brand", df["Cpu brand"].unique())

# 4th row of inputs
left_column, right_column = st.columns(2)
with left_column:
    hdd = st.selectbox("HDD (in GB)", [0, 128, 256, 512, 1024, 2048])

with right_column:
    ssd = st.selectbox("SSD (in GB)", [0, 8, 128, 256, 512, 1024])

# Remaining inputs
gpu = st.selectbox("GPU Brand", df["Gpu brand"].unique())
os_input = st.selectbox("OS Type", df["OpSys"].unique())

# Prediction button
if st.button("Predict Price"):
    # Encode yes/no to 1/0
    touchscreen_val = 1 if touchscreen == "Yes" else 0
    ips_val = 1 if ips == "Yes" else 0

    # Calculate PPI
    X_res = int(resolution.split("x")[0])
    Y_res = int(resolution.split("x")[1])
    ppi = ((X_res ** 2) + (Y_res ** 2)) ** 0.5 / screen_size

    # Create engineered 'os' feature
    os_engineered = map_os_category(os_input)

    # Create DataFrame with same columns as training
    query_df = pd.DataFrame([[
        company,         # Company
        type_,           # TypeName
        ram,             # Ram
        os_input,        # OpSys
        weight,          # Weight
        touchscreen_val, # Touchscreen
        ips_val,         # Ips
        ppi,             # ppi
        cpu,             # Cpu brand
        hdd,             # HDD
        ssd,             # SSD
        gpu,             # Gpu brand
        os_engineered    # os (engineered category)
    ]], columns=[
        "Company", "TypeName", "Ram", "OpSys", "Weight",
        "Touchscreen", "Ips", "ppi", "Cpu brand", "HDD", "SSD", "Gpu brand", "os"
    ])

    # Predict and inverse log-transform
    predicted_price = np.exp(pipe.predict(query_df))[0]

    st.success(f"The Predicted Price of Laptop = ₹ {int(predicted_price):,}")
