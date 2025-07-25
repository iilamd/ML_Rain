import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.impute import SimpleImputer

st.set_page_config(layout="wide", page_title="Prediksi Hujan Besok 🌧️")

st.title("Prediksi Hujan Besok 🌧️")
st.write("""
Aplikasi ini memungkinkan Anda untuk mengeksplorasi dataset cuaca Australia dan memprediksi apakah akan hujan besok,
menggunakan model Machine Learning.
""")

# Sidebar untuk navigasi
st.sidebar.header("Pengaturan Aplikasi")
analysis_type = st.sidebar.radio(
    "Pilih Tipe Analisis:",
    ("Gambaran Umum Data", "Analisis Data Eksplorasi (EDA)", "Prediksi Hujan Besok")
)

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("weatherAUS.csv")
    return df

df = load_data()

# Pra-pemrosesan sederhana
# Drop kolom yang jelas non-numerik
df = df.drop(columns=["Date", "Location"], errors="ignore")
df = df.dropna(subset=["RainTomorrow"])
df["RainTomorrow"] = df["RainTomorrow"].map({"No": 0, "Yes": 1})

# Pilih hanya kolom numerik + target
numeric_cols = df.select_dtypes(include="number").columns.tolist()
if "RainTomorrow" not in numeric_cols:
    numeric_cols.append("RainTomorrow")
df = df[numeric_cols]

# Imputasi nilai hilang
imputer = SimpleImputer(strategy="mean")
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)


X = df_imputed.drop("RainTomorrow", axis=1)
y = df_imputed["RainTomorrow"]

if analysis_type == "Gambaran Umum Data":
    st.header("Gambaran Umum Dataset")
    st.write("Berikut adalah 5 baris pertama:")
    st.dataframe(df.head())

    st.write("Bentuk Dataset:", df.shape)
    st.write("Statistik Deskriptif:")
    st.dataframe(df.describe())

    st.write("Distribusi Label (RainTomorrow):")
    st.bar_chart(y.value_counts())

elif analysis_type == "Analisis Data Eksplorasi (EDA)":
    st.header("Analisis Data Eksplorasi (EDA)")

    feature_columns = X.columns.tolist()
    selected_feature = st.selectbox(
        "Pilih Fitur untuk Visualisasi Distribusi:",
        feature_columns
    )

    st.subheader(f"Histogram Distribusi {selected_feature}")
    fig_hist = px.histogram(df_imputed, x=selected_feature, color='RainTomorrow',
                            color_discrete_map={0: 'blue', 1: 'orange'},
                            barmode="overlay", nbins=30)
    st.plotly_chart(fig_hist, use_container_width=True)

    st.subheader("Heatmap Korelasi")
    corr_matrix = df_imputed.corr()
    fig_corr = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="Blues",
        title='Heatmap Korelasi'
    )
    st.plotly_chart(fig_corr, use_container_width=True)

elif analysis_type == "Prediksi Hujan Besok":
    st.header("Prediksi Hujan Besok")

    st.subheader("Pelatihan Model")
    test_size = st.slider("Proporsi Data Uji:", 0.1, 0.5, 0.2, 0.05)
    random_state = st.slider("Random State:", 0, 100, 42)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    n_estimators = st.slider("Jumlah Estimator (pohon) Random Forest:", 50, 500, 100, 50)
    max_depth = st.slider("Kedalaman Maksimum Pohon:", 3, 20, 10, 1)

    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.subheader("Evaluasi Model")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Akurasi", f"{accuracy_score(y_test, y_pred):.2f}")
    col2.metric("Presisi", f"{precision_score(y_test, y_pred):.2f}")
    col3.metric("Recall", f"{recall_score(y_test, y_pred):.2f}")
    col4.metric("F1-Score", f"{f1_score(y_test, y_pred):.2f}")

    st.subheader("Input Data untuk Prediksi Baru")

    input_data = {}
    for col in X.columns:
        min_val = float(df_imputed[col].min())
        max_val = float(df_imputed[col].max())
        mean_val = float(df_imputed[col].mean())
        step = (max_val - min_val) / 100 if max_val != min_val else 1.0
        input_data[col] = st.number_input(col, min_val, max_val, mean_val, step=step)

    if st.button("Prediksi"):
        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)[0]
        prediction_proba = model.predict_proba(input_df)[0]

        if prediction == 1:
            st.success(f"🌧️ Diprediksi akan HUJAN besok ({prediction_proba[1]*100:.2f}%)")
        else:
            st.info(f"☀️ Diprediksi TIDAK HUJAN besok ({prediction_proba[0]*100:.2f}%)")

st.markdown("---")
st.write("Aplikasi ini menggunakan Random Forest Classifier dan dataset weatherAUS.csv")
