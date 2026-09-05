import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import plotly.express as px

st.set_page_config(
    page_title="Sports Predictor AI",
    page_icon="⚽",
    layout="wide"
)

st.title("⚽ Sports Predictor AI")

st.markdown("""
Plataforma de análisis predictivo para apuestas deportivas.

Esta herramienta utiliza modelos de Machine Learning para generar probabilidades
estimadas y detectar potenciales oportunidades de valor.
""")

st.sidebar.header("Configuración")

uploaded_file = st.sidebar.file_uploader(
    "Cargar dataset",
    type=["csv", "xlsx", "xls"]
)

if uploaded_file is None:
    st.info("""
    Cargue un archivo CSV, XLSX o XLS para comenzar.

    El dataset debe incluir variables estadísticas, una variable objetivo y datos
    históricos suficientes para entrenar el modelo.
    """)
    st.stop()


@st.cache_data
def load_data(file):
    """Carga un archivo CSV o Excel desde la memoria."""
    file_name = file.name.lower()
    if file_name.endswith(".csv"):
        return pd.read_csv(file)
    if file_name.endswith(".xlsx"):
        return pd.read_excel(file, engine="openpyxl")
    if file_name.endswith(".xls"):
        return pd.read_excel(file, engine="xlrd")
    raise ValueError("Formato de archivo no admitido.")


try:
    df = load_data(uploaded_file)
except Exception as error:
    st.error(f"No fue posible procesar el archivo: {error}")
    st.stop()

if df.empty:
    st.warning("El archivo no contiene registros.")
    st.stop()

df.columns = df.columns.astype(str).str.strip()
st.success(f"Archivo cargado: {uploaded_file.name}")

c1, c2, c3 = st.columns(3)
c1.metric("Filas", len(df))
c2.metric("Columnas", len(df.columns))
c3.metric("Valores faltantes", int(df.isna().sum().sum()))

st.divider()
st.subheader("Vista previa")
st.dataframe(df.head(50), use_container_width=True, hide_index=True)

st.subheader("Configuración del modelo")

target = st.selectbox("Seleccione la variable objetivo", df.columns)

numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
if target in numeric_columns:
    numeric_columns.remove(target)

features = st.multiselect(
    "Variables predictoras numéricas",
    numeric_columns,
    default=numeric_columns[:5]
)

if not features:
    st.warning("Seleccione al menos una variable predictora numérica.")
    st.stop()

dataset = df[features + [target]].dropna()

if dataset.empty:
    st.warning("No hay registros completos para entrenar el modelo con la selección actual.")
    st.stop()

if dataset[target].nunique() < 2:
    st.warning("La variable objetivo debe contener al menos dos clases diferentes.")
    st.stop()

if len(dataset) < 10:
    st.warning("Se requieren al menos 10 registros completos para realizar una división básica de entrenamiento y prueba.")
    st.stop()

X = dataset[features]
y = dataset[target]

try:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y if y.value_counts().min() >= 2 else None
    )
except ValueError:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

model_option = st.selectbox(
    "Modelo",
    ["Random Forest", "Regresión logística"]
)

if model_option == "Random Forest":
    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced"
    )
else:
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    )

try:
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
except Exception as error:
    st.error(f"No fue posible entrenar el modelo: {error}")
    st.stop()

accuracy = accuracy_score(y_test, preds)
precision = precision_score(y_test, preds, average="weighted", zero_division=0)
recall = recall_score(y_test, preds, average="weighted", zero_division=0)
f1 = f1_score(y_test, preds, average="weighted", zero_division=0)

st.subheader("Rendimiento del modelo")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Accuracy", f"{accuracy:.2%}")
m2.metric("Precision", f"{precision:.2%}")
m3.metric("Recall", f"{recall:.2%}")
m4.metric("F1 Score", f"{f1:.2%}")

if model_option == "Random Forest":
    importance = pd.DataFrame({
        "Variable": features,
        "Importancia": model.feature_importances_
    }).sort_values("Importancia", ascending=True)

    st.subheader("Variables más importantes")
    fig = px.bar(
        importance,
        x="Importancia",
        y="Variable",
        orientation="h",
        title="Importancia de variables"
    )
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Predicciones")

prediction_df = pd.DataFrame({
    "Fila_original": X_test.index,
    "Resultado_real": y_test.values,
    "Prediccion": preds,
    "Confianza_pct": probabilities.max(axis=1) * 100
})

prediction_df["Nivel_confianza"] = np.select(
    [
        prediction_df["Confianza_pct"] >= 75,
        prediction_df["Confianza_pct"] >= 60
    ],
    ["Alta", "Media"],
    default="Baja"
)

st.dataframe(prediction_df, use_container_width=True, hide_index=True)

st.subheader("Value Betting")
cuota = st.slider("Cuota decimal de la casa", 1.10, 10.00, 2.00, 0.05)
prediction_df["Cuota"] = cuota
prediction_df["Valor_esperado"] = (
    prediction_df["Confianza_pct"] / 100 * cuota
) - 1
prediction_df["Clasificacion_value"] = np.where(
    prediction_df["Valor_esperado"] > 0,
    "Value Bet potencial",
    "Sin valor estimado"
)

st.dataframe(
    prediction_df[[
        "Fila_original", "Prediccion", "Confianza_pct", "Cuota",
        "Valor_esperado", "Clasificacion_value"
    ]],
    use_container_width=True,
    hide_index=True
)

st.subheader("Kelly Criterion")
bankroll = st.number_input(
    "Bankroll disponible",
    min_value=0.0,
    value=1000.0,
    step=50.0
)
fraccion_kelly = st.slider("Fracción de Kelly", 0.10, 1.00, 0.25, 0.05)
prob = float(prediction_df["Confianza_pct"].mean() / 100)
b = cuota - 1
kelly = ((prob * b) - (1 - prob)) / b if b > 0 else 0
kelly = max(0.0, min(kelly, 1.0))
stake = bankroll * kelly * fraccion_kelly

k1, k2 = st.columns(2)
k1.metric("Kelly completo", f"{kelly:.2%}")
k2.metric("Stake orientativo", f"${stake:,.2f}")

st.subheader("Distribución de confianza")
fig2 = px.histogram(
    prediction_df,
    x="Confianza_pct",
    nbins=20,
    title="Distribución de la confianza del modelo"
)
st.plotly_chart(fig2, use_container_width=True)

st.subheader("Top Picks")
top = prediction_df.sort_values(
    ["Valor_esperado", "Confianza_pct"],
    ascending=False
).head(10)
st.dataframe(top, use_container_width=True, hide_index=True)

csv = prediction_df.to_csv(index=False).encode("utf-8-sig")
st.download_button(
    "Descargar predicciones",
    data=csv,
    file_name="predicciones.csv",
    mime="text/csv"
)

st.warning("""
Las predicciones son estimaciones estadísticas y no garantizan resultados futuros.
Esta aplicación es exclusivamente informativa. Las apuestas deportivas implican
riesgo de pérdida de dinero. Evite apostar recursos necesarios para sus gastos y
cumpla la legislación y los requisitos de edad aplicables en su jurisdicción.
""")
