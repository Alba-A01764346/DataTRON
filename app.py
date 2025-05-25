import streamlit as st
import pandas as pd
import joblib
import openai

# --- CONFIGURACIÓN GENERAL ---
openai.api_key = "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"
st.set_page_config(page_title="DATATRON | Asistente OXXO", layout="wide")

# --- ESTILOS PERSONALIZADOS ---
st.markdown("""
    <style>
        html, body, [data-testid="stAppViewContainer"] {
            background: linear-gradient(160deg, #b3d9f2, #dbefff, #f0f8ff);
            background-attachment: fixed;
            color: #000000;
        }
        .stTextInput > div > input {
            background-color: #ffffff !important;
            color: #000000 !important;
            border: 1px solid #ccc;
            padding: 0.5rem;
        }
        .stButton > button,
        .stDownloadButton {
            background-color: #ffffff !important;
            color: #003366 !important;
            border: 1px solid #999;
            font-weight: bold;
        }
        .stMarkdown, .stHeader, .stDataFrame th, .stDataFrame td {
            color: #000000 !important;
        }
        .stSuccess {
            background-color: #e6f9ec !important;
            color: #14532d !important;
            border-left: 5px solid #2e7d32;
            padding: 1rem;
            border-radius: 0.4rem;
        }
        h1, h2, h3 {
            color: #002244 !important;
        }
    </style>
""", unsafe_allow_html=True)

# --- LOGO Y TÍTULO ---
st.image("datatron.png", width=300)
st.markdown("<h2 style='text-align: center; color: #002244;'>Modelos de Machine Learning de OXXO</h2>", unsafe_allow_html=True)
st.markdown("---")

# --- CARGAR MODELO ---
modelo = joblib.load("modelo_xgboost_exito.pkl")
columnas_modelo = modelo.get_booster().feature_names

# --- SECCIÓN PREDICCIÓN ---
st.header("📈 Prueba nuestro modelo de predicción de éxito de tienda")
archivo = st.file_uploader("Sube tu archivo CSV", type=["csv"])

if archivo is not None:
    df = pd.read_csv(archivo)
    st.write("✅ Vista previa del archivo cargado:")
    st.dataframe(df.head())

    # --- Preprocesamiento ---
    cat_cols = [
        'ZONA', 'NIVELSOCIOECONOMICO_DES', 'ENTORNO_DES', 'SEGMENTO_MAESTRO_DESC',
        'LID_UBICACION_TIENDA', 'DATASET', 'Supermarket_200m',
        'convenience_200m', 'tipo_calle', 'SEGMENTO_DIRIGIDO'
    ]

    # Eliminar columna objetivo si está presente
    if "PRED_VENTAS" in df.columns:
        df = df.drop(columns=["PRED_VENTAS"])

    # Convertir categóricas y aplicar One-Hot
    df[cat_cols] = df[cat_cols].astype(str)
    df = pd.get_dummies(df, columns=cat_cols)

    # Rellenar columnas faltantes esperadas por el modelo
    for col in columnas_modelo:
        if col not in df.columns:
            df[col] = 0

    # Reordenar columnas en el orden del modelo
    df = df[columnas_modelo]

    # --- PREDICCIÓN ---
    if st.button("🎯 Predecir éxito de tienda"):
        predicciones = modelo.predict(df)
        df["Predicción_Éxito"] = predicciones
        st.success("✅ Predicciones generadas correctamente.")
        st.dataframe(df[["Predicción_Éxito"]])

        # Descargar archivo resultante
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Descargar predicciones", data=csv,
                           file_name="predicciones_oxxo.csv", mime="text/csv")

# --- SECCIÓN CHATBOT ---
st.markdown("---")
st.title("💬 Chat con el experto en tiendas OXXO")
st.subheader("Haz una pregunta")
pregunta = st.text_input("¿Qué quieres saber sobre OXXO?")

if pregunta:
    respuesta = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "Eres un experto en tiendas OXXO. Da recomendaciones claras y útiles."},
            {"role": "user", "content": pregunta}
        ]
    )
    st.success(respuesta.choices[0].message.content.strip())
