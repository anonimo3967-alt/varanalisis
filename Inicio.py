import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ============================================================
# Configuración de la página
# ============================================================
st.set_page_config(
    page_title="Datos de seguridad del banco de Lyon",
    page_icon="🏦",
    layout="wide"
)

st.markdown("""
    <style>
    .main { padding: 2rem; }
    .stAlert { margin-top: 1rem; }
    </style>
""", unsafe_allow_html=True)

CANDIDATAS_TIEMPO = ["time", "_time", "timestamp", "fecha", "date", "datetime"]


# ============================================================
# Preparación de datos
# ============================================================
def preparar_datos(df):
    """Detecta la columna de tiempo y las columnas numéricas del CSV."""
    df = df.copy()

    # 1. Columna de tiempo → índice
    col_tiempo = next(
        (c for c in df.columns if str(c).strip().lower() in CANDIDATAS_TIEMPO),
        None
    )
    if col_tiempo is not None:
        df[col_tiempo] = pd.to_datetime(df[col_tiempo], errors="coerce")
        df = df.dropna(subset=[col_tiempo]).set_index(col_tiempo).sort_index()
        df.index.name = "Time"

    # 2. Convertir texto numérico
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            convertida = pd.to_numeric(
                df[c].astype(str).str.replace(",", ".", regex=False),
                errors="coerce"
            )
            if convertida.notna().mean() >= 0.9:
                df[c] = convertida

    # 3. Variables numéricas disponibles
    numericas = df.select_dtypes(include="number").columns.tolist()
    return df, numericas, col_tiempo


def normalizar(df_num):
    """Escala únicamente las columnas numéricas a 0–1."""
    rango = (df_num.max() - df_num.min()).replace(0, 1)
    return (df_num - df_num.min()) / rango


# ============================================================
# Encabezado y mapa
# ============================================================
st.title("🏦 Análisis de seguridad para que no se meta un ladrooooon, o si se metio al menos saber")
st.markdown("""
    Esta aplicación permite analizar datos enviados por el sistema de seguridad Lyon™ para matar a un ladron en caso de ser necesario
""")

st.image("guardia_de_seguridad.jpeg")

eafit_location = pd.DataFrame({
    "lat": [45.7640],
    "lon": [4.8357],
    "location": ["Lyon - Francia"]
})

st.subheader("📍 Ubicación del sistema - Lyon")
st.map(eafit_location, zoom=15)

# ============================================================
# Carga de archivo(s) CSV
# ============================================================
uploaded_files = st.file_uploader(
    "Seleccione uno o varios archivos CSV", 
    type=["csv"], 
    accept_multiple_files=True
)

if uploaded_files:
    try:
        # Cargar y unir múltiples archivos si se sube más de uno
        lista_dfs = []
        for file in uploaded_files:
            df_temp = pd.read_csv(file)
            lista_dfs.append(df_temp)
        
        df_original = pd.concat(lista_dfs, ignore_index=True)
        df1, variables, col_tiempo = preparar_datos(df_original)

        if not variables:
            st.error("No se encontraron columnas numéricas en los archivos cargados.")
            st.stop()

        st.success(
            f"Se cargaron **{len(uploaded_files)}** archivo(s) exitosamente. "
            f"Se detectaron **{len(variables)}** variable(s) numérica(s): "
            + ", ".join(f"`{v}`" for v in variables)
        )
        if col_tiempo is None:
            st.info("No se encontró columna de tiempo; se usará el índice de registros como eje X.")

        tab1, tab2, tab3, tab4 = st.tabs(
            ["📈 Visualización Multivariable", "📊 Estadísticas y Correlación", "🔍 Filtros", "🗺️ Información del Sitio"]
        )

        # ----------------------------------------------------
        # TAB 1 · Visualización Multivariable
        # ----------------------------------------------------
        with tab1:
            st.subheader("Visualización Avanzada de Múltiples Sensores")

            col_a, col_b, col_c = st.columns([2, 1, 1])
            with col_a:
                vars_graf = st.multiselect(
                    "Variables a graficar simultáneamente",
                    variables,
                    default=variables if len(variables) <= 4 else variables[:3]
                )
            with col_b:
                chart_type = st.selectbox("Tipo de gráfico", ["Líneas", "Áreas", "Dispersión (Scatter)"])
            with col_c:
                normalizado = st.checkbox(
                    "Normalizar escalas (0–1)",
                    help="Escala las variables seleccionadas al rango 0-1 para compararlas juntas."
                )

            if not vars_graf:
                st.warning("Seleccione al menos una variable del menú desplegable.")
            else:
                # Tomar los datos seleccionados
                df_sub = df1[vars_graf].copy()

                if normalizado:
                    df_sub = normalizar(df_sub)

                # Construir el gráfico con Plotly
                fig = go.Figure()

                # Definir eje X (el índice de tiempo o rango numérico)
                eje_x = df_sub.index

                for var in vars_graf:
                    if chart_type == "Líneas":
                        fig.add_trace(go.Scatter(x=eje_x, y=df_sub[var], mode='lines', name=var))
                    elif chart_type == "Áreas":
                        fig.add_trace(go.Scatter(x=eje_x, y=df_sub[var], mode='lines', name=var, stackgroup='one'))
                    else:
                        fig.add_trace(go.Scatter(x=eje_x, y=df_sub[var], mode='markers', name=var))

                fig.update_layout(
                    title="Comparación Multivariable de Sensores",
                    xaxis_title="Tiempo" if col_tiempo else "Registro N°",
                    yaxis_title="Escala Normalizada (0–1)" if normalizado else "Valor de la variable",
                    hovermode="x unified",
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)

            if st.checkbox("Mostrar datos crudos"):
                st.dataframe(df1)

        # ----------------------------------------------------
        # TAB 2 · Estadísticas
        # ----------------------------------------------------
        with tab2:
            st.subheader("Análisis Estadístico y Multivariable")

            st.write("#### Resumen de todas las variables")
            st.dataframe(df1[variables].describe().T.round(2))

            st.write("#### Detalle por variable individual")
            var_stats = st.selectbox("Variable", variables, key="var_stats")
            serie = df1[var_stats].dropna()

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Promedio", f"{serie.mean():.2f}")
            m2.metric("Máximo", f"{serie.max():.2f}")
            m3.metric("Mínimo", f"{serie.min():.2f}")
            m4.metric("Desv. estándar", f"{serie.std():.2f}")

            if len(variables) >= 2:
                st.write("#### Matriz de Correlación Cruzada entre Sensores")
                corr_matrix = df1[variables].corr().round(2)
                
                fig_corr = px.imshow(
                    corr_matrix, 
                    text_auto=True, 
                    color_continuous_scale="RdBu_r", 
                    zmin=-1, zmax=1,
                    title="Mapa de Calor de Correlación"
                )
                st.plotly_chart(fig_corr, use_container_width=True)

        # ----------------------------------------------------
        # TAB 3 · Filtros
        # ----------------------------------------------------
        with tab3:
            st.subheader("Filtros de Datos")

            var_filtro = st.selectbox("Variable a filtrar", variables, key="var_filtro")
            min_value = float(df1[var_filtro].min())
            max_value = float(df1[var_filtro].max())

            if min_value == max_value:
                st.warning(f"⚠️ Todos los valores de `{var_filtro}` son iguales: {min_value:.2f}")
                st.dataframe(df1)
            else:
                rango = st.slider(
                    f"Rango de {var_filtro}",
                    min_value,
                    max_value,
                    (min_value, max_value),
                    key=f"rango_{var_filtro}"
                )

                filtrado = df1[df1[var_filtro].between(rango[0], rango[1])]
                st.write(f"**{len(filtrado)}** de {len(df1)} registros filtrados:")
                st.dataframe(filtrado)

                st.download_button(
                    label="⬇️ Descargar datos filtrados (CSV)",
                    data=filtrado.to_csv().encode("utf-8"),
                    file_name=f"datos_filtrados_{var_filtro}.csv",
                    mime="text/csv",
                )

        # ----------------------------------------------------
        # TAB 4 · Información del sitio
        # ----------------------------------------------------
        with tab4:
            st.subheader("Información del Sitio de Medición")

            col1, col2 = st.columns(2)
            with col1:
                st.write("### Ubicación del Sensor")
                st.write("**Lyon, Francia**")
                st.write("- Latitud: 6.2006")
                st.write("- Longitud: -75.5783")
                st.write("- Altitud: ~1,495 msnm")

            with col2:
                st.write("### Detalles del Sensor")
                st.write("- Tipo: ESP32")
                st.write("- Variables medidas: " + ", ".join(variables))
                st.write(f"- Número total de registros: {len(df1)}")
                if col_tiempo is not None and len(df1) > 0:
                    st.write(f"- Desde: {df1.index.min():%Y-%m-%d %H:%M}")
                    st.write(f"- Hasta: {df1.index.max():%Y-%m-%d %H:%M}")
                st.write("- Ubicación: banco de Lyon")

    except Exception as e:
        st.error(f"Error al procesar el archivo: {str(e)}")
        st.info("Asegúrese de que el archivo CSV tenga al menos una columna numérica.")
else:
    st.warning("Por favor, cargue uno o varios archivos CSV para comenzar el análisis.")

# ============================================================
# Pie de página
# ============================================================
st.markdown("""
    ---
    Desarrollado para el análisis de datos de seguridad para bancos.
    Ubicación: Lyon, Francia
""")
