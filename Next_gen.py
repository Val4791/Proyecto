import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from fuzzywuzzy import process
import groq
import os
os.system("pip install groq")
import groq
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns

# 📌 Configuración de la página
st.set_page_config(page_title="Análisis de Jugadores", layout="wide")

# 📌 Configurar la API de Groq
GROQ_API_KEY = "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"
client = groq.Client(api_key=GROQ_API_KEY)

# 📌 Función para normalizar nombres con Groq
def normalizar_nombre_groq(nombre_usuario):
    prompt = f"""
    Eres un experto en fútbol con conocimientos actualizados sobre jugadores de todas las ligas.

    Tarea:
    - Si "{nombre_usuario}" es un apodo, abreviación o variación del nombre de un jugador de fútbol, conviértelo a su nombre completo y correcto.
    - Si ya está bien escrito, devuélvelo igual.
    - Si el nombre no pertenece a ningún futbolista conocido, responde solo con "Desconocido" (sin comillas).

    Ejemplos:
    - "Vini jr" → "Vinícius José de Oliveira Júnior"
    - "CR7" → "Cristiano Ronaldo"
    - "Leo" → "Lionel Messi"
    - "K Mbappe" → "Kylian Mbappé"
    - "Pedri" → "Pedro González López"
    - "Pelé" → "Edson Arantes do Nascimento"

    Responde solo con el nombre corregido. No agregues explicaciones ni texto adicional.
    """
    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "system", "content": "Eres un experto en fútbol."},
                  {"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

# 📌 Función para justificar similitudes con Groq
def justificar_similitudes(jugador1, jugador2):
    prompt = f"""
    Compara a {jugador1} y {jugador2} en atributos como velocidad, tiro, pase, regate, defensa y físico.
    Explica en qué aspectos son similares y en qué se diferencian.
    """
    response = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[{"role": "system", "content": "Eres un experto en fútbol."},
                  {"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

# 📌 Cargar datos principales
@st.cache_data
def cargar_datos_principales():
    return pd.read_csv("skills_resultado.csv")

df = cargar_datos_principales()
jugadores = df["name"].unique()

# 📌 Estilos globales
st.markdown("""
<style>
    .main, .stApp {
        background-color: black;
        color: white;
    }
    .stButton>button {
        background-color: #FFA500;
        color: black;
        font-size: 16px;
        padding: 8px 16px;
        border-radius: 10px;
    }
    .stTextInput>div>div>input {
        color: white !important;
        width: 75% !important;
    }
    .title-container {
        text-align: center;
        font-size: 34px;
        font-weight: bold;
        margin-bottom: 10px;
        border-bottom: 3px solid #FFA500;
        padding-bottom: 5px;
        display: inline-block;
    }
    .full-width {
        width: 100%;
        display: block;
        text-align: justify;
        font-size: 18px;
        padding: 10px;
    }
    .stImage {
        display: flex;
        justify-content: flex-end;
        margin-top: -100px;
    }
</style>
""", unsafe_allow_html=True)

def image_to_base64(image):
    import base64
    from io import BytesIO
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

# 📌 Mostrar logo en la parte superior derecha
try:
    image = Image.open("Next gen.jpg")
    st.markdown(
        """
        <div style="display: flex; justify-content: flex-end;">
            <img src="data:image/jpeg;base64,{}" width="300" height="150">
        </div>
        """.format(image_to_base64(image)),
        unsafe_allow_html=True
    )
except:
    st.write("Logo 'Next gen.jpg' no encontrado. Continúa sin mostrar la imagen...")

# ----------------------------------------
# 📌 Menú Principal en barra lateral y Filtros Recomendador
# ----------------------------------------
with st.sidebar:
    st.markdown("## Menú Principal")
    main_pages = ["Analítica", "Jugadores", "Recomendador", "Comparador"]
    selected_page = st.radio("Navegación", main_pages)

    if selected_page == "Recomendador":
        st.markdown("<hr>", unsafe_allow_html=True)
        st.subheader("Filtros para Recomendador")

        # Cargar dataset
        df_skills_filter = pd.read_csv(r'skills_resultado.csv') # Use a different name to avoid conflict

        price_range = st.slider("Precio (millones de euros)",
            float(df_skills_filter['value_million_euro'].min()),
            float(df_skills_filter['value_million_euro'].max()),
            (float(df_skills_filter['value_million_euro'].min()),
             float(df_skills_filter['value_million_euro'].max()))
        )
        wage_range = st.slider("Salario (millones de euros)",
            float(df_skills_filter['wage_million_euro'].min()),
            float(df_skills_filter['wage_million_euro'].max()),
            (float(df_skills_filter['wage_million_euro'].min()),
             float(df_skills_filter['wage_million_euro'].max()))
        )
        age_range = st.slider("Edad",
            int(df_skills_filter['age'].min()),
            int(df_skills_filter['age'].max()),
            (int(df_skills_filter['age'].min()), int(df_skills_filter['age'].max()))
        )
        height_range = st.slider("Altura (cm)",
            int(df_skills_filter['height'].min()),
            int(df_skills_filter['height'].max()),
            (int(df_skills_filter['height'].min()), int(df_skills_filter['height'].max()))
        )
        position_options = st.multiselect("Posición", ["Todos"] + list(df_skills_filter['position'].unique()), default=["Todos"])
        nationality_options = st.multiselect("Nacionalidad", ["Todos"] + list(df_skills_filter['nationality'].unique()), default=["Todos"])
        preferred_foot = st.radio("Pie Preferido", ['Todos', 'Izquierda', 'Derecha'])

st.markdown("<hr>", unsafe_allow_html=True)

# ----------------------------------------
# Funciones y datos para Recomendador
# ----------------------------------------
def cargar_datos_recomendaciones():
    return pd.read_csv(r'skills_resultado.csv')

df_skills = cargar_datos_recomendaciones()

player_skills = [
    'overall', 'potential', 'skill_moves', 'attacking_work_rate',
    'defensive_work_rate', 'pace_total', 'shooting_total', 'passing_total',
    'dribbling_total', 'defending_total', 'physicality_total', 'finishing',
    'heading_accuracy', 'dribbling', 'ball_control', 'balance', 'shot_power',
    'strength', 'long_shots', 'aggression', 'positioning', 'vision', 'penalties',
    'mentality', 'passing', 'speed', 'goalkeeper_diving', 'goalkeeper_handling',
    'goalkeeper_kicking', 'goalkeeper_positioning', 'goalkeeper_reflexes'
]

def get_similar_players(df_skills, player_name, features, n_clusters=4):
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics.pairwise import cosine_similarity

    if player_name not in df_skills['name'].values:
        raise ValueError(f"El jugador '{player_name}' no se encuentra en el dataset.")

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df_skills['players_cluster'] = kmeans.fit_predict(df_skills[features])

    scaler = StandardScaler()
    scaler.fit(df_skills[features])

    player_cluster = df_skills.loc[df_skills['name'] == player_name, 'players_cluster'].values[0]
    cluster_players = df_skills[df_skills['players_cluster'] == player_cluster]
    cluster_X = cluster_players[features]
    cluster_X_scaled = scaler.transform(cluster_X)

    similarity_matrix = cosine_similarity(cluster_X_scaled)
    similarity_df = pd.DataFrame(similarity_matrix, index=cluster_players['name'], columns=cluster_players['name'])
    similarity_to_player = similarity_df.loc[player_name]
    similar_players = similarity_to_player.sort_values(ascending=False).drop(player_name)

    similar_players = similar_players.to_frame().reset_index()
    similar_players.columns = ['name', 'Similarity']
    similar_players['Similarity'] = (similar_players['Similarity'] * 100).round(2)

    return similar_players

# ----------------------------------------
# Pestaña: "Analítica"
# ----------------------------------------
if selected_page == "Analítica":
    st.markdown("<div class='title-container'>Analítica de Jugadores</div>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    # Bloque analítico con tabs
    tab1, tab2 = st.tabs(["📈 Analítica", "📊 Resumen"])

    # 1) Pestaña "Resumen" → Power BI
    with tab2:
        st.markdown("<div class='title-container'>Resumen</div>", unsafe_allow_html=True)
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("""
            <iframe title="Power BI Dashboard" width="100%" height="750"
            src="https://app.powerbi.com/reportEmbed?reportId=b5000a80-2dbf-4155-9519-2334bfeece53&autoAuth=true&ctid=032115c7-35fe-4637-b2c3-d0a42906ba7b"
            frameborder="0" allowFullScreen="true"></iframe>
        """, unsafe_allow_html=True)

    # 2) Pestaña "Analítica" → Gráficas con Filtros
    with tab1:
        st.markdown("<div class='title-container'>Analítica</div>", unsafe_allow_html=True)
        st.markdown("<hr>", unsafe_allow_html=True)

        # Cargar dataset
        skills_file_path = r'skills_resultado.csv'
        df_skills_ana = pd.read_csv(skills_file_path)

        st.sidebar.title("Filtros Específicos (Analítica)")

        # Filtro de Categorías
        categorias_disponibles = [
            "Rendimiento y Habilidades",
            "Análisis Comparativo y Correlaciones",
            "Distribución de Características",
            "Valor y Salario de Jugadores"
        ]
        selected_categories = st.sidebar.multiselect(
            "Selecciona Categorías",
            ["Todas"] + categorias_disponibles,
            default=["Todas"]
        )
        if "Todas" in selected_categories:
            selected_categories = categorias_disponibles

        # Filtros dinámicos encadenados
        selected_league = st.sidebar.selectbox("Liga", ["Todos"] + sorted(df_skills_ana["league"].dropna().unique()))
        filtered_df_ana = df_skills_ana[df_skills_ana["league"] == selected_league] if selected_league != "Todos" else df_skills_ana

        selected_team = st.sidebar.selectbox("Equipo", ["Todos"] + sorted(filtered_df_ana["team"].dropna().unique()))
        filtered_df_ana = filtered_df_ana[filtered_df_ana["team"] == selected_team] if selected_team != "Todos" else filtered_df_ana

        selected_formation = st.sidebar.selectbox("Formación", ["Todos"] + sorted(filtered_df_ana["formation"].dropna().unique()))
        filtered_df_ana = filtered_df_ana[filtered_df_ana["formation"] == selected_formation] if selected_formation != "Todos" else filtered_df_ana

        positions = ["Todos"] + sorted(filtered_df_ana["position"].dropna().unique())
        selected_position = st.sidebar.multiselect("Posición", positions, default=["Todos"])
        if "Todos" not in selected_position:
            filtered_df_ana = filtered_df_ana[filtered_df_ana["position"].isin(selected_position)]

        nationalities = ["Todos"] + sorted(filtered_df_ana["nationality"].dropna().unique())
        selected_nationality = st.sidebar.multiselect("Nacionalidad", nationalities, default=["Todos"])
        if "Todos" not in selected_nationality:
            filtered_df_ana = filtered_df_ana[filtered_df_ana["nationality"].isin(selected_nationality)]

        players = ["Todos"] + sorted(filtered_df_ana["name"].dropna().unique())
        selected_player = st.sidebar.multiselect("Jugador", players, default=["Todos"])
        if "Todos" not in selected_player:
            filtered_df_ana = filtered_df_ana[filtered_df_ana["name"].isin(selected_player)]

        # Filtros de Rango Dinámicos
        def safe_slider(col, dfX, label, fixed_range=False):
            """Función para sliders."""
            if dfX.empty or col not in dfX.columns or dfX[col].dropna().empty:
                return None

            min_val, max_val = int(dfX[col].min()), int(dfX[col].max())
            if fixed_range:
                # Usar rango global
                min_val, max_val = int(df_skills_ana[col].min()), int(df_skills_ana[col].max())

            if min_val == max_val:
                st.sidebar.write(f"**{label}:** {min_val}")
                return (min_val, min_val)

            return st.sidebar.slider(label, min_val, max_val, (min_val, max_val))

        value_range = safe_slider("value_million_euro", df_skills_ana, "Valor de Mercado (Millones de €)", fixed_range=True)
        if value_range:
            filtered_df_ana = filtered_df_ana[filtered_df_ana["value_million_euro"].between(value_range[0], value_range[1] + 1)]

        age_range = safe_slider("age", filtered_df_ana, "Edad")
        if age_range:
            filtered_df_ana = filtered_df_ana[filtered_df_ana["age"].between(age_range[0], age_range[1])]

        height_range = safe_slider("height", filtered_df_ana, "Altura (cm)")
        if height_range:
            filtered_df_ana = filtered_df_ana[filtered_df_ana["height"].between(height_range[0], height_range[1])]

        wage_range = safe_slider("wage_million_euro", df_skills_ana, "Salario (Millones de €)", fixed_range=True)
        if wage_range:
            filtered_df_ana = filtered_df_ana[filtered_df_ana["wage_million_euro"].between(wage_range[0], wage_range[1] + 1)]

        # Definir categorías y gráficas
        plt.style.use('dark_background')

        categorias = {
            "Rendimiento y Habilidades": [
                ("Top 10 Jugadores con Mayor Overall",
                 lambda ax: sns.barplot(
                     x='overall', y='name',
                     data=filtered_df_ana.sort_values(by='overall', ascending=False).head(10),
                     palette='viridis', ax=ax
                 )),
                ("Top 10 Jugadores con Mayor Potencial",
                 lambda ax: sns.barplot(
                     x='potential', y='name',
                     data=filtered_df_ana.sort_values(by='potential', ascending=False).head(10),
                     palette='viridis', ax=ax
                 )),
                ("Comparación de Media de Finalización y Disparo por Edad",
                 lambda ax: sns.lineplot(
                     x=filtered_df_ana.groupby('age')['finishing'].mean().index,
                     y=filtered_df_ana.groupby('age')['finishing'].mean().values,
                     marker='o', color='blue', ax=ax
                 ) or sns.lineplot(
                     x=filtered_df_ana.groupby('age')['shooting_total'].mean().index,
                     y=filtered_df_ana.groupby('age')['shooting_total'].mean().values,
                     marker='o', label='Shooting', color='green', ax=ax
                 )),
                ("Media de Ritmo de Juego por Edad",
                 lambda ax: sns.lineplot(
                     x=filtered_df_ana.groupby('age')['pace_total'].mean().index,
                     y=filtered_df_ana.groupby('age')['pace_total'].mean().values,
                     marker='o', color='purple', ax=ax
                 )),
                ("Media de Visión de Juego por Edad",
                 lambda ax: sns.lineplot(
                     x=filtered_df_ana.groupby('age')['vision'].mean().index,
                     y=filtered_df_ana.groupby('age')['vision'].mean().values,
                     marker='o', color='brown', ax=ax
                 )),
                ("Media de Fuerza por Edad",
                 lambda ax: sns.lineplot(
                     x=filtered_df_ana.groupby('age')['strength'].mean().index,
                     y=filtered_df_ana.groupby('age')['strength'].mean().values,
                     marker='o', color='orange', ax=ax
                 )),
            ],

            "Análisis Comparativo y Correlaciones": [
                ("Mapa de Calor de Correlaciones (Habilidades Seleccionadas)",
                 lambda ax: (
                     sns.heatmap(
                         filtered_df_ana[[
                             'overall','finishing','heading_accuracy','ball_control','balance','shot_power',
                             'strength','long_shots','aggression','positioning','vision','penalties',
                             'mentality','speed','passing'
                         ]].corr(),
                         annot=True, fmt=".2f", cmap='RdBu', ax=ax, annot_kws={"size":3}
                     ),
                     ax.tick_params(axis='x', labelsize=5),
                     ax.tick_params(axis='y', labelsize=5),
                     ax.set_title("Mapa de Calor de Correlaciones (Habilidades Seleccionadas)", fontsize=5)
                 )),
                ("Relación entre Overall y Potential",
                 lambda ax: sns.scatterplot(
                     data=filtered_df_ana, x='overall', y='potential',
                     hue='age', palette='viridis', ax=ax
                 )),
                ("Tendencia de Puntuación Total según la Edad",
                 lambda ax: sns.regplot(
                     data=filtered_df_ana, x='age', y='overall',
                     line_kws={"color":"red"}, ax=ax
                 )),
                ("Tendencia de Potencial según la Edad",
                 lambda ax: sns.regplot(
                     data=filtered_df_ana, x='age', y='potential',
                     line_kws={"color":"red"}, ax=ax
                 )),
                ("Media del Score Defensivo y Ofensivo por Edad",
                 lambda ax: sns.lineplot(
                     x=filtered_df_ana.groupby('age')['defending_total'].mean().index,
                     y=filtered_df_ana.groupby('age')['defending_total'].mean().values,
                     marker='o', color='blue', ax=ax
                 ) or sns.lineplot(
                     x=filtered_df_ana.groupby('age')['shooting_total'].mean().index,
                     y=filtered_df_ana.groupby('age')['shooting_total'].mean().values,
                     marker='o', label='Offensive Score', color='orange', ax=ax
                 ))
            ],

            "Distribución de Características": [
                ("Distribución de Edad",
                 lambda ax: sns.countplot(x='age', data=filtered_df_ana, width=0.3, ax=ax) or
                 [ax.bar_label(b, color='white') for b in ax.containers] or
                 ax.tick_params(axis='x', labelsize=1)),
                ("Distribución por Altura (cm)",
                 lambda ax: sns.countplot(x='height', data=filtered_df_ana, width=0.3, ax=ax, color='orange') or
                 ax.tick_params(axis='x', labelsize=6) or
                 [ax.bar_label(b, color='white') for b in ax.containers]),
                ("Distribución de la Puntuación General",
                 lambda ax: sns.histplot(filtered_df_ana['overall'], kde=True, bins=25, color='blue', ax=ax)),
                ("Distribución de la Puntuación de Potencial",
                 lambda ax: sns.histplot(filtered_df_ana['potential'], kde=True, bins=25, color='brown', ax=ax)),
                ("Distribución de Posiciones",
                 lambda ax: sns.countplot(
                     data=filtered_df_ana, x='position',
                     order=filtered_df_ana['position'].value_counts().index,
                     palette='magma', ax=ax
                 ) or ax.set_xticklabels(ax.get_xticklabels(), rotation=30)),
            ],

            "Valor y Salario de Jugadores": [
                ("Jugadores con Mayor Valor de Mercado (Millones de Euros)",
                 lambda ax: ax.barh(
                     filtered_df_ana.sort_values('value_million_euro', ascending=False).head(10)['name'][::-1],
                     filtered_df_ana.sort_values('value_million_euro', ascending=False).head(10)['value_million_euro'][::-1],
                     color='purple'
                 ) or ax.set(xlabel="Valor en Millones de Euros", ylabel="Jugador") or
                 ax.set_title("Jugadores con mayor valor de mercado") or ax.grid(axis='x', linestyle='--', alpha=0.7)),
                ("Jugadores con Mayor Salario (Millones de Euros)",
                 lambda ax: ax.barh(
                     filtered_df_ana.sort_values('wage_million_euro', ascending=False).head(10)['name'][::-1],
                     filtered_df_ana.sort_values('wage_million_euro', ascending=False).head(10)['wage_million_euro'][::-1],
                     color='darkorange'
                 ) or ax.set(xlabel="Salario en Millones de Euros", ylabel="Jugador") or
                 ax.set_title("Jugadores con mayor salario") or ax.grid(axis='x', linestyle='--', alpha=0.7)),
                ("Media de Valor de Mercado por Edad",
                 lambda ax: sns.lineplot(
                     x=filtered_df_ana.groupby('age')['value_million_euro'].mean().index,
                     y=filtered_df_ana.groupby('age')['value_million_euro'].mean().values,
                     marker='o', color='grey', ax=ax
                 )),
                ("Media de Salario por Edad",
                 lambda ax: sns.lineplot(
                     x=filtered_df_ana.groupby('age')['wage_million_euro'].mean().index,
                     y=filtered_df_ana.groupby('age')['wage_million_euro'].mean().values,
                     marker='o', color='blue', ax=ax
                 )),
            ],
        }

        # Mostrar los gráficos según la categoría
        for cat_name, graficos in categorias.items():
            if cat_name in selected_categories:
                st.markdown(f"### 🔹 {cat_name}")
                cols_cats = st.columns(2)

                for idx, (titulo, plot_func) in enumerate(graficos):
                    with cols_cats[idx % 2]:
                        st.markdown(f"#### {titulo}")

                        if filtered_df_ana.empty:
                            st.warning("No hay valores disponibles para el rango seleccionado.")
                        elif len(filtered_df_ana) > 1:
                            fig, ax = plt.subplots(figsize=(4, 3))
                            ax.set_facecolor('black')
                            ax.tick_params(colors='white', labelsize=8)
                            ax.title.set_color('white')

                            try:
                                plot_func(ax)
                                st.pyplot(fig)
                            except Exception as e:
                                st.error(f"No se pudo generar el gráfico: {str(e)}")

# ----------------------------------------
# Pestaña: "Jugadores"
# ----------------------------------------
elif selected_page == "Jugadores":
    st.markdown("<div class='title-container'>Analítica Jugadores</div>", unsafe_allow_html=True) # Title from the new code
    st.markdown("<hr>", unsafe_allow_html=True)

    # Bloque analítico con tabs
    tab1, tab2 = st.tabs(["📈 Analítica", "📊 Resumen"]) # Tabs from the new code

    # 1) Pestaña "Resumen" → Power BI
    with tab2:
        st.markdown("<div class='title-container'>Resumen</div>", unsafe_allow_html=True) # Title from the new code
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("""
            <iframe title="Power BI Dashboard" width="100%" height="750"
            src="https://app.powerbi.com/reportEmbed?reportId=67d2ad73-e415-49cf-9e2f-f278a646ded5&autoAuth=true&ctid=032115c7-35fe-4637-b2c3-d0a42906ba7b"
            frameborder="0" allowFullScreen="true"></iframe>
        """, unsafe_allow_html=True)

    # 2) Pestaña "Analítica" → Gráficas con Filtros
    with tab1:
        st.markdown("<div class='title-container'>Analítica Jugadores</div>", unsafe_allow_html=True) # Title from the new code
        st.markdown("<hr>", unsafe_allow_html=True)

        # Cargar dataset for Jugadores tab - using the path from the new code
        file_path = r'players_cleaned.csv'
        df_players = pd.read_csv(file_path)
        filtered_df_ana_jugadores = df_players.copy() # Rename to avoid conflict with "Analítica" tab

        st.sidebar.title("Filtros (Jugadores)") # Modified sidebar title for clarity

        categorias_disponibles_jugadores = ["Distribución de Jugadores", "Análisis y Correlaciones", "Goles y Asistencias", "Disciplina: Tarjetas y Faltas","Minutos y Partidos Jugados"] # Categories from the new code
        selected_categories_jugadores = st.sidebar.multiselect("Selecciona Categorías", ["Todas"] + categorias_disponibles_jugadores, default=["Todas"], key="categorias_jugadores") # Key added to avoid conflict
        if "Todas" in selected_categories_jugadores:
            selected_categories_jugadores = categorias_disponibles_jugadores

        # Filtros dinámicos - adjusted for df_players and key added
        for col in ["league", "team", "formation", "position", "player"]:
            values = ["Todos"] + sorted(filtered_df_ana_jugadores[col].dropna().unique())
            selected_value = st.sidebar.multiselect(col.capitalize(), values, default=["Todos"], key=f"filter_{col}_jugadores") # Keys added to avoid conflict
            if "Todos" not in selected_value:
                filtered_df_ana_jugadores = filtered_df_ana_jugadores[filtered_df_ana_jugadores[col].isin(selected_value)]

        # Estilo de gráficos - already set globally
        plt.style.use('dark_background')

        categorias_jugadores = { 
            "Minutos y Partidos Jugados": [
                ("Histograma de Partidos Jugados", lambda ax: sns.histplot(filtered_df_ana_jugadores['matches'], kde=True, color='blue', ax=ax) or ax.set(title="Histograma de Partidos Jugados", xlabel="Número de Partidos", ylabel="Número de Jugadores")),
                ("Promedio de Minutos Jugados por Edad", lambda ax: sns.lineplot(x=filtered_df_ana_jugadores.groupby('age')['minutes'].mean().index, y=filtered_df_ana_jugadores.groupby('age')['minutes'].mean().values, marker='o', color='darkorange', ax=ax) or ax.set(title="Promedio de Minutos Jugados por Edad", xlabel="Edad", ylabel="Minutos Jugados")),
            ],

            "Distribución de Jugadores": [
                ("Distribución de Posiciones", lambda ax: sns.countplot(data=filtered_df_ana_jugadores, x='position', order=filtered_df_ana_jugadores['position'].value_counts().index, palette='Reds', ax=ax) or ax.set(title="Distribución de Posiciones", xlabel="Posición", ylabel="Frecuencia") or ax.set_xticklabels(ax.get_xticklabels(), rotation=45)),
                ("Distribución de Formaciones", lambda ax: sns.countplot(data=filtered_df_ana_jugadores, x='formation', order=filtered_df_ana_jugadores['formation'].value_counts().index, palette='PuBuGn', ax=ax) or ax.set(title="Distribución de Formaciones", xlabel="Formación", ylabel="Frecuencia") or ax.set_xticklabels(ax.get_xticklabels(), rotation=45)),
                ("Top 15 Número de Jugadores por País", lambda ax: sns.barplot(y=filtered_df_ana_jugadores['nation'].value_counts().head(15).index, x=filtered_df_ana_jugadores['nation'].value_counts().head(15).values, color='blue', ax=ax) or ax.set(title="Número de Jugadores por País", xlabel="Número de Jugadores", ylabel="País") or ax.set_yticklabels(ax.get_yticklabels(), fontsize=6)),
            ],

            "Goles y Asistencias": [
                ("Histograma de Goles por Posición", lambda ax: sns.histplot(data=filtered_df_ana_jugadores, x='goals', hue='position', bins=20, alpha=0.7, ax=ax) or ax.set(title="Histograma de Goles por Posición", xlabel="Número de Goles", ylabel="Frecuencia")),
                ("Histograma de Asistencias por Posición", lambda ax: sns.histplot(data=filtered_df_ana_jugadores, x='assists', hue='position', bins=20, alpha=0.7, ax=ax) or ax.set(title="Histograma de Asistencias por Posición", xlabel="Número de Asistencias", ylabel="Frecuencia")),
                ("Distribución de Asistencias", lambda ax: sns.histplot(filtered_df_ana_jugadores['assists'], kde=True, color='red', bins=20, ax=ax) or ax.set(title="Distribución de Asistencias", xlabel="Número de Asistencias", ylabel="Número de Jugadores")),
                ("Asistencias por Posición", lambda ax: sns.barplot(x=filtered_df_ana_jugadores.groupby('position')['assists'].mean().index, y=filtered_df_ana_jugadores.groupby('position')['assists'].mean().values, color='orange', ax=ax) or ax.set(title="Asistencias por Posición", xlabel="Posición", ylabel="Asistencias")),
                ("Goles por Posición", lambda ax: sns.barplot(x=filtered_df_ana_jugadores.groupby('position')['goals'].mean().index, y=filtered_df_ana_jugadores.groupby('position')['goals'].mean().values, color='blue', ax=ax) or ax.set(title="Goles por Posición", xlabel="Posición", ylabel="Goles")),
            ],

            "Disciplina: Tarjetas y Faltas": [
                ("Histograma de Tarjetas Amarillas por Posición", lambda ax: sns.histplot(data=filtered_df_ana_jugadores, x='yellow_cards', hue='position', bins=20, alpha=0.7, ax=ax) or ax.set(title="Histograma de Tarjetas Amarillas", xlabel="Número de Tarjetas", ylabel="Frecuencia")),
                ("Histograma de Tarjetas Rojas por Posición", lambda ax: sns.histplot(data=filtered_df_ana_jugadores, x='red_cards', hue='position', bins=20, alpha=0.7, ax=ax) or ax.set(title="Histograma de Tarjetas Rojas", xlabel="Número de Tarjetas", ylabel="Frecuencia")),
                ("Tarjetas Rojas por Posición", lambda ax: sns.barplot(x=filtered_df_ana_jugadores.groupby('position')['red_cards'].mean().index, y=filtered_df_ana_jugadores.groupby('position')['red_cards'].mean().values, color='red', ax=ax) or ax.set(title="Tarjetas Rojas por Posición", xlabel="Posición", ylabel="Tarjetas Rojas")),
                ("Tarjetas Amarillas por Posición", lambda ax: sns.barplot(x=filtered_df_ana_jugadores.groupby('position')['yellow_cards'].mean().index, y=filtered_df_ana_jugadores.groupby('position')['yellow_cards'].mean().values, color='gold', ax=ax) or ax.set(title="Tarjetas Amarillas por Posición", xlabel="Posición", ylabel="Tarjetas Amarillas")),
            ],

            "Análisis y Correlaciones": [
                ("Mapa de Calor de Correlaciones Generales", lambda ax: sns.heatmap(filtered_df_ana_jugadores.corr(numeric_only=True), annot=True, fmt='.1f', cmap='RdBu', annot_kws={"size": 4}, ax=ax) or ax.set(title="Mapa de Calor de Correlaciones Generales")),
                ("Mapa de Calor de Expected Goals y Expected Assists", lambda ax: sns.heatmap(filtered_df_ana_jugadores[['goals', 'assists', 'expected_goals', 'expected_assists']].corr(), annot=True, fmt='.1f', cmap='RdBu', annot_kws={"size": 6}, ax=ax) or ax.set(title="Mapa de Calor de xG y xA")),
                ("Scatterplot: Goles vs Expected Goals (xG)", lambda ax: sns.scatterplot(x=filtered_df_ana_jugadores['expected_goals'], y=filtered_df_ana_jugadores['goals'], hue=filtered_df_ana_jugadores['goals'] - filtered_df_ana_jugadores['expected_goals'], palette='coolwarm', s=30, ax=ax) or ax.set(title="Goles vs Expected Goals (xG)", xlabel="Goles Esperados (xG)", ylabel="Goles")),
                ("Scatterplot: Asistencias vs Expected Asistencias (xA)", lambda ax: sns.scatterplot(x=filtered_df_ana_jugadores['expected_assists'], y=filtered_df_ana_jugadores['assists'], hue=filtered_df_ana_jugadores['assists'] - filtered_df_ana_jugadores['expected_assists'], palette='coolwarm', s=30, ax=ax) or ax.set(title="Asistencias vs Expected Asistencias (xA)", xlabel="Asistencias Esperadas (xA)", ylabel="Asistencias")),
            ],
        }

        # Mostrar gráficos
        for categoria, graficos in categorias_jugadores.items(): 
            if categoria in selected_categories_jugadores: 
                st.markdown(f"###  {categoria}")
                cols_cats = st.columns(2)
                for idx, (titulo, funcion) in enumerate(graficos):
                    with cols_cats[idx % 2]:
                        st.markdown(f"#### {titulo}")

                        if filtered_df_ana_jugadores.empty: 
                            st.warning("No hay valores disponibles para el rango seleccionado.")
                        else:
                            fig, ax = plt.subplots(figsize=(4, 3))
                            ax.set_facecolor('black')
                            ax.tick_params(colors='white', labelsize=8)
                            ax.title.set_color('white')
                            try:
                                funcion(ax)
                                st.pyplot(fig)
                            except Exception as e:
                                st.error(f"No se pudo generar el gráfico: {str(e)}")


# ----------------------------------------
# Pestaña: "Recomendador"
# ----------------------------------------
elif selected_page == "Recomendador":
    st.title("Recomendador de Jugadores")
    st.markdown("""
        <style>
            .main, .stApp {
                background-color: black;
                color: white;
            }
            .stButton>button {
                background-color: orange;
                color: black;
                font-size: 24px;
                padding: 12px 24px;
                border-radius: 10px;
            }
            .stSelectbox div[data-baseweb="select"] {
                background-color: white !important;
                color: black !important;
            }
            .stTable, table {
                color: white !important;
                background-color: transparent !important;
            }
            .sidebar .block-container {
                background-color: #222222;
                padding: 20px;
                border-radius: 10px;
            }
            hr {
                border: 2px solid white;
                border-image: linear-gradient(to right, white, yellow, orange) 1;
            }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    st.title("Búsqueda de Jugador")
    player_name = st.selectbox("Seleccione un jugador", [""] + list(df_skills['name'].unique()), index=0)

    if player_name:
        similar_players = get_similar_players(df_skills, player_name, player_skills)

        # Aplicar filtros a los resultados
        filtered_similar_players = similar_players.copy()
        filtered_similar_players = filtered_similar_players[
            (df_skills['value_million_euro'].between(price_range[0], price_range[1])) &
            (df_skills['wage_million_euro'].between(wage_range[0], wage_range[1])) &
            (df_skills['age'].between(age_range[0], age_range[1])) &
            (df_skills['height'].between(height_range[0], height_range[1]))
        ]

        if "Todos" not in position_options:
            filtered_similar_players = filtered_similar_players[df_skills['position'].isin(position_options)]
        if "Todos" not in nationality_options:
            filtered_similar_players = filtered_similar_players[df_skills['nationality'].isin(nationality_options)]
        if preferred_foot == "Izquierda":
            filtered_similar_players = filtered_similar_players[df_skills['preferred_foot'] == 0]
        elif preferred_foot == "Derecha":
            filtered_similar_players = filtered_similar_players[df_skills['preferred_foot'] == 1]

        if filtered_similar_players.empty:
            st.warning("No hay jugadores recomendados para el rango seleccionado.")
        else:
            top_3 = filtered_similar_players.head(min(3, len(filtered_similar_players)))
            cols = st.columns(len(top_3))
            medal_icons = ["🥇", "🥈", "🥉"]
            for i, row in enumerate(top_3.itertuples()):
                with cols[i]:
                    st.markdown(
                        f"<div style='text-align: center; font-size: 20px; border-top: 4px solid orange;"
                        f"border-bottom: 4px solid orange;'>"
                        f"<h3>Top {i+1} {medal_icons[i]}</h3><h4>{row.name}</h4>"
                        f"<p style='font-size: 22px;'>{row.Similarity}%</p></div>",
                        unsafe_allow_html=True
                    )

        st.write("### Tabla de Recomendaciones y Métricas Adicionales")
        cols = st.columns([1, 1.5])
        with cols[0]:
            st.dataframe(filtered_similar_players.style.format({'Similarity': '{:.2f}%'}))
        with cols[1]:
            df_metrics = df_skills[df_skills['name'].isin(filtered_similar_players['name'])][[
                'name', 'overall', 'potential', 'pace_total',
                'shooting_total', 'passing_total', 'dribbling_total',
                'defending_total', 'physicality_total'
            ]]
            df_metrics = df_metrics.set_index('name').loc[filtered_similar_players['name']].reset_index()
            st.dataframe(df_metrics)

# ----------------------------------------
# Pestaña: "Comparador"
# ----------------------------------------
elif selected_page == "Comparador":
    sub_options = ["Comparador de Jugadores", "Detalles"]
    sub_page = st.radio("", sub_options, horizontal=True, key="comparador_page")

    if sub_page == "Comparador de Jugadores":
        st.markdown("<div class='title-container'>Comparador de Jugadores</div>", unsafe_allow_html=True)

        col1, col2 = st.columns([1.2, 1])
        with col1:
            jugador1_input = st.text_input("Primer jugador")
            jugador2_input = st.text_input("Segundo jugador")

            if jugador1_input and jugador2_input:
                jugador1 = normalizar_nombre_groq(jugador1_input)
                jugador2 = normalizar_nombre_groq(jugador2_input)
                jugador1_match = process.extractOne(jugador1, jugadores)[0]
                jugador2_match = process.extractOne(jugador2, jugadores)[0]

                if jugador1_match and jugador2_match:
                    st.write(f"✅ **Jugadores seleccionados:** {jugador1_match} vs {jugador2_match}")
                    df_selected = df[df["name"].isin([jugador1_match, jugador2_match])]

                    attributes = ["pace_total", "shooting_total", "passing_total",
                                  "dribbling_total", "defending_total", "physicality_total"]
                    df_metrics = df_selected.set_index("name")[attributes].T
                    st.subheader("Comparación de Atributos")
                    st.dataframe(df_metrics.style.format("{:.2f}"))

        with col2:
            if jugador1_input and jugador2_input and jugador1_match and jugador2_match:
                attributes_labels = ["Velocidad", "Tiro", "Pase", "Regate", "Defensa", "Físico"]
                values1 = df_selected[df_selected["name"] == jugador1_match][attributes].values.flatten()
                values2 = df_selected[df_selected["name"] == jugador2_match][attributes].values.flatten()

                fig, ax = plt.subplots(figsize=(1.5, 1.5), subplot_kw={"projection": "polar"})
                fig.patch.set_alpha(0)
                ax.set_facecolor("black")
                angles = np.linspace(0, 2 * np.pi, len(attributes), endpoint=False).tolist()
                values1 = np.concatenate((values1, [values1[0]]))
                values2 = np.concatenate((values2, [values2[0]]))
                angles += angles[:1]

                ax.plot(angles, values1, color="deepskyblue", linewidth=1, linestyle="-", label=jugador1_match)
                ax.plot(angles, values2, color="darkorange", linewidth=1, linestyle="-", label=jugador2_match)
                ax.set_xticks(angles[:-1])
                ax.set_xticklabels(attributes_labels, fontsize=3, color="white", fontweight="bold")
                ax.set_yticklabels([])
                ax.spines["polar"].set_color("white")
                ax.grid(color="gray", linestyle="--", linewidth=0.2)

                legend = ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), frameon=False, fontsize=4)
                for text in legend.get_texts():
                    text.set_color("white")

                st.pyplot(fig)

        # Justificación de similitudes
        if jugador1_input and jugador2_input and jugador1_match and jugador2_match:
            justificacion = justificar_similitudes(jugador1_match, jugador2_match)
            with st.container():
                st.subheader("📊 Justificación de similitudes:", divider="gray")
                st.markdown(f"<div class='full-width'>{justificacion}</div>", unsafe_allow_html=True)

    elif sub_page == "Detalles":
        st.markdown("<div class='title-container'>Detalles</div>", unsafe_allow_html=True)
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("""
            <iframe title="Power BI Dashboard" width="100%" height="700"
            src="https://app.powerbi.com/reportEmbed?reportId=dd1a633c-cef1-4229-b257-ffb9b9137049&autoAuth=true&ctid=032115c7-35fe-4637-b2c3-d0a42906ba7b"
            frameborder="0" allowFullScreen="true"></iframe>
        """, unsafe_allow_html=True)
