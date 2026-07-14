"""
SpacedL2 — Aplicación Streamlit
Proyecto Final de Maestría · UFHEC · 2025
Autora: Dominga Rodríguez
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from math import exp

# ─── Configuración de página ───────────────────────────────────────────────
st.set_page_config(
    page_title="SpacedL2 — ML para Retención de Vocabulario",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CSS personalizado ─────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;700;800;900&display=swap');
    html, body, [class*="css"] { font-family: 'Outfit', sans-serif; }
    .stApp { background-color: #06050f; }
    h1, h2, h3 { color: #f0f0ff !important; }
    .metric-box {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 20px;
        text-align: center;
    }
    .model-card {
        background: rgba(255,255,255,0.03);
        border-radius: 16px;
        padding: 20px;
        margin: 8px 0;
        border-left: 4px solid;
    }
    .hero-banner {
        background: linear-gradient(135deg, rgba(124,58,237,0.15), rgba(6,182,212,0.1));
        border: 1px solid rgba(124,58,237,0.25);
        border-radius: 20px;
        padding: 32px;
        margin-bottom: 32px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ─── Motor ML (Regresión Logística) ────────────────────────────────────────
def predict_forget_risk(hours, reviews, accuracy, difficulty, latency, mcer):
    diff_map = {'Baja': 1, 'Media': 2, 'Alta': 3}
    mcer_map = {'A1': 1, 'A2': 2, 'B1': 3, 'B2': 4, 'C1': 5, 'C2': 6}
    diff_num = diff_map.get(difficulty, 2)
    mcer_num = mcer_map.get(mcer, 3)
    logit = (0.022 * hours) - (0.35 * reviews) - (3.5 * accuracy) + (0.9 * diff_num) + (0.0004 * latency) - (0.5 * mcer_num) + 0.8
    prob = 1 / (1 + exp(-logit))
    needs_review = 1 if prob > 0.5 else 0
    if needs_review == 0:
        interval = round(min(max(2.5 * reviews * (1.5 - prob), 2), 90))
    else:
        interval = 1
    return prob, needs_review, interval

# ─── Dataset de ejemplo ─────────────────────────────────────────────────────
vocabulary = [
    {"word": "reluctant", "category": "Adjetivo", "difficulty": "Media", "reviews": 3, "accuracy": 0.85, "elapsedHours": 72, "latency": 1800},
    {"word": "unprecedented", "category": "Adjetivo", "difficulty": "Alta", "reviews": 1, "accuracy": 0.50, "elapsedHours": 120, "latency": 4500},
    {"word": "mitigate", "category": "Verbo", "difficulty": "Media", "reviews": 4, "accuracy": 0.90, "elapsedHours": 48, "latency": 1200},
    {"word": "compelling", "category": "Adjetivo", "difficulty": "Media", "reviews": 2, "accuracy": 0.80, "elapsedHours": 96, "latency": 2200},
    {"word": "cognitive", "category": "Adjetivo", "difficulty": "Media", "reviews": 5, "accuracy": 0.95, "elapsedHours": 168, "latency": 900},
    {"word": "phrasal verb", "category": "Sustantivo", "difficulty": "Alta", "reviews": 2, "accuracy": 0.60, "elapsedHours": 24, "latency": 3800},
    {"word": "retention", "category": "Sustantivo", "difficulty": "Baja", "reviews": 6, "accuracy": 0.98, "elapsedHours": 240, "latency": 700},
    {"word": "decay", "category": "Verbo", "difficulty": "Baja", "reviews": 3, "accuracy": 0.75, "elapsedHours": 144, "latency": 1600},
    {"word": "retrieve", "category": "Verbo", "difficulty": "Media", "reviews": 4, "accuracy": 0.85, "elapsedHours": 12, "latency": 1100},
    {"word": "spaced repetition", "category": "Sustantivo", "difficulty": "Alta", "reviews": 3, "accuracy": 0.70, "elapsedHours": 180, "latency": 3200},
]

MODEL_RESULTS = {
    "🌲 Random Forest":       {"acc": 0.912, "precision": 0.908, "recall": 0.915, "f1": 0.911, "color": "#10b981"},
    "⚡ SVM (RBF)":            {"acc": 0.884, "precision": 0.879, "recall": 0.881, "f1": 0.880, "color": "#f59e0b"},
    "📈 Regresión Logística":  {"acc": 0.850, "precision": 0.846, "recall": 0.852, "f1": 0.849, "color": "#7c3aed"},
    "🌳 Árbol de Decisión":    {"acc": 0.831, "precision": 0.827, "recall": 0.834, "f1": 0.830, "color": "#06b6d4"},
}

# ─── SIDEBAR ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 SpacedL2")
    st.markdown("**Proyecto Final · UFHEC · 2025**")
    st.markdown("👩‍💻 **Dominga Rodríguez**")
    st.markdown("*Maestría en Inteligencia Artificial*")
    st.divider()
    page = st.radio("Navegar a:", [
        "🏠 Inicio",
        "🎛️ Simulador ML",
        "📊 Dashboard de Vocabulario",
        "🤖 Comparación de Modelos",
        "📐 Curva del Olvido"
    ])
    st.divider()
    st.markdown("**Links del Proyecto:**")
    st.markdown("🐙 [GitHub](https://github.com/domingamaestriaufhec/Proyecto-FinalML)")
    st.markdown("🌐 [Simulador Web](https://domingamaestriaufhec.github.io/Proyecto-FinalML/)")
    st.markdown("⚡ [Google Colab](https://colab.research.google.com/github/domingamaestriaufhec/Proyecto-FinalML/blob/main/notebook.ipynb)")

# ═══════════════════════════════════════════════════════════════════════════
#  PÁGINA 1 — INICIO
# ═══════════════════════════════════════════════════════════════════════════
if page == "🏠 Inicio":
    st.markdown("""
    <div class="hero-banner">
        <h1 style="font-size:3em; margin:0; color:#f0f0ff">🧠 SpacedL2</h1>
        <p style="color:#94a3b8; font-size:1.2em; margin-top:8px">Machine Learning para la Retención de Vocabulario en L2</p>
        <p style="color:#6b7280; font-size:0.9em; margin-top:4px">Proyecto Final · Maestría en IA · UFHEC 2025 · Dominga Rodríguez</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🤖 Modelos ML", "4", "supervisados")
    with col2:
        st.metric("🎯 Mejor Exactitud", "91.2%", "Random Forest")
    with col3:
        st.metric("📚 Vocabulario", "10 palabras", "5 features cada una")
    with col4:
        st.metric("📊 Dataset", "500 registros", "sintético balanceado")

    st.divider()
    st.subheader("¿De qué trata este proyecto?")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
        **Problema:** El cerebro olvida el vocabulario exponencialmente sin repaso estratégico.
        La Curva del Olvido de Ebbinghaus (1885) muestra que perdemos el 70% de lo aprendido
        en las primeras 24 horas sin refuerzo.

        **Solución:** Usar modelos de **clasificación supervisada** para predecir el riesgo
        de olvido de cada palabra y calcular el intervalo óptimo de repaso, personalizando
        el estudio para cada estudiante.
        """)
    with col_b:
        st.markdown("""
        **4 Modelos entrenados:**
        - 📈 Regresión Logística (baseline)
        - 🌳 Árbol de Decisión
        - 🌲 Random Forest ⭐ (mejor)
        - ⚡ SVM con kernel RBF

        **5 Variables predictoras (features):**
        - ⏱️ Horas sin repasar · ⚡ Latencia (ms) · 🔁 Repasos previos
        - 🎯 Tasa de aciertos · 📚 Dificultad + Nivel MCER
        """)

# ═══════════════════════════════════════════════════════════════════════════
#  PÁGINA 2 — SIMULADOR ML
# ═══════════════════════════════════════════════════════════════════════════
elif page == "🎛️ Simulador ML":
    st.title("🎛️ Simulador de Regresión Logística")
    st.markdown("Ajusta los parámetros y observa la predicción del modelo en tiempo real.")

    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.subheader("Variables Predictoras (Features)")
        hours = st.slider("⏱️ Horas sin repasar", 1, 720, 72, help="Tiempo transcurrido desde el último repaso")
        latency = st.slider("⚡ Latencia de respuesta (ms)", 400, 10000, 2500, step=100, help="Milisegundos para recordar la palabra")
        reviews = st.slider("🔁 Cantidad de repasos previos", 1, 15, 3, help="Número de sesiones de repaso anteriores")
        accuracy = st.slider("🎯 Tasa histórica de aciertos (%)", 0, 100, 85) / 100
        col_d, col_m = st.columns(2)
        with col_d:
            difficulty = st.selectbox("📚 Dificultad", ["Baja", "Media", "Alta"])
        with col_m:
            mcer = st.selectbox("🎓 Nivel MCER", ["A1", "A2", "B1", "B2", "C1", "C2"], index=2)

    with col2:
        prob, needs_review, interval = predict_forget_risk(hours, reviews, accuracy, difficulty, latency, mcer)
        percent = round(prob * 100, 1)

        st.subheader("Predicción del Modelo")

        # Circle gauge using matplotlib
        fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(aspect="equal"))
        fig.patch.set_alpha(0)
        ax.set_facecolor('none')
        color = "#ef4444" if prob > 0.5 else "#10b981"
        ax.pie([prob, 1-prob],
               startangle=90, counterclock=False,
               colors=[color, "#1e1e2d"],
               wedgeprops=dict(width=0.35, edgecolor='#06050f', linewidth=3))
        ax.text(0, 0, f"{percent}%", ha='center', va='center',
                fontsize=30, fontweight='bold', color='white',
                fontfamily='Outfit')
        ax.text(0, -0.55, "Riesgo de Olvido", ha='center', va='center',
                fontsize=9, color='#64748b')
        st.pyplot(fig)
        plt.close()

        if needs_review:
            st.error(f"⚠️ **¡Riesgo Alto!** — Repasar hoy\nProbabilidad de olvido: **{percent}%**")
        else:
            st.success(f"✅ **Retención Estable** — Próximo repaso en **{interval} días**\nProbabilidad de olvido: **{percent}%**")

        st.info(f"📅 Intervalo de repetición espaciada: **{interval} {'día' if interval == 1 else 'días'}**")

        # Feature contribution chart
        st.subheader("Contribución de cada Feature")
        diff_map = {'Baja': 1, 'Media': 2, 'Alta': 3}
        mcer_map = {'A1': 1, 'A2': 2, 'B1': 3, 'B2': 4, 'C1': 5, 'C2': 6}
        contributions = {
            "Horas sin repasar": 0.022 * hours,
            "Latencia (ms)": 0.0004 * latency,
            "Dificultad": 0.9 * diff_map.get(difficulty, 2),
            "Repasos (-)" : -0.35 * reviews,
            "Aciertos (-)": -3.5 * accuracy,
            "Nivel MCER (-)": -0.5 * mcer_map.get(mcer, 3),
        }
        contrib_df = pd.DataFrame(list(contributions.items()), columns=["Feature", "Contribución al Logit"])
        contrib_df = contrib_df.sort_values("Contribución al Logit")
        colors_bar = ["#ef4444" if v > 0 else "#10b981" for v in contrib_df["Contribución al Logit"]]
        fig2, ax2 = plt.subplots(figsize=(6, 3))
        fig2.patch.set_facecolor('#06050f')
        ax2.set_facecolor('#06050f')
        ax2.barh(contrib_df["Feature"], contrib_df["Contribución al Logit"], color=colors_bar, edgecolor='none')
        ax2.tick_params(colors='white', labelsize=9)
        ax2.set_xlabel("Contribución al Logit", color='#64748b', fontsize=9)
        ax2.spines['bottom'].set_color('#1e293b')
        ax2.spines['left'].set_color('#1e293b')
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.axvline(0, color='#334155', linewidth=1)
        ax2.set_title("Rojo = aumenta riesgo · Verde = reduce riesgo", color='#64748b', fontsize=8, pad=8)
        st.pyplot(fig2)
        plt.close()

# ═══════════════════════════════════════════════════════════════════════════
#  PÁGINA 3 — DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════
elif page == "📊 Dashboard de Vocabulario":
    st.title("📊 Dashboard de Retención de Vocabulario")

    mcer_user = st.selectbox("🎓 Tu nivel MCER:", ["A1", "A2", "B1", "B2", "C1", "C2"], index=2)

    rows = []
    for w in vocabulary:
        prob, needs, interval = predict_forget_risk(
            w["elapsedHours"], w["reviews"], w["accuracy"],
            w["difficulty"], w["latency"], mcer_user
        )
        days = round(w["elapsedHours"] / 24)
        rows.append({
            "Palabra": w["word"],
            "Categoría": w["category"],
            "Dificultad": w["difficulty"],
            "Repasos": w["reviews"],
            "Último repaso": f"{days}d ago",
            "Riesgo de Olvido": f"{round(prob*100)}%",
            "Estado": "⚠️ Repasar HOY" if needs else f"✅ En {interval}d",
            "_risk": prob
        })

    df = pd.DataFrame(rows)
    total = len(df)
    retained = len(df[df["_risk"] <= 0.5])
    at_risk = total - retained

    col1, col2, col3 = st.columns(3)
    with col1: st.metric("📚 Total palabras", total)
    with col2: st.metric("✅ Retención segura", retained, f"{round(retained/total*100)}%")
    with col3: st.metric("⚠️ En riesgo", at_risk, f"-{round(at_risk/total*100)}%", delta_color="inverse")

    st.divider()
    st.dataframe(
        df.drop(columns=["_risk"]).style.applymap(
            lambda v: "color: #f87171" if "⚠️" in str(v) else ("color: #34d399" if "✅" in str(v) else ""),
            subset=["Estado"]
        ),
        use_container_width=True
    )

    # Risk bar chart
    st.subheader("📊 Riesgo de Olvido por Palabra")
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor('#06050f')
    ax.set_facecolor('#06050f')
    risks = [round(r["_risk"]*100) for r in rows]
    words = [r["Palabra"] for r in rows]
    colors = ["#ef4444" if r > 50 else "#10b981" for r in risks]
    bars = ax.bar(words, risks, color=colors, edgecolor='none', width=0.6)
    ax.axhline(50, color="#f59e0b", linewidth=1.5, linestyle="--", alpha=0.7, label="Umbral de riesgo (50%)")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Riesgo de Olvido (%)", color='#94a3b8', fontsize=10)
    ax.tick_params(colors='white', labelsize=9, axis='both')
    plt.xticks(rotation=25, ha='right')
    ax.spines['bottom'].set_color('#1e293b')
    ax.spines['left'].set_color('#1e293b')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(facecolor='#06050f', labelcolor='white', fontsize=9)
    for bar, risk in zip(bars, risks):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5, f"{risk}%",
                ha='center', va='bottom', fontsize=8, color='white')
    st.pyplot(fig)
    plt.close()

# ═══════════════════════════════════════════════════════════════════════════
#  PÁGINA 4 — COMPARACIÓN DE MODELOS
# ═══════════════════════════════════════════════════════════════════════════
elif page == "🤖 Comparación de Modelos":
    st.title("🤖 Comparación de los 4 Modelos ML")
    st.markdown("Resultados obtenidos en el conjunto de prueba (test set 20%) con validación cruzada 5-fold.")

    # Metrics table
    df_models = pd.DataFrame([
        {"Modelo": k, "Accuracy": f"{v['acc']*100:.1f}%", "Precisión": f"{v['precision']*100:.1f}%",
         "Recall": f"{v['recall']*100:.1f}%", "F1-Score": f"{v['f1']*100:.1f}%"}
        for k, v in MODEL_RESULTS.items()
    ])
    st.dataframe(df_models, use_container_width=True, hide_index=True)

    # Bar chart comparison
    st.divider()
    st.subheader("📊 Comparación Visual de Métricas")
    metrics = ["acc", "precision", "recall", "f1"]
    metric_names = ["Accuracy", "Precisión", "Recall", "F1-Score"]
    x = np.arange(len(metric_names))
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor('#06050f')
    ax.set_facecolor('#06050f')

    for i, (name, vals) in enumerate(MODEL_RESULTS.items()):
        values = [vals[m] for m in metrics]
        bars = ax.bar(x + i*width, values, width, label=name, color=vals["color"], alpha=0.85, edgecolor='none')

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(metric_names, color='white', fontsize=10)
    ax.set_ylim(0.7, 1.0)
    ax.set_ylabel("Score", color='#94a3b8', fontsize=10)
    ax.tick_params(colors='white', axis='y')
    ax.spines['bottom'].set_color('#1e293b')
    ax.spines['left'].set_color('#1e293b')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(facecolor='#0d0d1a', labelcolor='white', fontsize=9, loc='lower right')
    ax.set_title("Comparación de los 4 Modelos Supervisados", color='white', fontsize=12, pad=12)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y*100:.0f}%"))
    st.pyplot(fig)
    plt.close()

    st.success("🏆 **Modelo ganador: Random Forest** con 91.2% de exactitud — supera la hipótesis planteada del 85%.")

# ═══════════════════════════════════════════════════════════════════════════
#  PÁGINA 5 — CURVA DEL OLVIDO
# ═══════════════════════════════════════════════════════════════════════════
elif page == "📐 Curva del Olvido":
    st.title("📐 Curva del Olvido de Ebbinghaus")
    st.markdown("Visualización interactiva del decaimiento de la memoria y el efecto de la repetición espaciada.")

    col1, col2 = st.columns([1.3, 1])
    with col1:
        stability = st.slider("Estabilidad de memoria (S)", 0.5, 5.0, 1.5, step=0.1,
                              help="Mayor estabilidad = decaimiento más lento")
        n_reviews = st.slider("Número de repasos programados", 0, 5, 3)

    t = np.linspace(0, 30, 300)
    R = np.exp(-t / stability)

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor('#06050f')
    ax.set_facecolor('#0d0c1a')

    ax.plot(t, R * 100, color='#ef4444', linewidth=2.5, label='Sin repaso')
    ax.fill_between(t, R * 100, alpha=0.1, color='#ef4444')

    # Draw spaced repetitions
    if n_reviews > 0:
        s_current = stability
        t_review = 1.0
        review_times = []
        review_R = []
        for i in range(n_reviews):
            if t_review > 30: break
            review_times.append(t_review)
            r_at_review = exp(-t_review / s_current) * 100
            review_R.append(r_at_review)
            ax.axvline(t_review, color='#10b981', linewidth=1, linestyle='--', alpha=0.6)
            ax.scatter([t_review], [r_at_review], color='#10b981', s=80, zorder=5)
            ax.annotate(f"R{i+1}", (t_review, r_at_review+3), color='#34d399', fontsize=9, ha='center')
            s_current *= 2.5
            t_review += s_current

    ax.set_xlabel("Tiempo (días)", color='#94a3b8', fontsize=10)
    ax.set_ylabel("Retención (%)", color='#94a3b8', fontsize=10)
    ax.set_ylim(0, 110)
    ax.set_xlim(0, 30)
    ax.tick_params(colors='white', labelsize=9)
    ax.spines['bottom'].set_color('#1e293b')
    ax.spines['left'].set_color('#1e293b')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.axhline(50, color='#f59e0b', linewidth=1, linestyle=':', alpha=0.5, label='Umbral de olvido (50%)')
    ax.set_title(f"R(t) = e^(-t/{stability}) · Repetición Espaciada con {n_reviews} repasos",
                 color='white', fontsize=11, pad=12)
    ax.legend(facecolor='#06050f', labelcolor='white', fontsize=9)
    st.pyplot(fig)
    plt.close()

    with col2:
        st.markdown("### La Fórmula")
        st.latex(r"R(t) = e^{-t/S}")
        st.markdown("""
        - **R** = Retención (0 a 1)
        - **t** = Tiempo transcurrido (días)
        - **S** = Estabilidad de memoria

        Cada repaso exitoso **multiplica la estabilidad**,
        haciendo que la curva decaiga más lentamente.
        """)
        r_now = round(exp(0) * 100)
        r_1d = round(exp(-1/stability) * 100, 1)
        r_7d = round(exp(-7/stability) * 100, 1)
        r_30d = round(exp(-30/stability) * 100, 1)
        st.markdown("### Retención proyectada")
        st.metric("Ahora mismo", f"{r_now}%")
        st.metric("En 1 día", f"{r_1d}%")
        st.metric("En 7 días", f"{r_7d}%")
        st.metric("En 30 días", f"{r_30d}%")

# ─── Footer ─────────────────────────────────────────────────────────────────
st.divider()
st.markdown("""
<div style="text-align:center; color:#374151; font-size:13px; padding:16px 0">
    🧠 <strong>SpacedL2</strong> · Dominga Rodríguez · Maestría en IA · UFHEC · 2025<br>
    Basado en Ebbinghaus (1885) · SM-2 · scikit-learn · 4 Modelos Supervisados
</div>
""", unsafe_allow_html=True)
