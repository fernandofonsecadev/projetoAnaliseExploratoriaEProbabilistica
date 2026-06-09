"""Dashboard interativo de churn — Streamlit.

Executar:  python -m streamlit run app.py

Seção 1 — Análise dos Dados: principais padrões da EDA.
Seção 2 — Classificação Probabilística: o usuário informa os atributos e o app
mostra a probabilidade do Teorema de Bayes (manual) e as predições da Regressão
Logística e da Random Forest, com comparação visual entre os três métodos.
"""

import os
import json

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# `src` é importável porque app.py está na raiz do projeto.
# Importar a classe é OBRIGATÓRIO para o joblib recarregar bayes_manual.pkl.
from src import preprocess as prep
from src.bayes import ImplementacaoBayesManual  # noqa: F401  (necessário para o unpickle)

st.set_page_config(page_title="Churn — Análise e Classificação", layout="wide")
sns.set_theme(style="whitegrid")

ROTULOS = {0: "Permanece (No)", 1: "Churn (Yes)"}


# --------------------------------------------------------------- carregamento
@st.cache_data
def carregar_dados():
    df = prep.load_and_clean()
    return prep.add_clusters(df)


@st.cache_resource
def carregar_modelos():
    p = prep.MODELS_DIR
    artefatos = {
        "bayes": joblib.load(os.path.join(p, "bayes_manual.pkl")),
        "lr": joblib.load(os.path.join(p, "regressao_logistica.pkl")),
        "rf": joblib.load(os.path.join(p, "random_forest.pkl")),
        "scaler": joblib.load(os.path.join(p, "scaler.pkl")),
        "encoders": joblib.load(os.path.join(p, "encoders.pkl")),
    }
    metrics_path = os.path.join(p, "metrics.json")
    artefatos["metrics"] = json.load(open(metrics_path, encoding="utf-8")) if os.path.exists(metrics_path) else None
    return artefatos


try:
    df = carregar_dados()
    art = carregar_modelos()
except FileNotFoundError:
    st.error("Modelos não encontrados. Rode o notebook `notebook/analise_e_modelagem.ipynb` "
             "do início ao fim para gerar os arquivos em `notebook/modelos/`.")
    st.stop()


# ------------------------------------------------------------------ navegação
st.sidebar.title("🔎 Churn Telco")
secao = st.sidebar.radio("Navegação", ["1. Análise dos Dados", "2. Classificação Probabilística"])
st.sidebar.caption("Dataset: Telco Customer Churn · Alvo: Churn")


# ============================================================ SEÇÃO 1: ANÁLISE
if secao == "1. Análise dos Dados":
    st.title("📊 Análise Exploratória dos Dados")

    c1, c2, c3 = st.columns(3)
    c1.metric("Clientes", f"{len(df):,}")
    c2.metric("Taxa de Churn", f"{(df['Churn'] == 'Yes').mean()*100:.1f}%")
    c3.metric("Atributos usados nos modelos", len(prep.ML_FEATURES))

    st.subheader("Distribuição da variável alvo (Churn)")
    st.caption("Objetivo: medir o balanceamento das classes.")
    fig, ax = plt.subplots(figsize=(5, 3.2))
    sns.countplot(data=df, x="Churn", hue="Churn", palette="Set2", legend=False, ax=ax)
    st.pyplot(fig)
    st.info("As classes são desbalanceadas (~73,5% No / 26,5% Yes). Por isso priorizamos "
            "**recall** e **F1** da classe Churn, não apenas a acurácia.")

    st.subheader("Taxa de churn por categoria")
    st.caption("Objetivo: identificar os perfis de cliente com maior evasão (linha = média geral).")
    cats = ["Contract", "InternetService", "PaymentMethod", "TechSupport", "OnlineSecurity", "PaperlessBilling"]
    base = (df["Churn"] == "Yes").mean()
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for ax, col in zip(axes.ravel(), cats):
        rates = df.groupby(col, observed=True)["Churn"].apply(lambda s: (s == "Yes").mean()).sort_values()
        sns.barplot(x=rates.values * 100, y=rates.index, ax=ax, palette="Reds")
        ax.axvline(base * 100, color="gray", ls="--")
        ax.set_title(col)
        ax.set_xlabel("% Churn")
    plt.tight_layout()
    st.pyplot(fig)
    st.info("Maiores fatores de churn: contrato **Month-to-month** (42,7%), **Fiber optic** (41,9%), "
            "pagamento por **Electronic check** (45,3%) e ausência de **TechSupport/OnlineSecurity**.")

    st.subheader("Variáveis numéricas por churn")
    st.caption("Objetivo: comparar o perfil de tempo de casa e gasto entre quem fica e quem sai.")
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for ax, col in zip(axes, ["tenure", "MonthlyCharges", "TotalCharges"]):
        sns.kdeplot(data=df, x=col, hue="Churn", fill=True, common_norm=False, palette="Set2", ax=ax)
        ax.set_title(col)
    plt.tight_layout()
    st.pyplot(fig)
    st.info("Quem dá churn é tipicamente **mais novo** (tenure ~18 vs. 38 meses) e tem **mensalidade mais alta**.")

    if art["metrics"]:
        st.subheader("Desempenho dos modelos (mesmo conjunto de teste)")
        tab = pd.DataFrame({k: v for k, v in
                            {n: {kk: m[kk] for kk in ["acuracia", "precisao", "recall", "f1"]}
                             for n, m in art["metrics"]["models"].items()}.items()}).T
        st.dataframe(tab.style.format("{:.3f}").highlight_max(axis=0, color="#86efac"))
        st.caption("Verde = melhor método em cada métrica. Bayes lidera em F1; Regressão Logística em recall; "
                   "Random Forest em acurácia/precisão.")


# ================================================ SEÇÃO 2: CLASSIFICAÇÃO
else:
    st.title("🤖 Classificação Probabilística")
    st.write("Informe o perfil do cliente. O app calcula a probabilidade de churn por **três métodos** "
             "e compara os resultados.")

    enc = art["encoders"]
    col1, col2 = st.columns(2)
    with col1:
        contract = st.selectbox("Contrato", list(enc["Contract"].classes_))
        internet = st.selectbox("Serviço de Internet", list(enc["InternetService"].classes_))
        payment = st.selectbox("Forma de Pagamento", list(enc["PaymentMethod"].classes_))
        paperless = st.selectbox("Fatura sem papel (PaperlessBilling)", list(enc["PaperlessBilling"].classes_))
    with col2:
        techsupport = st.selectbox("Suporte Técnico (TechSupport)", list(enc["TechSupport"].classes_))
        security = st.selectbox("Segurança Online (OnlineSecurity)", list(enc["OnlineSecurity"].classes_))
        tenure = st.slider("Meses de contrato (tenure)", 0, 72, 12)
        monthly = st.slider("Cobrança mensal (MonthlyCharges)", 18.0, 120.0, 70.0)

    user = {
        "Contract": contract, "PaperlessBilling": paperless, "InternetService": internet,
        "OnlineSecurity": security, "TechSupport": techsupport, "PaymentMethod": payment,
        "tenure": tenure, "MonthlyCharges": monthly,
    }

    if st.button("Calcular probabilidade de churn", type="primary"):
        # --- Bayes manual ---
        entrada_bayes = prep.build_bayes_input(user)
        post = art["bayes"].predizer_probabilidade(entrada_bayes)
        p_bayes = post.get("Yes", 0.0)

        # --- LR e RF ---
        vec = prep.build_ml_vector(user, enc)
        vec_s = art["scaler"].transform(vec)
        p_lr = float(art["lr"].predict_proba(vec_s)[0][1])
        p_rf = float(art["rf"].predict_proba(vec)[0][1])

        st.subheader("Probabilidade de Churn por método")
        m1, m2, m3 = st.columns(3)
        for col, nome, p in [(m1, "Bayes Manual", p_bayes), (m2, "Regressão Logística", p_lr), (m3, "Random Forest", p_rf)]:
            col.metric(nome, f"{p*100:.1f}%", "Churn" if p >= 0.5 else "Permanece",
                       delta_color="inverse")

        st.subheader("Comparação visual")
        fig, ax = plt.subplots(figsize=(7, 3.2))
        nomes = ["Bayes Manual", "Regressão Logística", "Random Forest"]
        valores = [p_bayes * 100, p_lr * 100, p_rf * 100]
        cores = ["#a78bfa", "#38bdf8", "#34d399"]
        ax.bar(nomes, valores, color=cores)
        ax.axhline(50, color="red", ls="--", label="Limiar 50%")
        ax.set_ylabel("P(Churn) %")
        ax.set_ylim(0, 100)
        for i, v in enumerate(valores):
            ax.text(i, v + 2, f"{v:.1f}%", ha="center")
        ax.legend()
        st.pyplot(fig)

        consenso = np.mean([p_bayes, p_lr, p_rf])
        if consenso >= 0.5:
            st.error(f"⚠️ Risco de churn (média dos métodos: {consenso*100:.1f}%). Cliente candidato a ação de retenção.")
        else:
            st.success(f"✅ Baixo risco de churn (média dos métodos: {consenso*100:.1f}%).")

        with st.expander("Como o Bayes chegou nesse número?"):
            st.write("Atributos categóricos usados pelo Bayes:", entrada_bayes)
            st.write("Probabilidades a posteriori P(C | X):", {k: round(v, 4) for k, v in post.items()})
