import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# =============================================================================
# CONFIGURAÇÃO GERAL
# =============================================================================
st.set_page_config(
    page_title="Dashboard - Churn Telco Predictor",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main { background-color: #f8fafc; }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    h1, h2, h3 { font-family: 'Segoe UI', Arial, sans-serif; }
    .resultado-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        color: #ffffff;
        padding: 20px;
        border-radius: 16px;
        border: 1px solid #334155;
        box-shadow: 0 4px 12px rgba(0,0,0,0.18);
        min-height: 128px;
    }
    .resultado-card .label {
        color: #cbd5e1;
        font-size: 0.92rem;
        font-weight: 700;
        margin-bottom: 8px;
    }
    .resultado-card .valor {
        color: #ffffff;
        font-size: 1.85rem;
        font-weight: 800;
        margin-bottom: 8px;
    }
    .resultado-card .delta {
        display: inline-block;
        background-color: #dcfce7;
        color: #166534;
        padding: 4px 9px;
        border-radius: 999px;
        font-size: 0.85rem;
        font-weight: 700;
    }
    div[data-testid="stExpander"] {
        background-color: white;
        border-radius: 8px;
        border: 1px solid #e2e8f0;
    }
    .insight-box {
        background-color: #ffffff;
        border-left: 5px solid #2563eb;
        padding: 12px 16px;
        border-radius: 8px;
        margin-bottom: 12px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.04);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Projeto Estatística e Probabilidade")
st.markdown(
    "**Tema:** Análise de Churn em Telecomunicações & Modelos de Classificação Probabilística"
)

# IMPLEMENTAÇÃO MANUAL DO TEOREMA DE BAYES
class ImplementacaoBayesManual:
    """Classificador Naive Bayes categórico implementado manualmente.

    A classe calcula:
    - P(C): probabilidade a priori de cada classe;
    - P(X|C): verossimilhança dos atributos observados em cada classe;
    - P(C|X): probabilidade a posteriori normalizada.
    """

    def __init__(self):
        self.p_prior = {}
        self.p_conditional = {}
        self.classes = None
        self.vocab_tamanho = {}

    def treinar(self, X_cat: pd.DataFrame, y: pd.Series):
        self.classes = y.unique()
        n_total = len(y)

        for c in self.classes:
            self.p_prior[c] = len(y[y == c]) / n_total

        X_cat_df = X_cat.copy()
        X_cat_df["target"] = y.values

        for col in X_cat.columns:
            self.p_conditional[col] = {}
            self.vocab_tamanho[col] = X_cat[col].nunique()
            for c in self.classes:
                sub_df = X_cat_df[X_cat_df["target"] == c]
                counts = sub_df[col].value_counts()
                total_classe = len(sub_df)
                vocab = self.vocab_tamanho[col]
                self.p_conditional[col][c] = {}

                # Laplace: evita probabilidade zero para categorias raras.
                for val in X_cat[col].unique():
                    count = counts.get(val, 0)
                    self.p_conditional[col][c][val] = (count + 1) / (total_classe + vocab)

    def predizer_probabilidade(self, instancia_dict: dict):
        posteriors = {}
        detalhes = {}

        for c in self.classes:
            prior = self.p_prior[c]
            verossimilhanca = 1.0
            fatores = []

            for col, val in instancia_dict.items():
                if col in self.p_conditional and val in self.p_conditional[col][c]:
                    prob = self.p_conditional[col][c][val]
                else:
                    # Suavização para categoria não observada.
                    prob = 1 / (self.vocab_tamanho.get(col, 10) + 10)
                verossimilhanca *= prob
                fatores.append((col, val, prob))

            bruto = prior * verossimilhanca
            posteriors[c] = bruto
            detalhes[c] = {
                "prior": prior,
                "verossimilhanca": verossimilhanca,
                "posterior_nao_normalizado": bruto,
                "fatores": fatores,
            }

        soma = sum(posteriors.values())
        if soma > 0:
            for c in posteriors:
                posteriors[c] /= soma
        else:
            posteriors = self.p_prior.copy()

        return posteriors, detalhes


# CARREGAMENTO E TRATAMENTO DOS DADOS
@st.cache_data
def carregar_e_tratar_dados():
    diretorio_atual = os.path.dirname(os.path.abspath(__file__))
    caminho_csv = os.path.join(diretorio_atual, "WA_Fn-UseC_-Telco-Customer-Churn.csv")

    if not os.path.exists(caminho_csv):
        st.error(
            "Ficheiro não encontrado. Coloque 'WA_Fn-UseC_-Telco-Customer-Churn.csv' "
            f"na mesma pasta deste app.py: {diretorio_atual}"
        )
        st.stop()

    df_original = pd.read_csv(caminho_csv)
    df = df_original.copy()

    log_tratamento = []

    duplicatas = int(df.duplicated().sum())
    if duplicatas > 0:
        df = df.drop_duplicates()
    log_tratamento.append(
        {
            "Etapa": "Remoção de duplicatas",
            "Evidência": f"{duplicatas} registros duplicados encontrados.",
            "Justificativa técnica": "Duplicatas podem distorcer frequências, probabilidades a priori e métricas dos modelos.",
            "Impacto esperado": "Evita que clientes repetidos tenham peso indevido na análise.",
        }
    )

    espacos_total = int((df["TotalCharges"] == " ").sum())
    df["TotalCharges"] = df["TotalCharges"].replace(" ", np.nan).astype(float)
    ausentes_total = int(df["TotalCharges"].isna().sum())
    mediana_total = float(df["TotalCharges"].median())
    df["TotalCharges"] = df["TotalCharges"].fillna(mediana_total)
    log_tratamento.append(
        {
            "Etapa": "Tratamento de valores ausentes em TotalCharges",
            "Evidência": f"{espacos_total} espaços em branco convertidos em NaN; {ausentes_total} valores imputados.",
            "Justificativa técnica": "TotalCharges estava como texto por conter espaços; modelos e gráficos precisam de variável numérica.",
            "Impacto esperado": "Mantém registros válidos e reduz perda de informação usando imputação robusta pela mediana.",
        }
    )

    outlier_info = []
    for col in ["tenure", "MonthlyCharges", "TotalCharges"]:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        inf, sup = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        qtd_outliers = int(((df[col] < inf) | (df[col] > sup)).sum())
        df[col] = np.where(df[col] > sup, sup, df[col])
        df[col] = np.where(df[col] < inf, inf, df[col])
        outlier_info.append(f"{col}: {qtd_outliers}")
    log_tratamento.append(
        {
            "Etapa": "Tratamento de outliers pelo método IQR",
            "Evidência": "; ".join(outlier_info),
            "Justificativa técnica": "Valores extremos podem deslocar médias, dispersões e fronteiras de decisão dos modelos.",
            "Impacto esperado": "Reduz influência exagerada de extremos sem excluir clientes do dataset.",
        }
    )

    df["TenureCluster"] = pd.cut(
        df["tenure"],
        bins=[0, 12, 36, 100],
        labels=["Curto Prazo", "Medio Prazo", "Longo Prazo"],
        include_lowest=True,
    )
    df["MonthlyChargesCluster"] = pd.cut(
        df["MonthlyCharges"],
        bins=[0, 35, 75, 150],
        labels=["Baixo Gasto", "Medio Gasto", "Alto Gasto"],
        include_lowest=True,
    )
    log_tratamento.append(
        {
            "Etapa": "Engenharia de atributos",
            "Evidência": "Criação de TenureCluster e MonthlyChargesCluster.",
            "Justificativa técnica": "O Teorema de Bayes manual trabalha melhor com atributos categóricos interpretáveis.",
            "Impacto esperado": "Facilita cálculo de verossimilhanças e explicação conceitual na apresentação.",
        }
    )

    if "customerID" in df.columns:
        df.drop(columns=["customerID"], inplace=True, errors="ignore")
        log_tratamento.append(
            {
                "Etapa": "Remoção de identificador",
                "Evidência": "Coluna customerID removida.",
                "Justificativa técnica": "Identificadores únicos não possuem poder estatístico generalizável.",
                "Impacto esperado": "Evita ruído e risco de overfitting em modelos de classificação.",
            }
        )

    return df_original, df, pd.DataFrame(log_tratamento)


@st.cache_resource
def treinar_todos_modelos(data_df):
    df_ml = data_df.copy()
    df_ml.drop(columns=["TenureCluster", "MonthlyChargesCluster"], errors="ignore", inplace=True)

    features_bayes = ["Contract", "PaperlessBilling", "TenureCluster", "MonthlyChargesCluster"]
    X_b = data_df[features_bayes].astype(str)
    y_b = data_df["Churn"]
    bayes_m = ImplementacaoBayesManual()
    bayes_m.treinar(X_b, y_b)

    encoders = {}
    for col in df_ml.select_dtypes(include=["object", "category"]).columns:
        if col != "Churn":
            le = LabelEncoder()
            df_ml[col] = le.fit_transform(df_ml[col].astype(str))
            encoders[col] = le

    le_target = LabelEncoder()
    df_ml["Churn"] = le_target.fit_transform(df_ml["Churn"].astype(str))
    encoders["Churn"] = le_target

    X = df_ml.drop(columns=["Churn"])
    y = df_ml["Churn"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    lr = LogisticRegression(max_iter=1000, random_state=42)
    rf = RandomForestClassifier(n_estimators=250, random_state=42, class_weight="balanced")
    lr.fit(X_train_sc, y_train)
    rf.fit(X_train, y_train)

    pred_lr = lr.predict(X_test_sc)
    pred_rf = rf.predict(X_test)

    metricas = pd.DataFrame(
        {
            "Modelo": ["Regressão Logística", "Random Forest"],
            "Acurácia": [accuracy_score(y_test, pred_lr), accuracy_score(y_test, pred_rf)],
            "Precisão": [precision_score(y_test, pred_lr), precision_score(y_test, pred_rf)],
            "Recall": [recall_score(y_test, pred_lr), recall_score(y_test, pred_rf)],
            "F1-Score": [f1_score(y_test, pred_lr), f1_score(y_test, pred_rf)],
        }
    )

    matrizes = {
        "Regressão Logística": confusion_matrix(y_test, pred_lr),
        "Random Forest": confusion_matrix(y_test, pred_rf),
    }

    return {
        "bayes": bayes_m,
        "lr": lr,
        "rf": rf,
        "encoders": encoders,
        "scaler": scaler,
        "metricas": metricas,
        "matrizes": matrizes,
        "features": X.columns.tolist(),
        "features_bayes": features_bayes,
        "y_test": y_test,
    }


df_original, df, log_tratamento = carregar_e_tratar_dados()
artefatos = treinar_todos_modelos(df)
bayes_manual = artefatos["bayes"]
lr_model = artefatos["lr"]
rf_model = artefatos["rf"]
encoders = artefatos["encoders"]
scaler = artefatos["scaler"]
metricas_modelos = artefatos["metricas"]
matrizes_confusao = artefatos["matrizes"]
ml_feature_names = artefatos["features"]
features_bayes = artefatos["features_bayes"]

# =============================================================================
# FUNÇÕES AUXILIARES
# =============================================================================
def taxa_churn_por_grupo(data, coluna):
    return (
        data.groupby(coluna, observed=False)["Churn"]
        .apply(lambda s: (s == "Yes").mean() * 100)
        .sort_values(ascending=False)
        .reset_index(name="Taxa de Churn (%)")
    )


def plot_bar_churn(data, coluna, titulo):
    taxa = taxa_churn_por_grupo(data, coluna)
    fig, ax = plt.subplots(figsize=(7, 3.7))
    sns.barplot(data=taxa, x=coluna, y="Taxa de Churn (%)", ax=ax)
    ax.set_title(titulo)
    ax.set_ylabel("Taxa de Churn (%)")
    ax.tick_params(axis="x", rotation=25)
    return fig, taxa


def card_resultado(titulo, valor, detalhe=""):
    detalhe_html = f'<span class="delta">{detalhe}</span>' if detalhe else ""
    st.markdown(
        f"""<div class="resultado-card">
                <div class="label">{titulo}</div>
                <div class="valor">{valor}</div>
                {detalhe_html}
            </div>""",
        unsafe_allow_html=True,
    )


# NAVEGAÇÃO
tabs = st.tabs(
    [
        "Seção 1: Análise Exploratória",
        "Seção 2: Classificação Probabilística",
        "Validação dos Modelos",
    ]
)

# =============================================================================
# ABA 1 - EDA
# =============================================================================
with tabs[0]:
    st.header("Seção 1 - Análise Exploratória dos Dados")

    total_clientes = len(df)
    churn_rate = (df["Churn"].eq("Yes").mean() * 100)
    colm1, colm2, colm3, colm4 = st.columns(4)
    colm1.metric("Clientes analisados", f"{total_clientes:,}".replace(",", "."))
    colm2.metric("Taxa geral de churn", f"{churn_rate:.2f}%")
    colm3.metric("Cobrança mensal média", f"${df['MonthlyCharges'].mean():.2f}")
    colm4.metric("Tempo médio de contrato", f"{df['tenure'].mean():.1f} meses")

    st.markdown(
        '<div class="insight-box"><b>Insight geral:</b> a variável alvo é <b>Churn</b>, uma variável categórica binária. O objetivo é entender padrões associados à saída de clientes e prever o risco para novos perfis informados pelo usuário.</div>',
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Distribuição quantitativa: tempo de contrato")
        st.caption("Objetivo analítico: verificar se clientes recentes concentram maior volume de churn.")
        fig, ax = plt.subplots(figsize=(7, 3.7))
        sns.histplot(data=df, x="tenure", hue="Churn", kde=True, multiple="stack", ax=ax)
        ax.set_title("Distribuição de Tenure por Churn")
        ax.set_xlabel("Meses de contrato")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        st.info("Interpretação: clientes com menor tempo de contrato costumam ser um grupo crítico para retenção, pois ainda não criaram vínculo duradouro com a empresa.")

    with c2:
        st.subheader("Relação financeira: mensalidade x total pago")
        st.caption("Objetivo analítico: observar padrões entre cobrança mensal, valor acumulado e churn.")
        fig, ax = plt.subplots(figsize=(7, 3.7))
        sns.scatterplot(data=df, x="MonthlyCharges", y="TotalCharges", hue="Churn", alpha=0.55, ax=ax)
        ax.set_title("MonthlyCharges x TotalCharges por Churn")
        ax.set_xlabel("Cobrança mensal")
        ax.set_ylabel("Cobrança total")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        st.info("Interpretação: mensalidades altas combinadas com baixo tempo/baixo total acumulado podem indicar clientes novos e mais suscetíveis à saída.")

    c3, c4 = st.columns(2)
    with c3:
        st.subheader("Variável qualitativa: tipo de contrato")
        st.caption("Objetivo analítico: comparar a taxa de churn entre categorias contratuais.")
        fig, taxa_contract = plot_bar_churn(df, "Contract", "Taxa de Churn por Tipo de Contrato")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        maior = taxa_contract.iloc[0]
        st.info(f"Interpretação: a categoria com maior churn é **{maior['Contract']}**, com aproximadamente **{maior['Taxa de Churn (%)']:.1f}%**.")

    with c4:
        st.subheader("Variável qualitativa: método de pagamento")
        st.caption("Objetivo analítico: identificar meios de pagamento associados a maior churn.")
        fig, taxa_pag = plot_bar_churn(df, "PaymentMethod", "Taxa de Churn por Método de Pagamento")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        maior = taxa_pag.iloc[0]
        st.info(f"Interpretação: o método com maior churn é **{maior['PaymentMethod']}**, com aproximadamente **{maior['Taxa de Churn (%)']:.1f}%**.")

    c5, c6 = st.columns(2)
    with c5:
        st.subheader("Comparação de distribuição: MonthlyCharges")
        st.caption("Objetivo analítico: comparar a cobrança mensal entre clientes que saíram e ficaram.")
        fig, ax = plt.subplots(figsize=(7, 3.7))
        sns.boxplot(data=df, x="Churn", y="MonthlyCharges", ax=ax)
        ax.set_title("MonthlyCharges por Churn")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        st.info("Interpretação: o boxplot ajuda a verificar se o grupo com churn apresenta mensalidades mais elevadas.")

    with c6:
        st.subheader("Correlação entre variáveis numéricas")
        st.caption("Objetivo analítico: avaliar relações lineares entre variáveis quantitativas.")
        fig, ax = plt.subplots(figsize=(7, 3.7))
        corr = df[["tenure", "MonthlyCharges", "TotalCharges"]].corr()
        sns.heatmap(corr, annot=True, cmap="Blues", fmt=".2f", ax=ax)
        ax.set_title("Mapa de Correlação")
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        st.info("Interpretação: TotalCharges tende a se relacionar com tenure, pois clientes antigos acumulam maior valor pago.")

# =============================================================================
# ABA 2 - CLASSIFICAÇÃO PROBABILÍSTICA
# =============================================================================
with tabs[1]:
    st.header("Seção 2 - Sistema Interativo de Classificação Probabilística")
    inp_contract = st.sidebar.selectbox("Tipo de Contrato", ["Month-to-month", "One year", "Two year"])
    inp_paperless = st.sidebar.selectbox("Faturamento Digital", ["Yes", "No"])
    inp_internet = st.sidebar.selectbox("Serviço de Internet", ["DSL", "Fiber optic", "No"])
    inp_payment = st.sidebar.selectbox(
        "Método de Pagamento",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
    )
    inp_tenure = st.sidebar.slider("Meses de Contrato (Tenure)", 1, 72, 24)
    inp_monthly = st.sidebar.slider("Cobrança Mensal ($)", 18.0, 120.0, 65.0)

    # TotalCharges é derivado para reduzir filtros, mas manter coerência com o modelo.
    inp_total = float(inp_tenure * inp_monthly)
    st.sidebar.metric("Total acumulado estimado", f"${inp_total:,.2f}")

    t_cluster = "Curto Prazo" if inp_tenure <= 12 else ("Medio Prazo" if inp_tenure <= 36 else "Longo Prazo")
    m_cluster = "Baixo Gasto" if inp_monthly <= 35 else ("Medio Gasto" if inp_monthly <= 75 else "Alto Gasto")

    instancia_bayes = {
        "Contract": inp_contract,
        "PaperlessBilling": inp_paperless,
        "TenureCluster": t_cluster,
        "MonthlyChargesCluster": m_cluster,
    }
    probs_bayes, detalhes_bayes = bayes_manual.predizer_probabilidade(instancia_bayes)
    pred_bayes_final = max(probs_bayes, key=probs_bayes.get)

    # Para manter a interface limpa, variáveis menos importantes são preenchidas
    # com valores típicos do dataset. As seis entradas acima continuam sendo
    # responsáveis por alterar o perfil do cliente na demonstração.
    valores_padrao = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
    }

    if inp_internet == "No":
        valores_padrao.update({
            "OnlineSecurity": "No internet service",
            "OnlineBackup": "No internet service",
            "DeviceProtection": "No internet service",
            "TechSupport": "No internet service",
            "StreamingTV": "No internet service",
            "StreamingMovies": "No internet service",
        })

    instancia_ml_dict = {
        **valores_padrao,
        "tenure": inp_tenure,
        "InternetService": inp_internet,
        "Contract": inp_contract,
        "PaperlessBilling": inp_paperless,
        "PaymentMethod": inp_payment,
        "MonthlyCharges": inp_monthly,
        "TotalCharges": inp_total,
    }

    df_instancia = pd.DataFrame([instancia_ml_dict])
    for col in df_instancia.columns:
        if col in encoders and col != "Churn":
            try:
                df_instancia[col] = encoders[col].transform(df_instancia[col].astype(str))
            except ValueError:
                df_instancia[col] = 0

    df_instancia = df_instancia[ml_feature_names]
    df_instancia_scaled = scaler.transform(df_instancia)

    pred_lr_raw = lr_model.predict(df_instancia_scaled)[0]
    pred_rf_raw = rf_model.predict(df_instancia)[0]
    prob_lr_yes = lr_model.predict_proba(df_instancia_scaled)[0][list(encoders["Churn"].classes_).index("Yes")] * 100
    prob_rf_yes = rf_model.predict_proba(df_instancia)[0][list(encoders["Churn"].classes_).index("Yes")] * 100
    pred_lr_text = encoders["Churn"].inverse_transform([pred_lr_raw])[0]
    pred_rf_text = encoders["Churn"].inverse_transform([pred_rf_raw])[0]

    c_res1, c_res2, c_res3 = st.columns(3)
    with c_res1:
        card_resultado(
            "Teorema de Bayes Manual",
            f"Churn: {pred_bayes_final}",
            f"P(Yes): {probs_bayes.get('Yes', 0) * 100:.1f}%",
        )
    with c_res2:
        card_resultado("Regressão Logística", f"Churn: {pred_lr_text}", f"P(Yes): {prob_lr_yes:.1f}%")
    with c_res3:
        card_resultado("Random Forest", f"Churn: {pred_rf_text}", f"P(Yes): {prob_rf_yes:.1f}%")

    st.subheader("Probabilidades calculadas pelo Teorema de Bayes")
    df_probs_bayes = pd.DataFrame(
        {"Classe": list(probs_bayes.keys()), "Probabilidade posterior (%)": [v * 100 for v in probs_bayes.values()]}
    )
    fig, ax = plt.subplots(figsize=(7, 3.4))
    sns.barplot(data=df_probs_bayes, x="Classe", y="Probabilidade posterior (%)", ax=ax)
    ax.set_ylim(0, 100)
    ax.set_title("P(C|X) calculada manualmente pelo Teorema de Bayes")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    st.dataframe(df_probs_bayes.style.format({"Probabilidade posterior (%)": "{:.2f}"}), use_container_width=True)

    st.subheader("Comparação visual entre os três métodos")
    df_comp = pd.DataFrame(
        {
            "Método": ["Bayes Manual", "Regressão Logística", "Random Forest"],
            "Predição final": [pred_bayes_final, pred_lr_text, pred_rf_text],
            "Probabilidade de Churn = Yes (%)": [probs_bayes.get("Yes", 0) * 100, prob_lr_yes, prob_rf_yes],
        }
    )
    fig, ax = plt.subplots(figsize=(8, 3.8))
    sns.barplot(data=df_comp, x="Método", y="Probabilidade de Churn = Yes (%)", ax=ax)
    ax.set_ylim(0, 100)
    ax.set_title("Comparação da probabilidade de churn entre os métodos")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    st.dataframe(df_comp.style.format({"Probabilidade de Churn = Yes (%)": "{:.2f}"}), use_container_width=True)

    with st.expander("Demonstração conceitual do Teorema de Bayes usado no dashboard"):
        st.latex(r"P(C|X)=\frac{P(C)\cdot P(X|C)}{P(X)}")
        st.markdown(
            "Neste projeto, **C** representa a classe da variável alvo `Churn` e **X** representa os atributos informados pelo usuário. "
            "As probabilidades a priori são calculadas diretamente do dataset, e as verossimilhanças são calculadas pelas frequências condicionais dos atributos categorizados."
        )
        st.write("**Atributos usados no Bayes manual:**", ", ".join(features_bayes))
        st.write("**Entrada categorizada do cliente:**", instancia_bayes)
        for classe, det in detalhes_bayes.items():
            st.markdown(f"**Classe {classe}**")
            st.write(f"P(C): {det['prior']:.4f}")
            st.write(f"P(X|C): {det['verossimilhanca']:.8f}")
            st.write(f"P(C) × P(X|C): {det['posterior_nao_normalizado']:.8f}")
            st.dataframe(
                pd.DataFrame(det["fatores"], columns=["Atributo", "Valor observado", "P(valor|classe)"]).style.format({"P(valor|classe)": "{:.5f}"}),
                use_container_width=True,
            )

# =============================================================================
# ABA 3 - VALIDAÇÃO DOS MODELOS
# =============================================================================
with tabs[2]:
    st.header("Validação dos Modelos de Classificação")
    st.caption("A rubrica solicita dois algoritmos treinados e avaliados com métricas adequadas, além de comparação entre métodos.")

    st.subheader("Métricas de desempenho")
    st.dataframe(metricas_modelos.style.format({c: "{:.3f}" for c in metricas_modelos.columns if c != "Modelo"}), use_container_width=True)

    metricas_long = metricas_modelos.melt(id_vars="Modelo", var_name="Métrica", value_name="Valor")
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.barplot(data=metricas_long, x="Métrica", y="Valor", hue="Modelo", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_title("Comparação de Métricas: Regressão Logística x Random Forest")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)

    st.subheader("Matrizes de confusão")
    mc1, mc2 = st.columns(2)
    for col_obj, nome in zip([mc1, mc2], ["Regressão Logística", "Random Forest"]):
        with col_obj:
            fig, ax = plt.subplots(figsize=(4.5, 3.5))
            sns.heatmap(
                matrizes_confusao[nome],
                annot=True,
                fmt="d",
                cmap="Blues",
                xticklabels=encoders["Churn"].classes_,
                yticklabels=encoders["Churn"].classes_,
                ax=ax,
            )
            ax.set_xlabel("Predito")
            ax.set_ylabel("Real")
            ax.set_title(nome)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)

    melhor_f1 = metricas_modelos.sort_values("F1-Score", ascending=False).iloc[0]
    st.success(
        f"Conclusão técnica: pelo F1-Score, o melhor modelo nesta divisão treino/teste foi **{melhor_f1['Modelo']}** "
        f"com F1 = **{melhor_f1['F1-Score']:.3f}**. O F1 é importante porque equilibra precisão e recall."
    )

