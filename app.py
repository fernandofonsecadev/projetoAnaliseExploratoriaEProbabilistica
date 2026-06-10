import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


st.set_page_config(
    page_title="Churn — Análise e Classificação",
    layout="wide",
    initial_sidebar_state="collapsed"
)

sns.set_theme(style="whitegrid")


class ImplementacaoBayesManual:
    def __init__(self):
        self.p_prior = {}
        self.p_conditional = {}
        self.classes = None

    def treinar(self, X_cat, y):
        self.classes = y.unique()
        n_total = len(y)

        for c in self.classes:
            self.p_prior[c] = len(y[y == c]) / n_total

        X_temp = X_cat.copy()
        X_temp["target"] = y

        for col in X_cat.columns:
            self.p_conditional[col] = {}

            for c in self.classes:
                sub_df = X_temp[X_temp["target"] == c]
                counts = sub_df[col].value_counts()
                total_classe = len(sub_df)
                vocab = X_cat[col].nunique()

                self.p_conditional[col][c] = {
                    val: (count + 1) / (total_classe + vocab)
                    for val, count in counts.items()
                }

    def predizer_probabilidade(self, instancia_dict):
        posteriors = {}
        soma = 0

        for c in self.classes:
            prior = self.p_prior[c]
            verossimilhanca = 1.0

            for col, val in instancia_dict.items():
                if col in self.p_conditional and val in self.p_conditional[col][c]:
                    verossimilhanca *= self.p_conditional[col][c][val]
                else:
                    verossimilhanca *= 1 / (len(self.p_prior) + 10)

            posteriors[c] = prior * verossimilhanca
            soma += posteriors[c]

        if soma > 0:
            for c in posteriors:
                posteriors[c] /= soma

        return posteriors


@st.cache_data
def carregar_e_treinar():
    base_dir = Path(__file__).resolve().parent
    caminho_csv = base_dir / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

    if not caminho_csv.exists():
        st.error(f"Arquivo CSV não encontrado em: {caminho_csv}")
        st.stop()

    df = pd.read_csv(caminho_csv)

    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    for col in ["tenure", "MonthlyCharges", "TotalCharges"]:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        inf = q1 - 1.5 * iqr
        sup = q3 + 1.5 * iqr
        df[col] = np.where(df[col] < inf, inf, df[col])
        df[col] = np.where(df[col] > sup, sup, df[col])

    df["TenureCluster"] = pd.cut(
        df["tenure"],
        bins=[-1, 12, 36, 100],
        labels=["Curto Prazo", "Médio Prazo", "Longo Prazo"]
    )

    df["MonthlyChargesCluster"] = pd.cut(
        df["MonthlyCharges"],
        bins=[0, 35, 75, 150],
        labels=["Baixo Gasto", "Médio Gasto", "Alto Gasto"]
    )

    features_bayes = [
        "Contract",
        "PaperlessBilling",
        "InternetService",
        "PaymentMethod",
        "TenureCluster",
        "MonthlyChargesCluster"
    ]

    X_bayes = df[features_bayes].astype(str)
    y_bayes = df["Churn"]

    bayes = ImplementacaoBayesManual()
    bayes.treinar(X_bayes, y_bayes)

    df_ml = df.copy()
    df_ml = df_ml.drop(columns=["TenureCluster", "MonthlyChargesCluster"])

    encoders = {}

    for col in df_ml.select_dtypes(include=["object", "category"]).columns:
        if col != "Churn":
            le = LabelEncoder()
            df_ml[col] = le.fit_transform(df_ml[col].astype(str))
            encoders[col] = le

    le_churn = LabelEncoder()
    df_ml["Churn"] = le_churn.fit_transform(df_ml["Churn"].astype(str))
    encoders["Churn"] = le_churn

    features_ml = [
        "Contract",
        "InternetService",
        "PaymentMethod",
        "PaperlessBilling",
        "tenure",
        "MonthlyCharges",
        "TotalCharges"
    ]

    X = df_ml[features_ml]
    y = df_ml["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    scaler = StandardScaler()

    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]

    X_train_scaled[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test_scaled[num_cols] = scaler.transform(X_test[num_cols])

    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_scaled, y_train)

    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        random_state=42
    )
    rf.fit(X_train, y_train)

    pred_lr = lr.predict(X_test_scaled)
    pred_rf = rf.predict(X_test)

    metricas = pd.DataFrame({
        "Métrica": ["Acurácia", "Precisão", "Recall", "F1-Score"],
        "Regressão Logística": [
            accuracy_score(y_test, pred_lr),
            precision_score(y_test, pred_lr),
            recall_score(y_test, pred_lr),
            f1_score(y_test, pred_lr)
        ],
        "Random Forest": [
            accuracy_score(y_test, pred_rf),
            precision_score(y_test, pred_rf),
            recall_score(y_test, pred_rf),
            f1_score(y_test, pred_rf)
        ]
    }).set_index("Métrica")

    return df, bayes, lr, rf, scaler, encoders, metricas, features_ml


df, bayes, lr, rf, scaler, encoders, metricas, features_ml = carregar_e_treinar()


st.title("Dashboard Analítico de Churn em Telecomunicações")

aba1, aba2 = st.tabs([
    "Seção 1: Análise dos Dados",
    "Seção 2: Classificação Probabilística"
])


with aba1:
    st.header("Análise Exploratória dos Dados")

    c1, c2, c3 = st.columns(3)

    c1.metric("Clientes", f"{len(df):,}")
    c2.metric("Taxa de Churn", f"{(df['Churn'] == 'Yes').mean():.1%}")
    c3.metric("Atributos do Dataset", len(df.columns))

    st.subheader("Distribuição da variável alvo")

    fig, ax = plt.subplots(figsize=(5, 3))
    sns.countplot(data=df, x="Churn", hue="Churn", palette="Set2", legend=False, ax=ax)
    ax.set_title("Distribuição de Churn")
    st.pyplot(fig)

    st.info(
        "A variável alvo é desbalanceada: a maioria dos clientes permanece, "
        "mas a classe Churn é a mais importante para ações de retenção."
    )

    st.subheader("Taxa de churn por categoria")

    cats = [
        "Contract",
        "InternetService",
        "PaymentMethod",
        "TechSupport",
        "OnlineSecurity",
        "PaperlessBilling"
    ]

    nomes_legiveis = {
        "Contract": "Tipo de Contrato",
        "InternetService": "Serviço de Internet",
        "PaymentMethod": "Método de Pagamento",
        "TechSupport": "Suporte Técnico",
        "OnlineSecurity": "Segurança Online",
        "PaperlessBilling": "Fatura Digital"
    }

    base = (df["Churn"] == "Yes").mean()

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(18, 10),
        constrained_layout=True
    )

    for ax, col in zip(axes.ravel(), cats):
        rates = df.groupby(col, observed=True)["Churn"].apply(
            lambda s: (s == "Yes").mean()
        ).sort_values()

        sns.barplot(
            x=rates.values * 100,
            y=rates.index,
            ax=ax,
            palette="Reds"
        )

        ax.axvline(
            base * 100,
            color="gray",
            linestyle="--",
            linewidth=1.5,
            label="Média geral"
        )

        ax.set_title(
            nomes_legiveis[col],
            fontsize=14,
            fontweight="bold",
            pad=18
        )

        ax.set_xlabel("% Churn", fontsize=11)
        ax.set_ylabel("")
        ax.tick_params(axis="both", labelsize=10)

        # melhora a leitura dos rótulos longos, principalmente no método de pagamento
        for label in ax.get_yticklabels():
            label.set_wrap(True)

        ax.grid(axis="x", alpha=0.25)

    fig.suptitle(
        "Principais Fatores Associados ao Churn",
        fontsize=18,
        fontweight="bold",
        y=1.03
    )

    st.pyplot(fig, use_container_width=True)

    st.info(
        "Os maiores riscos aparecem em clientes com contrato mensal, fibra óptica, "
        "pagamento por electronic check e ausência de suporte técnico/segurança online."
    )

    st.subheader("Variáveis numéricas por churn")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    for ax, col in zip(axes, ["tenure", "MonthlyCharges", "TotalCharges"]):
        sns.kdeplot(
            data=df,
            x=col,
            hue="Churn",
            fill=True,
            common_norm=False,
            palette="Set2",
            ax=ax
        )
        ax.set_title(col)

    plt.tight_layout()
    st.pyplot(fig)

    st.info(
        "Clientes com churn tendem a ter menor tempo de contrato e mensalidades mais altas."
    )


with aba2:
    st.header("Classificação Probabilística")

    st.write(
        "Informe o perfil do cliente para comparar o resultado do Teorema de Bayes, "
        "da Regressão Logística e da Random Forest."
    )

    col1, col2 = st.columns(2)

    with col1:
        contract = st.selectbox("Contrato", sorted(df["Contract"].unique()))
        internet = st.selectbox("Serviço de Internet", sorted(df["InternetService"].unique()))
        payment = st.selectbox("Forma de Pagamento", sorted(df["PaymentMethod"].unique()))
        paperless = st.selectbox("Fatura sem papel", sorted(df["PaperlessBilling"].unique()))

    with col2:
        tenure = st.slider("Meses de contrato", 0, 72, 12)
        monthly = st.slider("Cobrança mensal", 18.0, 120.0, 70.0)
        total = st.slider("Cobrança total", 18.0, 8600.0, 900.0)

    tenure_cluster = (
        "Curto Prazo" if tenure <= 12
        else "Médio Prazo" if tenure <= 36
        else "Longo Prazo"
    )

    monthly_cluster = (
        "Baixo Gasto" if monthly <= 35
        else "Médio Gasto" if monthly <= 75
        else "Alto Gasto"
    )

    entrada_bayes = {
        "Contract": contract,
        "PaperlessBilling": paperless,
        "InternetService": internet,
        "PaymentMethod": payment,
        "TenureCluster": tenure_cluster,
        "MonthlyChargesCluster": monthly_cluster
    }

    post = bayes.predizer_probabilidade(entrada_bayes)
    p_bayes = post.get("Yes", 0.0)

    entrada_ml = pd.DataFrame([{
        "Contract": encoders["Contract"].transform([contract])[0],
        "InternetService": encoders["InternetService"].transform([internet])[0],
        "PaymentMethod": encoders["PaymentMethod"].transform([payment])[0],
        "PaperlessBilling": encoders["PaperlessBilling"].transform([paperless])[0],
        "tenure": tenure,
        "MonthlyCharges": monthly,
        "TotalCharges": total
    }])

    entrada_ml = entrada_ml[features_ml]

    entrada_ml_scaled = entrada_ml.copy()
    entrada_ml_scaled[["tenure", "MonthlyCharges", "TotalCharges"]] = scaler.transform(
        entrada_ml[["tenure", "MonthlyCharges", "TotalCharges"]]
    )

    p_lr = lr.predict_proba(entrada_ml_scaled)[0][1]
    p_rf = rf.predict_proba(entrada_ml)[0][1]

    st.subheader("Probabilidade de Churn por método")

    m1, m2, m3 = st.columns(3)

    m1.metric("Bayes Manual", f"{p_bayes:.1%}")
    m2.metric("Regressão Logística", f"{p_lr:.1%}")
    m3.metric("Random Forest", f"{p_rf:.1%}")

    st.subheader("Comparação visual")

    fig, ax = plt.subplots(figsize=(7, 3.5))

    nomes = ["Bayes Manual", "Regressão Logística", "Random Forest"]
    valores = [p_bayes * 100, p_lr * 100, p_rf * 100]
    cores = ["#a78bfa", "#38bdf8", "#34d399"]

    ax.bar(nomes, valores, color=cores)
    ax.axhline(50, color="red", linestyle="--", label="Limiar 50%")
    ax.set_ylabel("P(Churn) %")
    ax.set_ylim(0, 100)

    for i, v in enumerate(valores):
        ax.text(i, v + 2, f"{v:.1f}%", ha="center")

    ax.legend()
    st.pyplot(fig)

    st.subheader("Métricas dos modelos")

    st.dataframe(metricas.style.format("{:.2%}"))

    consenso = np.mean([p_bayes, p_lr, p_rf])

    if consenso >= 0.5:
        st.error(f"Alto risco de churn. Média dos métodos: {consenso:.1%}")
    else:
        st.success(f"Baixo risco de churn. Média dos métodos: {consenso:.1%}")

    with st.expander("Como o Bayes chegou nesse número?"):
        st.write("Atributos usados pelo Bayes:")
        st.json(entrada_bayes)

        st.write("Probabilidades posteriores:")
        st.json({k: round(v, 4) for k, v in post.items()})
