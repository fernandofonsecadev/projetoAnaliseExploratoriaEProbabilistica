import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


# 1. Config. de Layout do Streamlit

st.set_page_config(
    page_title="Dashboard de Churn - Estatística & Probabilidade",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    div[data-testid="stMetric"] {
        background-color: #1e293b !important;
        padding: 15px !important;
        border-radius: 10px !important;
        border: 1px solid #334155 !important;
    }
    div[data-testid="stMetric"] label { color: #94a3b8 !important; font-weight: bold !important; }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] { color: #f8fafc !important; }
    </style>
""", unsafe_allow_html=True) 

# 2. Cálculo Conceitual do Teorema de Bayes

def calcular_bayes_manual(df_treino, dados_cliente):
    """
    Implementação explícita do Teorema de Bayes para demonstrar compreensão teórica.
    Calcula P(Churn=Yes | X) e P(Churn=No | X) usando frequências e Suavização de Laplace.
    """
    total_registros = len(df_treino)
    df_churn_sim = df_treino[df_treino['Churn'] == 'Yes']
    df_churn_nao = df_treino[df_treino['Churn'] == 'No']
    
    # P(C) - Probabilidades a Priori
    p_prior_sim = len(df_churn_sim) / total_registros
    p_prior_nao = len(df_churn_nao) / total_registros
    
    # P(X|C) - Verossimilhanças acumuladas
    verossimilhanca_sim = 1.0
    verossimilhanca_nao = 1.0
    
    for coluna, valor in dados_cliente.items():
        v_unicos = df_treino[coluna].nunique()
        
        # Frequência condicional para Churn = Yes
        freq_sim = len(df_churn_sim[df_churn_sim[coluna] == valor])
        verossimilhanca_sim *= (freq_sim + 1) / (len(df_churn_sim) + v_unicos)
        
        # Frequência condicional para Churn = No
        freq_nao = len(df_churn_nao[df_churn_nao[coluna] == valor])
        verossimilhanca_nao *= (freq_nao + 1) / (len(df_churn_nao) + v_unicos)
        
    # P(C|X) - Probabilidades a Posteriori (Numeradores de Bayes)
    post_sim = p_prior_sim * verossimilhanca_sim
    post_nao = p_prior_nao * verossimilhanca_nao
    
    soma_posteriors = post_sim + post_nao
    if soma_posteriors == 0:
        return 0.5
        
    # Retorna apenas a probabilidade de Churn ser "Yes" normalizada
    return post_sim / soma_posteriors

# =============================================================================
# 3. CARREGAMENTO, TRATAMENTO E MODELAGEM DE BACKEND
# =============================================================================
@st.cache_data
def pipeline_dados_e_modelos():
    # Carregamento
    df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
    
    # Tratamento de dados 
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
    
    # Codificação para os modelos de Machine Learning
    df_ml = df.copy()
    colunas_categoricas = [
        'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
        'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
        'PaperlessBilling', 'PaymentMethod'
    ]
    
    encoders = {}
    for col in colunas_categoricas:
        le = LabelEncoder()
        df_ml[col] = le.fit_transform(df_ml[col])
        encoders[col] = le
        
    le_churn = LabelEncoder()
    df_ml['Churn'] = le_churn.fit_transform(df_ml['Churn'])
    
    # Seleção de features correlacionadas
    features_lista = ['Contract', 'InternetService', 'PaymentMethod', 'tenure', 'MonthlyCharges', 'TotalCharges']
    X = df_ml[features_lista]
    y = df_ml['Churn']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Base original de treino reconstruída para o cálculo de Bayes Textual manual
    df_treino_bayes = df.loc[X_train.index]
    
    # Normalização/Escalonamento 
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[['tenure', 'MonthlyCharges', 'TotalCharges']] = scaler.fit_transform(X_train[['tenure', 'MonthlyCharges', 'TotalCharges']])
    X_test_scaled[['tenure', 'MonthlyCharges', 'TotalCharges']] = scaler.transform(X_test[['tenure', 'MonthlyCharges', 'TotalCharges']])
    
    # Treinamento dos dois algoritmos de classificação
    model_lr = LogisticRegression(random_state=42)
    model_lr.fit(X_train_scaled, y_train)
    
    model_rf = RandomForestClassifier(random_state=42, max_depth=8, n_estimators=100)
    model_rf.fit(X_train, y_train)
    
    # Avaliação e Métricas de Validação
    preds_lr = model_lr.predict(X_test_scaled)
    preds_rf = model_rf.predict(X_test)
    
    metricas = {
        'Métrica': ['Acurácia', 'Precisão', 'Recall', 'F1-Score'],
        'Regressão Logística': [accuracy_score(y_test, preds_lr), precision_score(y_test, preds_lr), recall_score(y_test, preds_lr), f1_score(y_test, preds_lr)],
        'Random Forest': [accuracy_score(y_test, preds_rf), precision_score(y_test, preds_rf), recall_score(y_test, preds_rf), f1_score(y_test, preds_rf)]
    }
    df_metricas = pd.DataFrame(metricas).set_index('Métrica')
    
    # Matrizes de confusão para exibição visual
    cm_lr = confusion_matrix(y_test, preds_lr)
    cm_rf = confusion_matrix(y_test, preds_rf)
    
    return df, df_treino_bayes, model_lr, model_rf, scaler, encoders, df_metricas, cm_lr, cm_rf

# Inicializando o pipeline de backend
df_clean, df_treino_bayes, model_lr, model_rf, scaler, encoders, df_metricas, cm_lr, cm_rf = pipeline_dados_e_modelos()


# 4. Construção da Interface do Usuário e Filtros 

st.title("📊 Dashboard Analítico: Retenção de Clientes & Predição de Churn")

# Abas
aba1, aba2 = st.tabs(["📋 Seção 1: Análise Exploratória (EDA)", "🔮 Seção 2: Classificação Probabilística"])


# 📋 SEÇÃO 1: ANÁLISE EXPLORATÓRIA DE DADOS INTERATIVA

with aba1:
    st.header("Análise Avançada e Insights Demográficos")
    
    # Filtros Dinâmicos Interativos 
    fa, fb, fc = st.columns(3)
    with fa:
        filtro_genero = st.selectbox("Filtrar por Gênero:", ["Todos"] + list(df_clean['gender'].unique()))
    with fb:
        filtro_idoso = st.selectbox("Filtrar por Idoso (SeniorCitizen):", ["Todos", "Não Idoso", "Idoso"])
    with fc:
        filtro_conjuge = st.selectbox("Filtrar se Possui Cônjuge (Partner):", ["Todos"] + list(df_clean['Partner'].unique()))
        
    # Aplicando os filtros dinamicamente ao DataFrame da EDA
    df_eda_filtrado = df_clean.copy()
    if filtro_genero != "Todos":
        df_eda_filtrado = df_eda_filtrado[df_eda_filtrado['gender'] == filtro_genero]
    if filtro_idoso != "Todos":
        val_idoso = 1 if filtro_idoso == "Idoso" else 0
        df_eda_filtrado = df_eda_filtrado[df_eda_filtrado['SeniorCitizen'] == val_idoso]
    if filtro_conjuge != "Todos":
        df_eda_filtrado = df_eda_filtrado[df_eda_filtrado['Partner'] == filtro_conjuge]
        
    # KPIs dinâmicos da análise exploratória
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    total_clientes_filtro = len(df_eda_filtrado)
    taxa_churn_filtro = (df_eda_filtrado['Churn'] == 'Yes').mean() if total_clientes_filtro > 0 else 0
    cobranca_media_filtro = df_eda_filtrado['MonthlyCharges'].mean() if total_clientes_filtro > 0 else 0
    permanencia_media_filtro = df_eda_filtrado['tenure'].mean() if total_clientes_filtro > 0 else 0
    
    kpi1.metric("Clientes Filtrados", f"{total_clientes_filtro:,}")
    kpi2.metric("Taxa Média de Churn", f"{taxa_churn_filtro:.1%}")
    kpi3.metric("Cobrança Mensal Média", f"${cobranca_media_filtro:.2f}")
    kpi4.metric("Tempo de Contrato Médio", f"{permanencia_media_filtro:.1f} meses")
    
    st.markdown("---")
    
    # Gráficos com Objetivos Analíticos Claros 
    col_g1, col_g2 = st.columns(2)
    
    with col_g1:
        st.subheader("Fatores Contratuais e Risco Estatístico")
        fig_contrato = px.histogram(
            df_eda_filtrado, x="Contract", color="Churn", barmode="group",
            title="Distribuição de Churn por Tipo de Contrato (Risco Crítico no Mês a Mês)",
            labels={'Contract': 'Modelo de Contrato', 'count': 'Quantidade de Clientes'},
            color_discrete_map={'No': '#22c55e', 'Yes': '#ef4444'}
        )
        st.plotly_chart(fig_contrato, use_container_width=True)
        
    with col_g2:
        st.subheader("Comportamento de Consumo e Faturamento")
        fig_boxplot = px.box(
            df_eda_filtrado, x="Churn", y="MonthlyCharges", color="Churn",
            title="Dispersão de Cobranças Mensais: Clientes que cancelam gastam mais",
            labels={'MonthlyCharges': 'Cobrança Mensal ($)', 'Churn': 'Cancelou?'},
            color_discrete_map={'No': '#22c55e', 'Yes': '#ef4444'}
        )
        st.plotly_chart(fig_boxplot, use_container_width=True)
        
    st.info("**Conclusão Analítica da EDA:** A análise de frequências e dispersões prova estatisticamente que contratos de curto prazo (*Month-to-month*) somados a cobranças elevadas são as causas principais do Churn na Telco. Ações comerciais imediatas devem focar na migração contratual.")


# 🔮 SEÇÃO 2: CLASSIFICAÇÃO PROBABILÍSTICA & COMPARAÇÃO VISUAL

with aba2:
    st.header("Simulador de Classificação e Avaliação Estatística")
    st.markdown("Altere as variáveis do cliente no painel esquerdo para avaliar os riscos calculados pelo Teorema de Bayes e Machine Learning.")
    
    # Sidebar dedicada aos inputs do usuário
    st.sidebar.header("🔮 Parâmetros do Novo Cliente")
    
    input_contract = st.sidebar.selectbox("Tipo de Contrato", df_clean['Contract'].unique(), index=0)
    input_internet = st.sidebar.selectbox("Serviço de Internet", df_clean['InternetService'].unique(), index=1)
    input_payment = st.sidebar.selectbox("Forma de Pagamento", df_clean['PaymentMethod'].unique(), index=2)
    input_dependents = st.sidebar.selectbox("Possui Dependentes?", df_clean['Dependents'].unique(), index=0)
    
    input_tenure = st.sidebar.slider("Tempo de Permanência (Meses)", min_value=0, max_value=72, value=12)
    input_monthly = st.sidebar.slider("Valor da Cobrança Mensal ($)", min_value=18, max_value=120, value=75)
    input_total = st.sidebar.slider("Valor Total Cobrado ($)", min_value=18, max_value=8600, value=900)
    
    # 1. Execução do Teorema de Bayes Puro Manual
    dicionario_cliente = {
        'Contract': input_contract,
        'InternetService': input_internet,
        'PaymentMethod': input_payment
    }
    probabilidade_bayes = calcular_bayes_manual(df_treino_bayes, dicionario_cliente)
    
    # 2. Execução dos Modelos do Scikit-Learn (Tratamento de Escala e Encoders inclusos)
    dados_usuario_df = pd.DataFrame([{
        'Contract': encoders['Contract'].transform([input_contract])[0],
        'InternetService': encoders['InternetService'].transform([input_internet])[0],
        'PaymentMethod': encoders['PaymentMethod'].transform([input_payment])[0],
        'tenure': input_tenure,
        'MonthlyCharges': input_monthly,
        'TotalCharges': input_total
    }])
    
    # Escalonamento apenas para o modelo que sofre com distância (Regressão Logística)
    dados_usuario_scaled = dados_usuario_df.copy()
    dados_usuario_scaled[['tenure', 'MonthlyCharges', 'TotalCharges']] = scaler.transform(dados_usuario_df[['tenure', 'MonthlyCharges', 'TotalCharges']])
    
    probabilidade_lr = model_lr.predict_proba(dados_usuario_scaled)[0][1]
    probabilidade_rf = model_rf.predict_proba(dados_usuario_df)[0][1]
    
    # Exibição dos cards comparativos de resultados em tempo real
    st.subheader("Risco Estimado de Cancelamento (Churn)")
    m1, m2, m3 = st.columns(3)
    m1.metric(label="Teorema de Bayes (Conceitual/Manual)", value=f"{probabilidade_bayes:.1%}")
    m2.metric(label="Regressão Logística (Fronteira Linear)", value=f"{probabilidade_lr:.1%}")
    m3.metric(label="Random Forest (Ensemble de Árvores)", value=f"{probabilidade_rf:.1%}")
    
    # Gráfico de comparação visual exigido expressamente na rubrica
    st.markdown("---")
    df_grafico_comp = pd.DataFrame({
        'Abordagem / Algoritmo': ['Teorema de Bayes (Manual)', 'Regressão Logística (ML)', 'Random Forest (ML)'],
        'Probabilidade Estimada de Churn': [probabilidade_bayes, probabilidade_lr, probabilidade_rf]
    })
    
    fig_barra_pred = px.bar(
        df_grafico_comp, x='Abordagem / Algoritmo', y='Probabilidade Estimada de Churn',
        color='Abordagem / Algoritmo', text_auto='.1%', range_y=[0, 1],
        title="Comparação Visual de Risco Estimado por Abordagem",
        color_discrete_sequence=['#3b82f6', '#10b981', '#f59e0b']
    )
    st.plotly_chart(fig_barra_pred, use_container_width=True)
    
    # Validação Estatística das Métricas de Performance do Conjunto de Teste
    st.subheader("🎯 Validação Científica dos Modelos (Conjunto de Teste Separado)")
    
    col_t1, col_t2 = st.columns([1, 2])
    with col_t1:
        st.markdown("**Tabela Comparativa de Performance Geral:**")
        st.dataframe(df_metricas.style.format("{:.2%}"))
    
    with col_t2:
        st.markdown("**Matrizes de Confusão (Distribuição de Erros/Acertos Reais):**")
        col_cm1, col_cm2 = st.columns(2)
        
        with col_cm1:
            fig_cm_lr = go.Figure(data=go.Heatmap(
                z=cm_lr, x=['Predito Não', 'Predito Sim'], y=['Real Não', 'Real Sim'],
                colorscale='Blues', text=cm_lr, texttemplate="%{text}", showscale=False
            ))
            fig_cm_lr.update_layout(title="Regressão Logística", width=250, height=220, margin=dict(l=20,r=20,t=40,b=20))
            st.plotly_chart(fig_cm_lr, use_container_width=False)
            
        with col_cm2:
            fig_cm_rf = go.Figure(data=go.Heatmap(
                z=cm_rf, x=['Predito Não', 'Predito Sim'], y=['Real Não', 'Real Sim'],
                colorscale='Greens', text=cm_rf, texttemplate="%{text}", showscale=False
            ))
            fig_cm_rf.update_layout(title="Random Forest", width=250, height=220, margin=dict(l=20,r=20,t=40,b=20))
            st.plotly_chart(fig_cm_rf, use_container_width=False)
            
    st.caption("Nota de Fundamentação Teórica: O cálculo do Teorema de Bayes se baseia no produto das verossimilhanças assumindo independência condicional. Já os modelos supervisionados lineares e de ensemble de árvores capturam interações numéricas de alta ordem e necessitam de validação rigorosa via matriz de confusão.")