"""Limpeza, discretização e helpers de inferência reutilizados pelo
notebook e pelo dashboard (`app.py`).

Centralizar isto evita duplicação de lógica e garante que o cliente montado
no dashboard passe pelo MESMO pré-processamento usado no treino dos modelos.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd

# --------------------------------------------------------------------- caminhos
# Resolução robusta de caminhos a partir da localização DESTE arquivo, para que
# o projeto funcione em qualquer máquina/diretório de trabalho.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(PROJECT_ROOT, "dataset", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
MODELS_DIR = os.path.join(PROJECT_ROOT, "notebook", "modelos")

# ------------------------------------------------------------------- features
# Conjunto curado de atributos: fortes preditores de churn e que o usuário
# consegue informar no dashboard. Os TRÊS modelos são treinados sobre as mesmas
# linhas (mesmo split) para uma comparação justa.
CATEGORICAL_FEATURES = [
    "Contract",
    "PaperlessBilling",
    "InternetService",
    "OnlineSecurity",
    "TechSupport",
    "PaymentMethod",
]
NUMERIC_FEATURES = ["tenure", "MonthlyCharges"]
ML_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
# O Bayes usa as versões DISCRETIZADAS das numéricas (Naive Bayes categórico).
BAYES_FEATURES = CATEGORICAL_FEATURES + ["TenureCluster", "MonthlyChargesCluster"]

TARGET = "Churn"

# Bins de discretização. A borda inferior é -1 para INCLUIR clientes com
# tenure == 0 (11 clientes novos), que de outra forma virariam NaN.
TENURE_BINS = [-1, 12, 36, 100]
TENURE_LABELS = ["Curto Prazo (0-12m)", "Medio Prazo (13-36m)", "Longo Prazo (37m+)"]
MONTHLY_BINS = [-1, 35, 75, 200]
MONTHLY_LABELS = ["Baixo Gasto", "Medio Gasto", "Alto Gasto"]


# ----------------------------------------------------------------- limpeza
def load_and_clean(path: str = DATASET_PATH) -> pd.DataFrame:
    """Carrega o CSV e aplica a limpeza justificada no relatório técnico.

    - `TotalCharges` vem como texto com 11 espaços em branco (clientes novos,
      tenure == 0). Convertemos para numérico e imputamos pela MEDIANA.
    - Removemos `customerID` (identificador, sem valor preditivo).
    """
    df = pd.read_csv(path)

    # TotalCharges: " " -> NaN -> float -> imputação pela mediana
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].replace(" ", np.nan), errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])
    return df


def tenure_to_cluster(value: float) -> str:
    return str(pd.cut([value], bins=TENURE_BINS, labels=TENURE_LABELS)[0])


def monthly_to_cluster(value: float) -> str:
    return str(pd.cut([value], bins=MONTHLY_BINS, labels=MONTHLY_LABELS)[0])


def add_clusters(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona as colunas discretizadas usadas pelo Bayes."""
    df = df.copy()
    df["TenureCluster"] = pd.cut(df["tenure"], bins=TENURE_BINS, labels=TENURE_LABELS)
    df["MonthlyChargesCluster"] = pd.cut(df["MonthlyCharges"], bins=MONTHLY_BINS, labels=MONTHLY_LABELS)
    return df


# --------------------------------------------------------- helpers de inferência
def build_bayes_input(user: dict) -> dict:
    """Monta o dicionário de atributos categóricos para o Bayes manual,
    derivando os clusters a partir de `tenure` e `MonthlyCharges`."""
    entrada = {f: user[f] for f in CATEGORICAL_FEATURES}
    entrada["TenureCluster"] = tenure_to_cluster(user["tenure"])
    entrada["MonthlyChargesCluster"] = monthly_to_cluster(user["MonthlyCharges"])
    return entrada


def build_ml_vector(user: dict, encoders: dict) -> pd.DataFrame:
    """Monta a linha de features para LR/RF na ordem de `ML_FEATURES`,
    codificando as categóricas com os LabelEncoders salvos no treino.

    Retorna um DataFrame (com nomes de coluna) para casar com o formato usado
    no treino e evitar o warning de "missing feature names" do scikit-learn.
    """
    linha = {}
    for col in CATEGORICAL_FEATURES:
        linha[col] = int(encoders[col].transform([str(user[col])])[0])
    for col in NUMERIC_FEATURES:
        linha[col] = float(user[col])
    return pd.DataFrame([linha])[ML_FEATURES]
