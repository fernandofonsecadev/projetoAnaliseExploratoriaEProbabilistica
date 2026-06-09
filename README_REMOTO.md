# 📊 Dashboard de Análise de Churn em Telecomunicações

## 📖 Sobre o Projeto

Este projeto foi desenvolvido para a disciplina de **Estatística e Probabilidade**, com o objetivo de aplicar conceitos de:

* Tratamento e limpeza de dados;
* Análise Exploratória de Dados (EDA);
* Probabilidade e Teorema de Bayes;
* Algoritmos de classificação supervisionada;
* Desenvolvimento de dashboards interativos.

O problema estudado é a **previsão de Churn (cancelamento de clientes)** em uma empresa de telecomunicações.

---

# 🎯 Objetivo

Analisar o comportamento dos clientes de uma operadora de telecomunicações e estimar a probabilidade de cancelamento (Churn) com base em características fornecidas pelo usuário.

Além da abordagem probabilística baseada no **Teorema de Bayes**, foram utilizados dois algoritmos de Machine Learning para comparação dos resultados:

* Regressão Logística
* Random Forest

---

# 📂 Dataset

Dataset utilizado:

**Telco Customer Churn**

Fonte:

* Kaggle
* IBM Sample Data

O conjunto de dados contém informações sobre clientes de uma empresa de telecomunicações, incluindo:

* Tempo de contrato (Tenure)
* Tipo de contrato
* Serviço de internet
* Método de pagamento
* Cobranças mensais
* Cobranças totais
* Variável alvo: Churn

---

# 🧹 Tratamento dos Dados

Foram realizadas as seguintes etapas de pré-processamento:

### Valores Ausentes

A variável `TotalCharges` continha registros vazios.

Tratamento aplicado:

* Conversão para tipo numérico;
* Substituição dos valores ausentes pela mediana da variável.

### Tratamento de Outliers

Aplicação do método:

* Intervalo Interquartil (IQR)

Variáveis analisadas:

* tenure
* MonthlyCharges
* TotalCharges

### Conversão de Tipos

Conversão de variáveis para formatos adequados para análise estatística e treinamento dos modelos.

### Engenharia de Atributos

Criação das variáveis:

* TenureCluster
* MonthlyChargesCluster

Essas variáveis foram utilizadas na implementação manual do Teorema de Bayes.

---

# 📈 Análise Exploratória dos Dados (EDA)

Foram realizadas análises para identificar:

* Distribuições de variáveis quantitativas;
* Distribuições de variáveis qualitativas;
* Relações entre atributos;
* Correlações;
* Perfis de clientes com maior propensão ao churn.

Principais insights:

* Clientes com contratos mensais apresentam maior taxa de cancelamento.
* Clientes com menor tempo de permanência possuem maior risco de churn.
* Cobranças mensais elevadas estão associadas a maiores taxas de cancelamento.
* Métodos de pagamento eletrônicos apresentam maior incidência de churn.

---

# 🧠 Implementação do Teorema de Bayes

O Teorema de Bayes foi implementado manualmente, sem utilização de bibliotecas prontas para classificação bayesiana.

Etapas realizadas:

1. Cálculo das probabilidades a priori.
2. Cálculo das verossimilhanças.
3. Aplicação da correção de Laplace.
4. Cálculo das probabilidades a posteriori.
5. Classificação da instância informada pelo usuário.

A implementação permite visualizar a probabilidade de cada classe:

* Churn = Yes
* Churn = No

---

# 🤖 Algoritmos de Classificação

Foram treinados dois modelos supervisionados:

## Regressão Logística

Modelo linear utilizado para classificação binária.

Métricas avaliadas:

* Acurácia
* Precisão
* Recall
* F1-Score

## Random Forest

Modelo baseado em múltiplas árvores de decisão.

Métricas avaliadas:

* Acurácia
* Precisão
* Recall
* F1-Score

Os resultados foram comparados com a abordagem probabilística baseada no Teorema de Bayes.

---

# 🖥️ Dashboard Interativo

O dashboard foi desenvolvido utilizando **Streamlit**.

## Funcionalidades

### Seção 1 — Análise Exploratória

* Visualização de distribuições;
* Correlações;
* Comparações entre variáveis;
* Insights estatísticos.

### Seção 2 — Classificação Probabilística

O usuário pode informar características de um novo cliente e obter:

* Probabilidade calculada pelo Teorema de Bayes;
* Predição da Regressão Logística;
* Predição do Random Forest;
* Comparação visual entre os métodos.

---

# 🛠️ Tecnologias Utilizadas

* Python
* Streamlit
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-Learn

---

# ▶️ Como Executar

## Instalar dependências

```bash
pip install -r requirements.txt
```

## Executar o dashboard

```bash
streamlit run app.py
```

---

# 📁 Estrutura do Projeto

```text
Projeto-Churn/
│
├── app.py
├── requirements.txt
├── README.md
└── WA_Fn-UseC_-Telco-Customer-Churn.csv
```

---

# 👥 Integrantes

* Enzo Martins Brito
* Fernando Fonseca
* Brenda Nascimento

---

# 🤖 Declaração de Uso de IA Generativa

Ferramentas de IA generativa foram utilizadas como apoio ao desenvolvimento do projeto para:

* Auxílio na depuração de código;
* Esclarecimento de conceitos estatísticos;
* Sugestões de visualização de dados;
* Apoio na documentação.

Todas as decisões metodológicas, análises e interpretações foram realizadas e validadas pela equipe.
