# 📉 Análise Exploratória e Probabilística de Churn de Clientes

Projeto da disciplina de **Estatística e Probabilidade** — análise exploratória, tratamento de dados, Teorema de Bayes (implementação manual), dois algoritmos de classificação e um dashboard interativo, aplicados à **previsão de churn (evasão) de clientes** de uma operadora de telecomunicações.

## 🎯 Problema

Identificar quais clientes têm maior probabilidade de **cancelar o serviço (churn)** e entender os fatores que mais influenciam essa decisão, permitindo ações de retenção. A variável categórica alvo é **`Churn`** (`Yes` / `No`).

## 👥 Integrantes

- Fernando Fonseca
- Brenda Nascimento
- Enzo Brito

## 📂 Dataset

- **Nome:** Telco Customer Churn (IBM Sample Data)
- **Fonte original:** https://www.kaggle.com/datasets/blastchar/telco-customer-churn
- **Arquivo:** `dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv`
- **Dimensões:** 7.043 clientes × 21 atributos
- **Variável alvo:** `Churn` (26,5% dos clientes deram churn)

## 🗂️ Estrutura de pastas

```
projetoAnaliseExploratoriaEProbabilistica/
├── dataset/                       # CSV original
├── notebook/
│   ├── analise_e_modelagem.ipynb  # pipeline completo (EDA → Bayes → ML)
│   └── modelos/                   # modelos .pkl + metrics.json gerados
├── src/
│   ├── bayes.py                   # Teorema de Bayes manual (classe importável)
│   └── preprocess.py              # limpeza, discretização e helpers de inferência
├── app.py                         # dashboard Streamlit
├── requirements.txt
├── documentação/
    ├── RELATORIO_TECNICO.md
    ├── GUIA_ARGUICAO.md
    ├── DECLARACAO_USO_IA.md
    └── CHECKLIST_RUBRICA.md
```

## ⚙️ Como criar o ambiente virtual

```powershell
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

```bash
# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

## 📦 Como instalar as dependências

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Verificação rápida:

```bash
python -c "import pandas, numpy, matplotlib, seaborn, sklearn, joblib; print('imports ok')"
```

## ▶️ Como executar o notebook

Abra `notebook/analise_e_modelagem.ipynb` no VS Code / Jupyter e selecione o kernel do ambiente virtual. Rode **todas as células do início ao fim** — isso regenera os modelos em `notebook/modelos/`.

Pela linha de comando (opcional):

```bash
python -m jupyter nbconvert --to notebook --execute notebook/analise_e_modelagem.ipynb
```

## 📊 Como executar o dashboard

```bash
python -m streamlit run app.py
```

O dashboard tem duas seções: **Análise dos Dados** e **Classificação Probabilística** (o usuário informa o perfil do cliente e vê a probabilidade de churn pelo Bayes manual, Regressão Logística e Random Forest, com comparação visual).

> ⚠️ O dashboard depende dos arquivos em `notebook/modelos/`. Se eles não existirem, rode o notebook antes.

## 🧪 Principais técnicas usadas

- Tratamento de dados: imputação pela mediana, análise de outliers (IQR), discretização.
- EDA com interpretação: taxa de churn por categoria, distribuições por classe, correlações.
- **Teorema de Bayes** implementado manualmente (Naive Bayes categórico com suavização de Laplace).
- **Regressão Logística** e **Random Forest** (`scikit-learn`), com `class_weight='balanced'`.
- Avaliação com acurácia, precisão, recall, F1-score e matriz de confusão.

## 🏆 Principais resultados (mesmo conjunto de teste, 1.409 clientes)

| Método | Acurácia | Precisão | Recall | F1 |
|---|---|---|---|---|
| Bayes Manual | 0,749 | 0,519 | 0,757 | **0,616** |
| Regressão Logística | 0,727 | 0,491 | **0,791** | 0,606 |
| Random Forest | **0,759** | **0,540** | 0,639 | 0,585 |

O cliente típico de risco é **novo, com fibra óptica, contrato mensal, pagamento por cheque eletrônico e mensalidade alta**. Detalhes em [`RELATORIO_TECNICO.md`](RELATORIO_TECNICO.md).

## 🤖 Uso de IA generativa

Este projeto utilizou IA generativa como **apoio** (organização de código, explicações conceituais e revisão), com compreensão e revisão da equipe. Ver [`DECLARACAO_USO_IA.md`](DECLARACAO_USO_IA.md).
