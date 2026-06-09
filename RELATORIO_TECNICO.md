# Relatório Técnico — Análise Probabilística de Churn

> Projeto de Estatística e Probabilidade · Dataset: Telco Customer Churn · Alvo: `Churn`

---

## 1. Descrição do dataset

| Item | Valor |
|---|---|
| Origem | IBM Sample Data — distribuído via [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) |
| Domínio | Telecomunicações (clientes de uma operadora) |
| Nº de instâncias | 7.043 clientes |
| Nº de atributos | 21 (20 preditores + 1 alvo) |
| Variável alvo | `Churn` — o cliente cancelou o serviço? (`Yes` / `No`) |
| Tipos | demográficos (gênero, idoso, dependentes), serviços (internet, suporte, streaming), contrato e cobrança |

A distribuição do alvo é **desbalanceada**: **73,5% `No`** e **26,5% `Yes`**.

## 2. Justificativa da escolha

- É um problema **real e relevante** (retenção de clientes), não dados aleatórios.
- Combina variáveis **categóricas e numéricas**, adequado à EDA e à modelagem.
- Possui uma **variável categórica de interesse clara** (`Churn`) para a análise bayesiana.
- **Exige tratamento** (a coluna `TotalCharges` vem como texto com valores em branco), atendendo ao requisito de limpeza.
- Volume e complexidade adequados (7 mil linhas, 20 atributos) para uma análise rica.

## 3. Tratamentos aplicados

| # | Problema detectado | Decisão técnica | Justificativa / impacto |
|---|---|---|---|
| 1 | `TotalCharges` é texto com **11 espaços em branco** (clientes com `tenure = 0`) | Converter para numérico e **imputar pela mediana** | Mediana é robusta à assimetria; preserva as 11 linhas sem distorcer a distribuição |
| 2 | `customerID` é identificador | **Remover** | Sem valor preditivo; evita ruído na modelagem |
| 3 | **Duplicatas**: 22 perfis idênticos (após remover ID) | **Manter** | Sem ID único, são clientes distintos com perfis coincidentes — não são erros de coleta |
| 4 | **Outliers** em `tenure`, `MonthlyCharges`, `TotalCharges` | **Manter** (sem capping) | Pelo IQR (1,5×IQR) há **0 outliers**; variáveis limitadas por natureza. Capping seria tratamento automático sem efeito |
| 5 | Bayes exige atributos discretos | **Discretizar** `tenure`→`TenureCluster` e `MonthlyCharges`→`MonthlyChargesCluster` | Habilita o Naive Bayes categórico; bins incluem `tenure = 0` |

Trecho-chave (em [`src/preprocess.py`](src/preprocess.py)):

```python
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].replace(" ", np.nan), errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
df = df.drop(columns=["customerID"])
```

## 4. Insights da análise exploratória

Referência aos gráficos do notebook (Seção 4). Taxa de churn de cada grupo comparada à **média geral de 26,5%**:

- **Contrato (`Contract`)** — fator mais forte: `Month-to-month` = **42,7%** vs. `Two year` = **2,8%**.
- **Internet (`InternetService`)** — `Fiber optic` = **41,9%** vs. `No internet` = **7,4%**.
- **Pagamento (`PaymentMethod`)** — `Electronic check` = **45,3%** vs. pagamentos automáticos (~15–17%).
- **Suporte/Segurança** — ausência de `TechSupport`/`OnlineSecurity` ~**41%** vs. ~15% com o serviço.
- **Numéricas** — quem dá churn tem `tenure` médio de **18 meses** (vs. 37,6) e mensalidade **74,4** (vs. 61,3).
- **Correlação** — `tenure` correlaciona-se **negativamente** com churn; `MonthlyCharges`, **positivamente**; `TotalCharges` é redundante com `tenure`.

**Conclusão da EDA:** o cliente de risco é **novo, com fibra, contrato mensal, pagamento por cheque eletrônico e mensalidade alta**.

## 5. Análise probabilística — Teorema de Bayes

Implementação manual (Naive Bayes categórico) em [`src/bayes.py`](src/bayes.py):

$$P(C \mid X) = \frac{P(X \mid C)\,P(C)}{P(X)}, \qquad P(X \mid C) = \prod_i P(x_i \mid C)$$

com suavização de Laplace $P(x_i \mid C) = \dfrac{\text{cont}(x_i,C)+1}{N_C+V_i}$.

### 5.1 Probabilidades a priori P(C)

| Classe | P(C) |
|---|---|
| `Churn = No` | 0,7346 |
| `Churn = Yes` | 0,2654 |

### 5.2 Verossimilhanças P(x | C) — exemplos

**P(Contract | Churn):**

| Contract | P( · \| No) | P( · \| Yes) |
|---|---|---|
| Month-to-month | 0,429 | **0,886** |
| One year | 0,252 | 0,087 |
| Two year | 0,319 | 0,027 |

**P(InternetService | Churn):**

| InternetService | P( · \| No) | P( · \| Yes) |
|---|---|---|
| Fiber optic | 0,347 | **0,698** |
| DSL | 0,380 | 0,242 |
| No | 0,272 | 0,059 |

**P(TenureCluster | Churn):**

| TenureCluster | P( · \| No) | P( · \| Yes) |
|---|---|---|
| Curto Prazo (0-12m) | 0,221 | **0,549** |
| Medio Prazo (13-36m) | 0,269 | 0,250 |
| Longo Prazo (37m+) | 0,510 | 0,200 |

### 5.3 Probabilidade a posteriori P(C | X) — exemplo

Para um cliente `Month-to-month` + `Fiber optic` + `Curto Prazo`, multiplicam-se as priori pelas verossimilhanças de cada atributo (Seção 6.2 do notebook). Como cada termo é muito maior na classe `Yes`, o numerador de `Yes` domina e, após a normalização pela evidência, **P(Churn = Yes | X) supera 0,5** — o modelo prevê churn. O cálculo completo, termo a termo, está impresso no notebook.

## 6. Resultados dos algoritmos de classificação

Treino/teste estratificado 80/20 (5.634 / 1.409), `class_weight='balanced'`, classe positiva = `Churn`.

| Modelo | Acurácia | Precisão | Recall | F1 |
|---|---|---|---|---|
| Regressão Logística | 0,727 | 0,491 | **0,791** | 0,606 |
| Random Forest | **0,759** | **0,540** | 0,639 | 0,585 |

**Matrizes de confusão** (`[[VN, FP], [FN, VP]]`, rótulos [No, Yes]):

- Regressão Logística: `[[728, 307], [78, 296]]`
- Random Forest: `[[831, 204], [135, 239]]`

## 7. Comparação entre Bayes, Regressão Logística e Random Forest

Todos avaliados nas **mesmas 1.409 linhas de teste**:

| Método | Acurácia | Precisão | Recall | F1 |
|---|---|---|---|---|
| **Bayes Manual** | 0,749 | 0,519 | 0,757 | **0,616** |
| Regressão Logística | 0,727 | 0,491 | **0,791** | 0,606 |
| Random Forest | **0,759** | **0,540** | 0,639 | 0,585 |

**Leitura:**
- **Não há um vencedor único** — depende do objetivo. Para **reter clientes**, o que importa é **recall na classe Churn** (não deixar passar quem vai sair): a **Regressão Logística** lidera (0,791), seguida do **Bayes** (0,757).
- O **Bayes Manual** tem o **melhor F1 (0,616)**, ou seja, o melhor equilíbrio precisão×recall — impressionante por usar só 8 atributos categóricos e cálculo explícito.
- A **Random Forest** tem a maior acurácia/precisão, mas o **menor recall** — erra mais churners reais, o pior tipo de erro para retenção.
- **Limitação das métricas:** por causa do desbalanceamento (26,5% churn), a **acurácia é enganosa** (um modelo trivial "sempre No" atinge 73,5%). Por isso o foco em recall/F1.

## 8. Conclusões

- O churn é explicado sobretudo por **tipo de contrato, tempo de casa, serviço de internet e forma de pagamento**.
- A abordagem bayesiana **manual** demonstrou, de forma transparente, como esses fatores elevam a probabilidade de churn, com desempenho competitivo frente aos modelos de ML.
- Os três métodos **concordam nos padrões** e se complementam: Bayes pela interpretabilidade, LR pelo recall, RF pela precisão.

## 9. Limitações

- **Desbalanceamento** das classes (26,5% churn) limita precisão e exige foco em recall/F1.
- O **Naive Bayes** assume independência condicional entre atributos — irreal, mas eficaz aqui.
- Usamos um **subconjunto curado de 8 atributos** (para manter o dashboard interpretável); mais atributos poderiam elevar a performance.
- Imputação pela mediana é uma simplificação para os 11 clientes novos.

## 10. Aprendizados

- A diferença entre **avaliar por acurácia vs. recall/F1** em dados desbalanceados.
- Como o **Teorema de Bayes** opera por trás de um classificador, incluindo o papel da **suavização de Laplace**.
- Boas práticas de engenharia: **evitar data leakage** (scaler ajustado só no treino), modularizar código (`src/`) e salvar/recarregar modelos de forma reprodutível.
