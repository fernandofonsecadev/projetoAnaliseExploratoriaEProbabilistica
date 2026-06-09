# Checklist da Rubrica — Status

Legenda: ✅ atendido · 🟡 parcial · ❌ ausente

## Critério 1 — Tratamento dos dados (0,6)
| Item | Status | Onde |
|---|---|---|
| Verificar valores ausentes | ✅ | Notebook 3.1 |
| Tratar/justificar ausentes | ✅ | 3.1 (imputação mediana, justificada) |
| Espaços em branco / tipos / inconsistências | ✅ | 3.1–3.2 |
| Converter `TotalCharges` | ✅ | 3.1 / `src/preprocess.py` |
| Verificar duplicatas | ✅ | 3.2 (22 perfis, decisão de manter) |
| Detectar outliers | ✅ | 3.3 (boxplots + IQR) |
| Decidir tecnicamente sobre outliers | ✅ | 3.3 (manter, justificado) |
| Justificar cada tratamento | ✅ | Relatório §3 |
| Evitar tratamento automático sem explicação | ✅ | capping removido |
| Seção textual explicando decisões | ✅ | células markdown do notebook |

## Critério 2 — EDA e insights (0,6)
| Item | Status | Onde |
|---|---|---|
| Interpretações, não só gráficos | ✅ | markdown após cada gráfico |
| Distribuição de `Churn` | ✅ | 4.1 |
| Categóricas (Contract/Internet/Payment/TechSupport/OnlineSecurity/Paperless) | ✅ | 4.2 (6 subplots) |
| Numéricas (tenure/Monthly/Total) | ✅ | 4.3 (KDE por churn) |
| Correlações | ✅ | 4.4 (heatmap) |
| Título + objetivo + interpretação por gráfico | ✅ | 4.1–4.4 |
| Conclusões embasadas | ✅ | 4.x + §9 |

## Critério 3 — Teorema de Bayes (0,6)
| Item | Status | Onde |
|---|---|---|
| Implementação manual correta | ✅ | `src/bayes.py` |
| Explicar C, X, P(C), P(X\|C), P(C\|X), Laplace | ✅ | Notebook §6 (markdown) |
| Instância de exemplo interpretável | ✅ | 6.2 (cálculo passo a passo) |
| Exibir prioris Yes/No | ✅ | §6 |
| Exibir condicionais principais | ✅ | 6.1 |
| Não depender só de biblioteca | ✅ | classe própria |
| Bayes manual recebe entrada no dashboard | ✅ | `app.py` seção 2 |

## Critério 4 — Algoritmos de classificação (0,6)
| Item | Status | Onde |
|---|---|---|
| Dois algoritmos (LR + RF) | ✅ | §7 |
| `stratify=y` | ✅ | §5 |
| Evitar data leakage | ✅ | scaler/encoders só no treino |
| Pré-processamento explicado | ✅ | §7 markdown |
| Acurácia/precisão/recall/F1 | ✅ | §7–8 |
| Matriz de confusão | ✅ | §7 (ConfusionMatrixDisplay) |
| Comparar os dois entre si | ✅ | §8 |
| Comparar com Bayes manual | ✅ | §8 (mesmo test set) |
| Explicar qual venceu e por quê | ✅ | §8 + Relatório §7 |
| Limitações/desbalanceamento | ✅ | §8 + Relatório §9 |

## Critério 5 — Dashboard (0,6)
| Item | Status | Onde |
|---|---|---|
| Existe dashboard | ✅ | `app.py` (Streamlit) |
| Seção análise + seção classificação | ✅ | sidebar |
| Entrada de atributos pelo usuário | ✅ | seção 2 |
| Probabilidade Bayes por classe | ✅ | seção 2 |
| Predição LR e RF | ✅ | seção 2 |
| Comparação visual dos 3 | ✅ | gráfico de barras |
| Roda com `streamlit run` | ✅ | `python -m streamlit run app.py` |

## Entregáveis e organização (6–10)
| Item | Status |
|---|---|
| `RELATORIO_TECNICO.md` | ✅ |
| `README.md` | ✅ |
| `DECLARACAO_USO_IA.md` | ✅ |
| `GUIA_ARGUICAO.md` | ✅ |
| `CHECKLIST_RUBRICA.md` | ✅ |
| `requirements.txt` válido/portável | ✅ (versões `>=`) |
| `.gitignore` | ✅ |
| Caminhos relativos seguros | ✅ (`src/preprocess.py` resolve por `__file__`) |
| Modelos recarregáveis no dashboard | ✅ (classe em `src/bayes.py`) |
| Roda em outro PC pelo README | ✅ |

## Pendência da equipe (não-técnica)
- 🟡 Preencher os **nomes dos integrantes** no `README.md`.
- 🟡 Cada integrante revisar o `GUIA_ARGUICAO.md` para a defesa individual.
