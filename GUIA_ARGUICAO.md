# Guia de Arguição Individual

Perguntas e respostas para a defesa presencial. **Todos os integrantes devem dominar todo o conteúdo** — a arguição é individual.

---

### 1. Por que esse dataset foi escolhido?
Porque churn é um problema **real e relevante** (reter clientes é mais barato que conquistar novos), mistura variáveis **categóricas e numéricas**, tem uma **variável alvo categórica clara** (`Churn`) e **exige tratamento** (a coluna `TotalCharges` vem como texto com valores em branco). Tem 7.043 clientes e 20 atributos — volume adequado para uma análise rica.

### 2. O que é churn?
É a **evasão/cancelamento** do cliente: quando ele deixa de usar o serviço da empresa. No dataset, `Churn = Yes` indica que o cliente cancelou.

### 3. Qual é a variável alvo?
`Churn` (categórica, `Yes`/`No`). Está desbalanceada: **73,5% No** e **26,5% Yes**.

### 4. Quais tratamentos foram feitos?
(1) Conversão de `TotalCharges` de texto para número + **imputação pela mediana** dos 11 valores em branco; (2) **remoção** de `customerID`; (3) verificação de **duplicatas** (22 perfis idênticos, mantidos por serem clientes distintos); (4) análise de **outliers** (mantidos, pois não há nenhum pelo IQR); (5) **discretização** de `tenure` e `MonthlyCharges` para o Bayes. Cada decisão está justificada no relatório.

### 5. Por que `TotalCharges` precisou de conversão?
Porque veio como **texto (object)** e tinha **11 espaços em branco** no lugar de números — clientes com `tenure = 0` (recém-chegados, sem cobrança acumulada). Sem converter para numérico, não dá para calcular estatísticas nem usar nos modelos.

### 6. O que são outliers e como foram tratados?
Outliers são valores **muito distantes** do padrão. Verificamos pelo critério **IQR (1,5×IQR)** e encontramos **0 outliers** nas três variáveis numéricas (são limitadas por natureza: meses, faixa de mensalidade). **Decisão: manter** — aplicar capping seria um tratamento automático sem efeito e sem justificativa.

### 7. O que é o Teorema de Bayes?
É a fórmula que atualiza a probabilidade de uma hipótese (classe `C`) à luz de evidências (atributos `X`):
**P(C|X) = P(X|C)·P(C) / P(X)**. Permite calcular a probabilidade de um cliente dar churn dado o seu perfil.

### 8. Como P(C), P(X|C) e P(C|X) foram calculados?
- **P(C)** (priori): frequência da classe no treino — `P(No)=0,735`, `P(Yes)=0,265`.
- **P(X|C)** (verossimilhança): pela hipótese **naive**, é o **produto** das `P(x_i|C)` de cada atributo, contadas no treino (ex.: `P(Contract=Month-to-month | Yes)=0,886`).
- **P(C|X)** (posteriori): multiplicamos priori × verossimilhanças e **normalizamos** pela soma das classes (a evidência `P(X)`).

### 9. O que é suavização de Laplace?
É somar **+1** a cada contagem (e o nº de categorias `V` ao denominador) para evitar que um atributo **nunca observado** em uma classe gere probabilidade **zero**, o que zeraria todo o produto. Fórmula: `P(x|C) = (contagem+1)/(N_C+V)`.

### 10. Quais modelos foram usados?
Três: **Teorema de Bayes manual** (implementado do zero), **Regressão Logística** e **Random Forest** (`scikit-learn`). Todos avaliados no mesmo conjunto de teste.

### 11. O que significam acurácia, precisão, recall e F1-score?
- **Acurácia:** % de acertos totais.
- **Precisão:** dos que o modelo disse "churn", quantos realmente eram (evita falso alarme).
- **Recall:** dos que realmente deram churn, quantos o modelo pegou (evita deixar passar).
- **F1:** média harmônica entre precisão e recall (equilíbrio). Em dados desbalanceados, **recall e F1 valem mais** que acurácia.

### 12. Qual modelo performou melhor?
**Depende do objetivo.** Para retenção, o que importa é **recall**: a **Regressão Logística** lidera (0,791). O **Bayes** tem o **melhor F1 (0,616)** — melhor equilíbrio. A **Random Forest** tem a maior acurácia (0,759), mas o menor recall. Como queremos **não perder churners**, priorizamos recall/F1.

### 13. Como funciona o dashboard?
Feito em **Streamlit** (`app.py`). Tem duas seções: **Análise dos Dados** (gráficos da EDA) e **Classificação Probabilística**, onde o usuário informa o perfil do cliente (contrato, internet, pagamento, tenure, mensalidade, etc.) e o app mostra a **probabilidade de churn** pelo Bayes, pela Regressão Logística e pela Random Forest, com **comparação visual**. Roda com `python -m streamlit run app.py`.

### 14. Quais limitações existem no projeto?
Desbalanceamento das classes; suposição de independência do Naive Bayes; uso de um subconjunto de 8 atributos (para o dashboard ser interpretável); e imputação simples pela mediana.

### 15. O que cada integrante precisa dominar?
- O **fluxo completo**: limpeza → EDA → Bayes → ML → dashboard.
- A **mecânica do Teorema de Bayes** (priori, verossimilhança, posteriori, Laplace) e saber calcular um exemplo.
- A **justificativa de cada tratamento** de dados.
- A **leitura das métricas** e por que a acurácia engana aqui.
- Como **rodar o notebook e o dashboard** em outra máquina.
