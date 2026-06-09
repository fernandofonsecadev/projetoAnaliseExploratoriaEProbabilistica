"""Implementação manual do Teorema de Bayes (Naive Bayes categórico).

O objetivo deste módulo é DEMONSTRAR a compreensão conceitual do Teorema de
Bayes, e não apenas usar uma biblioteca pronta. Toda a matemática é explícita.

Teorema de Bayes:

        P(C | X) = ( P(X | C) * P(C) ) / P(X)

onde, no nosso problema de churn:
  - C  -> a classe da variável alvo `Churn` ("Yes" ou "No")
  - X  -> o conjunto de atributos observados do cliente (ex.: Contract = "Month-to-month")
  - P(C)     -> probabilidade a priori da classe (frequência no dataset)
  - P(X | C) -> verossimilhança: prob. de observar os atributos dada a classe
  - P(C | X) -> probabilidade a posteriori (o que queremos calcular)

Hipótese "naive" (ingênua): assumimos independência condicional entre os
atributos dada a classe, então:

        P(X | C) = produto de P(x_i | C) para cada atributo x_i

Suavização de Laplace: para evitar que um único atributo nunca observado
em uma classe zere todo o produto (P = 0), somamos 1 a cada contagem e o
tamanho do vocabulário (nº de categorias distintas) ao denominador:

        P(x_i | C) = ( contagem(x_i, C) + 1 ) / ( N_C + V_i )
"""

from __future__ import annotations


class ImplementacaoBayesManual:
    """Classificador Naive Bayes categórico implementado do zero.

    Atributos esperados: colunas categóricas (strings). Variáveis numéricas
    contínuas devem ser discretizadas antes (ver `preprocess.add_clusters`).
    """

    def __init__(self):
        self.classes = None              # lista de classes, ex.: ['No', 'Yes']
        self.p_prior = {}                # P(C) por classe
        self.p_conditional = {}          # P(x_i | C): {coluna: {classe: {valor: prob}}}
        self._n_por_classe = {}          # N_C: nº de amostras por classe
        self._vocab = {}                 # V_i: nº de categorias distintas por coluna
        self.feature_names = []          # ordem das colunas usadas no treino

    # ------------------------------------------------------------------ treino
    def treinar(self, X_cat, y):
        """Estima P(C) e P(x_i | C) a partir dos dados de treino.

        Parameters
        ----------
        X_cat : pandas.DataFrame  (apenas colunas categóricas, como string)
        y     : pandas.Series     (a variável alvo)
        """
        self.feature_names = list(X_cat.columns)
        self.classes = sorted(y.unique().tolist())
        n_total = len(y)

        # --- Probabilidades a priori P(C) = N_C / N_total ---
        for c in self.classes:
            n_c = int((y == c).sum())
            self._n_por_classe[c] = n_c
            self.p_prior[c] = n_c / n_total

        # --- Verossimilhanças P(x_i | C) com suavização de Laplace ---
        X = X_cat.copy()
        X["__target__"] = y.values
        for col in self.feature_names:
            self._vocab[col] = int(X_cat[col].nunique())
            self.p_conditional[col] = {}
            for c in self.classes:
                sub = X[X["__target__"] == c]
                contagens = sub[col].value_counts().to_dict()
                n_c = self._n_por_classe[c]
                v = self._vocab[col]
                self.p_conditional[col][c] = {
                    val: (cont + 1) / (n_c + v) for val, cont in contagens.items()
                }
        return self

    # ----------------------------------------------------------- probabilidade
    def _laplace_unseen(self, col, c):
        """P(x_i | C) para um valor nunca visto naquela classe (contagem 0)."""
        n_c = self._n_por_classe[c]
        v = self._vocab.get(col, 1)
        return 1 / (n_c + v)

    def predizer_probabilidade(self, instancia_dict):
        """Retorna o dicionário de probabilidades a posteriori normalizadas.

        Ex.: {'No': 0.78, 'Yes': 0.22}
        """
        posteriors = {}
        for c in self.classes:
            prob = self.p_prior[c]                       # começa com P(C)
            for col, val in instancia_dict.items():
                if col not in self.p_conditional:
                    continue
                tabela = self.p_conditional[col][c]
                prob *= tabela.get(val, self._laplace_unseen(col, c))  # * P(x_i|C)
            posteriors[c] = prob

        # Normalização pela evidência P(X) = soma dos numeradores
        evidencia = sum(posteriors.values())
        if evidencia > 0:
            posteriors = {c: p / evidencia for c, p in posteriors.items()}
        else:
            posteriors = dict(self.p_prior)
        return posteriors

    def predizer(self, instancia_dict):
        """Retorna a classe de maior probabilidade a posteriori (MAP)."""
        posteriors = self.predizer_probabilidade(instancia_dict)
        return max(posteriors, key=posteriors.get)

    def predizer_lote(self, X_cat):
        """Prediz a classe para cada linha de um DataFrame categórico."""
        return [self.predizer(linha) for linha in X_cat.to_dict(orient="records")]
