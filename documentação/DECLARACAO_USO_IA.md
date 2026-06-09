# Declaração de Uso de IA Generativa

Conforme exigido pela disciplina, declaramos de forma transparente o uso de ferramentas de IA generativa neste projeto.

## O que foi feito com IA

A IA foi utilizada como **apoio**, principalmente em:

- **Organização e refatoração de código** — estruturação do pacote `src/` (separando a classe do Teorema de Bayes e o pré-processamento) para que os modelos salvos pudessem ser recarregados pelo dashboard sem erros.
- **Explicações conceituais** — esclarecimento do funcionamento do Teorema de Bayes, da suavização de Laplace e da diferença entre as métricas (acurácia, precisão, recall, F1) em cenários desbalanceados.
- **Revisão técnica** — identificação de pontos de melhoria, como o tratamento de outliers que estava sendo aplicado sem justificativa e a ausência de comparação entre os três métodos no mesmo conjunto de teste.
- **Apoio na redação** da documentação (README, relatório técnico e guia de arguição).

## Objetivo de aprendizado

O objetivo ao usar a ferramenta foi **compreender melhor os conceitos** (especialmente a mecânica do Teorema de Bayes e a avaliação de modelos em dados desbalanceados) e **adotar boas práticas de engenharia** (evitar data leakage, modularizar o código, garantir reprodutibilidade), e **não** terceirizar o raciocínio do projeto.

## Dificuldades que motivaram o uso

- Erro de ambiente (`ModuleNotFoundError`) causado por múltiplos ambientes virtuais e seleção de kernel incorreta no VS Code.
- Dúvida sobre **como avaliar de forma justa** o Bayes manual junto com os algoritmos de ML (resolvido com um split único compartilhado).
- Dúvida conceitual sobre **por que a acurácia engana** quando a classe de interesse é minoritária.

## Como a equipe revisou e compreendeu o que foi gerado

- Todo trecho de código foi **lido, testado e executado** pela equipe; o notebook roda do início ao fim e regenera os modelos.
- As **decisões técnicas** (manter outliers, imputar pela mediana, usar `class_weight='balanced'`) foram discutidas e justificadas no relatório com base nos dados.
- Os **conceitos** (priori, verossimilhança, posteriori, Laplace) foram estudados e cada integrante é capaz de explicá-los na arguição (ver [`GUIA_ARGUICAO.md`](GUIA_ARGUICAO.md)).

## Declaração final

> A IA generativa foi utilizada **como apoio ao aprendizado, e não como substituição dele**. A equipe entende, sabe justificar e é capaz de reproduzir todas as etapas do projeto.
