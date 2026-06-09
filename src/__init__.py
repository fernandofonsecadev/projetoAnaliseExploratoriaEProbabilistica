"""Pacote de código compartilhado entre o notebook e o dashboard.

Contém:
- bayes.py       -> implementação manual do Teorema de Bayes (Naive Bayes categórico)
- preprocess.py  -> limpeza, discretização e helpers de inferência reutilizáveis

Manter a lógica aqui (em vez de só dentro do notebook) garante que os modelos
salvos com joblib possam ser recarregados pelo `app.py` sem o erro clássico
`AttributeError: Can't get attribute 'ImplementacaoBayesManual' on __main__`.
"""
