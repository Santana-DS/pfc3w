# Turan e Jäschke (2021)

## Pergunta e método

Classificação multiclasse durante a fase transiente. O estudo compara métodos
lineares e baseados em árvores, usa dados reais e simulados, reserva 30% dos
episódios para teste e aplica validação cruzada em cinco folds no restante.

## Protocolo relevante

Todas as janelas de um evento são mantidas no mesmo conjunto. A normalização é
ajustada sem o fold retido. O estudo usa F1-macro e acurácia balanceada e
seleciona hiperparâmetros dentro da validação cruzada.

## Implicação

É um comparador metodológico importante porque reconhece o episódio como grupo,
mas não declara separação de todos os episódios de um mesmo poço. Assim, sua
pergunta se aproxima de E0/E2, não de E1/E3.

## Evidência auditada

Seções 3 e 4, páginas 158--161 dos anais: divisão de 30% dos episódios para
teste, validação em cinco folds no restante, preservação de todas as janelas do
mesmo evento no mesmo conjunto e comparação entre modelos lineares e de árvore.
