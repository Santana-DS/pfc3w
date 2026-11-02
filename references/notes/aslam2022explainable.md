# Aslam et al. (2022)

## Pergunta e método

Detecção binária de normalidade contra evento no 3W. Compara Random Forest,
árvore de decisão, KNN e regressão logística, com dados originais e SMOTE.

## Protocolo relevante

O artigo descreve divisão estratificada aleatória 70/30 e validação cruzada
estratificada em dez folds para ajuste. Não declara bloqueio por episódio ou
poço. Emprega modelo substituto global, SHAP e LIME para interpretar o Random
Forest.

## Implicação

É antecedente direto de XAI no 3W, mas responde a uma tarefa binária e usa uma
unidade de separação distinta. O PFC acrescenta rastreabilidade a poço,
episódio e janela e condiciona a explicação ao cenário interpoços.

## Evidência auditada

Seções de metodologia e resultados: tarefa binária normal/evento, divisão
estratificada aleatória 70/30, comparação de RF, DT, KNN e regressão logística,
e explicações por modelo substituto global, SHAP e LIME.
