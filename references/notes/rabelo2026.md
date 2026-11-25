# rabelo2026

## Relação com o projeto

Trabalho de referência direta para o pipeline de classificação no 3W Dataset. A
reprodução de seus resultados constitui a referência histórica R0, não uma
contribuição original deste PFC.

## Elementos utilizados

- seleção de oito sensores;
- janelas de 300 segundos e passo de 150 segundos;
- conjunto de 88 features estatísticas;
- comparação entre Random Forest, XGBoost, CNN 1D e CNN LSTM;
- agrupamento por `instance_id` na validação cruzada;
- resultados publicados e discussão de classes raras;
- aplicação de SHAP.

## Limitação relevante

O agrupamento por instância mantém juntas as janelas do mesmo episódio, mas não
impede que episódios distintos do mesmo poço físico pertençam a folds diferentes.
O próprio trabalho relaciona a escolha à pequena quantidade de poços e à
concentração de classes raras.

## Uso no PFC

Fonte do baseline, dos valores de referência e da justificativa para comparar
generalização por episódio e por poço físico.

## Evidência auditada

- resumo e Capítulo 4, páginas 58--67: pré-processamento, janelas, 88 features,
  modelos RF, XGBoost, CNN-1D e CNN-LSTM e protocolo de avaliação;
- capítulos de resultados: métricas, matrizes de confusão e análise SHAP;
- os detalhes executáveis de `instance_id` e agrupamento foram confrontados com
  o repositório associado, e não inferidos somente do texto do TG.
