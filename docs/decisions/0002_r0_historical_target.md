# ADR 0002 — separar o baseline histórico do código final

## Contexto

As métricas publicadas pelo baseline foram geradas no commit `853b1cde`, quando
a imputação por mediana era ajustada globalmente antes da validação cruzada. Os
scripts foram alterados no commit `6a8275d1` para imputação por instância, mas os
JSONs de métricas e os valores do README não foram recalculados.

## Decisão

O projeto tratará dois alvos distintos:

1. **R0 histórico:** reproduz o procedimento que gerou as métricas publicadas,
   incluindo a imputação global;
2. **revisão final do baseline:** é auditada como variante posterior, mas não é
   usada para atribuir origem aos números publicados.

O R0 usa os melhores hiperparâmetros registrados e reproduz as predições OOF.
A busca aleatória completa não é repetida por padrão; quando omitida, essa
limitação deve ser declarada.

## Consequências

- métricas e código deixam de ser tratados como se pertencessem à mesma versão;
- a reprodução histórica mantém deliberadamente um vazamento de pré-processamento;
- experimentos E0–E3 partem de componentes explicitamente versionados;
- qualquer comparação futura registra estratégia de imputação e commit de origem.
