# Resultado preliminar E0 × E1

> Este documento registra o primeiro contraste intermediário. A matriz causal
> completa e a interpretação atual estão em
> [e0_e3_preliminary.md](e0_e3_preliminary.md).

## Controle experimental

E0 e E1 utilizaram a mesma tabela de 159.779 janelas reais, as mesmas 88
features, a mesma imputação por instância, o mesmo Random Forest e os mesmos
hiperparâmetros publicados no R0. A única intervenção foi a unidade de separação:

- E0: 960 episódios independentes;
- E1: 35 poços físicos independentes.

Os hiperparâmetros não foram reajustados para dados reais. Essa escolha mantém o
primeiro contraste controlado e reduz custo, mas caracteriza o resultado como
preliminar.

## Resultados OOF

| Métrica | E0 — episódio | E1 — poço | Diferença E1 − E0 |
|---|---:|---:|---:|
| F1-macro, 17 classes | 0,3791 | 0,1146 | −0,2645 |
| F1-macro, 16 classes fisicamente identificáveis | 0,4028 | 0,1218 | −0,2810 |
| F1-weighted | 0,8793 | 0,4019 | −0,4774 |
| Acurácia balanceada | 0,3716 | 0,1269 | −0,2447 |
| Acurácia | 0,8848 | 0,4497 | −0,4351 |

O valor alto de acurácia em E0 coexistindo com F1-macro baixo é consequência do
desbalanceamento. Estados raros já são difíceis mesmo quando episódios do mesmo
poço podem aparecer em treino e teste.

## Comportamento por classe

| Estado | Poços | F1 E0 | F1 E1 |
|---:|---:|---:|---:|
| 0 | 35 | 0,9062 | 0,6068 |
| 2 | 5 | 0,2360 | 0,0000 |
| 3 | 2 | 0,9069 | 0,0000 |
| 4 | 6 | 0,8873 | 0,0177 |
| 8 | 8 | 0,2344 | 0,2313 |
| 102 | 6 | 0,6811 | 0,2273 |
| 107 | 5 | 0,9003 | 0,0108 |
| 108 | 8 | 0,9020 | 0,6573 |
| 109 | 6 | 0,5208 | 0,1655 |

Os estados 1, 5, 6, 7 e 101 tiveram F1 igual a zero já em E0. Em E1, nove dos
17 estados tiveram F1 zero. O estado 6 possui 83 janelas de um único poço e ficou
ausente do treino do fold que contém essas janelas; as 83 previsões foram
marcadas explicitamente como classe não vista no treino.

## Heterogeneidade entre poços

A acurácia por poço em E1 variou de zero a um. Poços contendo somente condição
normal podem atingir desempenho perfeito, enquanto alguns poços multiclasse
tiveram desempenho próximo de zero. Essa amplitude mostra que métricas agregadas
por janela não descrevem sozinhas a capacidade de generalização física.

## Verificações de integridade

- as sequências de rótulos de E0 e E1 são idênticas;
- nenhuma instância cruza folds em E0;
- nenhum poço cruza folds em E1;
- todas as janelas receberam exatamente uma previsão OOF;
- E1 registrou exatamente 83 janelas cuja classe não existia no respectivo
  conjunto de treino, todas do estado 6.

## Interpretação limitada

O contraste fornece evidência preliminar forte de que a correlação entre
episódios do mesmo poço infla o desempenho estimado. Ainda não é a conclusão
final do trabalho porque:

1. somente o Random Forest foi executado;
2. os hiperparâmetros vieram do R0, que mistura origens;
3. o pré-processamento continua retrospectivo e não causal;
4. os folds não podem criar suporte físico inexistente para classes raras;
5. E2 e E3 ainda são necessários para separar o efeito da causalidade.

Identificadores científicos das execuções:

- `e0-rf-retrospective-oof__c7a61d9f00`;
- `e1-rf-retrospective-oof__c0383f2b87`.
