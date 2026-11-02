# Base retrospectiva real de E0 e E1

## Extração completa

A execução `features-real-retrospective__1ca64ec90e` processou todas as
1.119 instâncias reais do 3W Dataset. O filtro de qualidade preservou 960
instâncias e 35 dos 40 poços físicos, produzindo 159.779 janelas com 88 features.

| Item | Resultado |
|---|---:|
| Arquivos reais examinados | 1.119 |
| Instâncias aprovadas | 960 |
| Instâncias descartadas | 159 |
| Poços antes da limpeza | 40 |
| Poços depois da limpeza | 35 |
| Janelas | 159.779 |
| Colunas de identidade/metadados | 9 |
| Features | 88 |

Os poços removidos integralmente foram `WELL-00008`, `WELL-00010`,
`WELL-00023`, `WELL-00026` e `WELL-00033`.

O SHA-256 da tabela de features é
`13cf7de19e6b11ea5a8f5ecd7ff37787454eb2fe3621c85b69fe8c0cc07d7043c`.

## Paridade com R0

As 159.779 janelas foram comparadas com todas as janelas de origem `WELL` do R0,
ordenadas por classe, identificador legado e início da janela. A comparação
confirmou:

- igualdade exata dos rótulos operacionais;
- igualdade das 88 features, incluindo posições ausentes;
- diferença numérica máxima igual a zero;
- uma única associação `instance_id`–`well_id`;
- ausência de duplicatas em `(instance_id, window_start)`.

Portanto, a nova camada não altera o processamento numérico do baseline. Ela
acrescenta identidade única e unidade física.

## Suporte físico dos estados

| Estado | Janelas | Instâncias | Poços |
|---:|---:|---:|---:|
| 0 | 83.971 | 668 | 35 |
| 1 | 61 | 4 | 3 |
| 2 | 121 | 8 | 5 |
| 3 | 3.777 | 32 | 2 |
| 4 | 12.238 | 260 | 6 |
| 5 | 83 | 6 | 3 |
| 6 | 83 | 3 | 1 |
| 7 | 197 | 5 | 4 |
| 8 | 889 | 11 | 8 |
| 9 | 115 | 3 | 3 |
| 101 | 435 | 4 | 3 |
| 102 | 569 | 21 | 6 |
| 105 | 2.131 | 11 | 3 |
| 106 | 44 | 6 | 2 |
| 107 | 35.358 | 32 | 5 |
| 108 | 18.415 | 11 | 8 |
| 109 | 1.292 | 13 | 6 |

O estado ativo 6 aparece em somente um poço depois da limpeza. Assim, não existe
evidência no 3W que permita simultaneamente treinar essa classe em um poço e
testá-la em outro. Essa classe não sustenta uma afirmação de generalização
física, independentemente do algoritmo utilizado.

## Folds congelados

E0 contém 960 grupos por `instance_id`, com aproximadamente 31,9 mil janelas por
fold. Todas as 17 classes estão presentes no treino dos cinco folds.

E1 contém 35 grupos por `well_id`, também com aproximadamente 31,9 mil janelas
por fold. O fold que testa o único poço com estado 6 não possui essa classe no
treino. Várias classes também estão ausentes de folds de teste, o que é esperado
dada sua concentração física.

As avaliações reportarão:

1. F1-macro sobre todas as 17 classes, incluindo zero quando uma classe não pode
   ser aprendida;
2. F1-macro sobre as classes fisicamente identificáveis, definidas como aquelas
   presentes em pelo menos dois poços;
3. F1 por classe, suporte em janelas e número de poços;
4. cobertura de treino e teste em cada fold;
5. métricas agrupadas por poço, além da consolidação OOF.

Essa política não elimina a classe 6 do resultado. Ela separa “falha do modelo”
de “ausência de informação no dataset”.
