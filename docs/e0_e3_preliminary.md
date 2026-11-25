# Resultado preliminar da matriz E0–E3

## Pergunta respondida pelo piloto

O experimento separa dois fatores que frequentemente aparecem misturados em
classificação de séries temporais industriais:

1. **unidade de separação**: episódio (`instance_id`) ou poço físico (`well_id`);
2. **regime temporal**: pré-processamento retrospectivo ou causal.

As quatro células usam somente dados reais, 159.779 janelas, oito sensores, 88
features e os mesmos rótulos operacionais. A população foi aprovada por um
filtro de disponibilidade de `P-TPT` aplicado ao episódio completo; a
comparação é condicional a essa seleção retrospectiva. Random Forest e XGBoost usam seus
respectivos hiperparâmetros publicados no R0. E0/E2 compartilham folds por
episódio; E1/E3 compartilham folds por poço. Não houve nova busca de
hiperparâmetros.

| Experimento | Separação | Pré-processamento | Interpretação |
|---|---|---|---|
| E0 | episódio | retrospectivo | controle mais permissivo |
| E1 | poço | retrospectivo | efeito da independência física |
| E2 | episódio | causal | efeito temporal sem independência física |
| E3 | poço | causal | cenário principal de implantação |

## Resultados OOF

### Random Forest

| Métrica | E0 | E1 | E2 | E3 |
|---|---:|---:|---:|---:|
| F1-macro, 17 estados | 0,3791 | 0,1146 | 0,4409 | 0,1865 |
| F1-macro, 16 estados com ≥2 poços | 0,4028 | 0,1218 | 0,4618 | 0,1981 |
| F1 ponderado pelo suporte | 0,8793 | 0,4019 | 0,8950 | 0,4542 |
| Acurácia balanceada | 0,3716 | 0,1269 | 0,4229 | 0,1738 |
| Acurácia | 0,8848 | 0,4497 | 0,8970 | 0,4823 |

### XGBoost

| Métrica | E0 | E1 | E2 | E3 |
|---|---:|---:|---:|---:|
| F1-macro, 17 estados | 0,3929 | 0,1297 | 0,4722 | 0,1954 |
| F1-macro, 16 estados com ≥2 poços | 0,4175 | 0,1378 | 0,4969 | 0,2076 |
| F1 ponderado pelo suporte | 0,8665 | 0,3575 | 0,8812 | 0,4676 |
| Acurácia balanceada | 0,3810 | 0,1508 | 0,4718 | 0,2105 |
| Acurácia | 0,8707 | 0,3910 | 0,8786 | 0,5104 |

As métricas são calculadas sobre todas as previsões out-of-fold agregadas; não
são a média simples das métricas dos cinco folds. Isso evita atribuir o mesmo
peso a folds com populações e coberturas de classe diferentes.

## Contrastes controlados

| Contraste | Fator alterado | Δ RF | Δ XGBoost |
|---|---|---:|---:|
| E1 − E0 | episódio → poço, retrospectivo | −0,2645 | −0,2632 |
| E3 − E2 | episódio → poço, causal | −0,2544 | −0,2768 |
| E2 − E0 | retrospectivo → causal, por episódio | +0,0618 | +0,0793 |
| E3 − E1 | retrospectivo → causal, por poço | +0,0719 | +0,0657 |

Nos dois regimes temporais, impedir que o mesmo poço apareça em treino e teste
provoca a maior perda. Portanto, a correlação física é o fator dominante do
otimismo observado na validação por episódio. O pré-processamento causal elevou
o desempenho dos dois modelos nos dois regimes de separação, mas não compensou
a dificuldade de generalizar para um poço desconhecido.

XGBoost supera Random Forest em F1-macro por 0,0138 em E0, 0,0151 em E1,
0,0313 em E2 e 0,0089 em E3. A diferença entre modelos é muito menor que a
diferença entre protocolos de separação. Trocar o algoritmo, portanto, não
remove o principal gargalo experimental.

Esse ganho não autoriza afirmar que “causal sempre é melhor”. A normalização
expansiva e o filtro unilateral alteram a representação e podem funcionar como
regularização mais adequada ao modelo. O resultado vale para este dataset,
estes folds e estes hiperparâmetros e precisa de confirmação no PFC2.

## F1 por estado operacional

### Random Forest

| Estado | E0 | E1 | E2 | E3 |
|---:|---:|---:|---:|---:|
| 0 | 0,9062 | 0,6068 | 0,9179 | 0,6550 |
| 1 | 0,0000 | 0,0000 | 0,0000 | 0,0000 |
| 2 | 0,2360 | 0,0000 | 0,3736 | 0,0000 |
| 3 | 0,9069 | 0,0000 | 0,9309 | 0,0000 |
| 4 | 0,8873 | 0,0177 | 0,8526 | 0,1284 |
| 5 | 0,0000 | 0,0000 | 0,0000 | 0,0000 |
| 6 | 0,0000 | 0,0000 | 0,1053 | 0,0000 |
| 7 | 0,0000 | 0,0000 | 0,0000 | 0,0000 |
| 8 | 0,2344 | 0,2313 | 0,1513 | 0,0082 |
| 9 | 0,0000 | 0,0317 | 0,0000 | 0,0000 |
| 101 | 0,0000 | 0,0000 | 0,3311 | 0,7516 |
| 102 | 0,6811 | 0,2273 | 0,7850 | 0,5155 |
| 105 | 0,1732 | 0,0000 | 0,4997 | 0,0000 |
| 106 | 0,0968 | 0,0000 | 0,0706 | 0,0000 |
| 107 | 0,9003 | 0,0108 | 0,9333 | 0,0169 |
| 108 | 0,9020 | 0,6573 | 0,8840 | 0,7808 |
| 109 | 0,5208 | 0,1655 | 0,6596 | 0,3140 |

### XGBoost

| Estado | E0 | E1 | E2 | E3 |
|---:|---:|---:|---:|---:|
| 0 | 0,8976 | 0,5465 | 0,9157 | 0,7018 |
| 1 | 0,0000 | 0,0000 | 0,0000 | 0,0000 |
| 2 | 0,2091 | 0,0411 | 0,5000 | 0,0303 |
| 3 | 0,9175 | 0,0000 | 0,9306 | 0,0000 |
| 4 | 0,8927 | 0,0190 | 0,8539 | 0,1041 |
| 5 | 0,0000 | 0,0000 | 0,1309 | 0,0000 |
| 6 | 0,0000 | 0,0000 | 0,0761 | 0,0000 |
| 7 | 0,0000 | 0,0000 | 0,0084 | 0,0339 |
| 8 | 0,2269 | 0,2481 | 0,1836 | 0,1485 |
| 9 | 0,0000 | 0,3418 | 0,0000 | 0,0000 |
| 101 | 0,0501 | 0,0864 | 0,7103 | 0,6829 |
| 102 | 0,8018 | 0,2190 | 0,8435 | 0,5635 |
| 105 | 0,2911 | 0,0009 | 0,4742 | 0,0000 |
| 106 | 0,1944 | 0,0000 | 0,0822 | 0,0000 |
| 107 | 0,9016 | 0,0466 | 0,9326 | 0,0550 |
| 108 | 0,8061 | 0,4713 | 0,7669 | 0,6133 |
| 109 | 0,4907 | 0,1835 | 0,6178 | 0,3886 |

O resultado é heterogêneo: ambos os modelos preservam desempenho relevante em
parte dos estados 101, 102, 108 e 109, mas continuam falhando em várias classes
raras. Os estados raros não podem ser avaliados somente por número de janelas,
pois sua sustentação física é pequena. O estado 6, por exemplo, ocorre em um
único poço; quando esse poço é teste, não existe exemplo da classe no treino.
E1 e E3 registram exatamente 83 janelas nessa condição.

## Heterogeneidade por poço

| Modelo | Experimento | F1 médio por poço | F1 mediano por poço | Acurácia média por poço | Acurácia mediana por poço |
|---|---|---:|---:|---:|---:|
| RF | E0 | 0,5865 | 0,5998 | 0,8237 | 0,8923 |
| RF | E1 | 0,4054 | 0,3529 | 0,5830 | 0,6200 |
| RF | E2 | 0,6171 | 0,6117 | 0,8402 | 0,8927 |
| RF | E3 | 0,4387 | 0,3673 | 0,6331 | 0,6800 |
| XGBoost | E0 | 0,5915 | 0,6186 | 0,8093 | 0,8835 |
| XGBoost | E1 | 0,4187 | 0,3913 | 0,5574 | 0,5708 |
| XGBoost | E2 | 0,6365 | 0,6287 | 0,8248 | 0,9033 |
| XGBoost | E3 | 0,4756 | 0,4793 | 0,6431 | 0,6800 |

Essas médias tratam cada poço com o mesmo peso e complementam as métricas por
janela. Ainda existe ampla variação: em E3, a acurácia individual vai de 0,0027
a 1,0000 no RF e de 0,0301 a 0,9949 no XGBoost. Poços normais e poços
multiclasse não têm a mesma dificuldade.

## Integridade e rastreabilidade

- nenhum `instance_id` cruza folds em E0/E2;
- nenhum `well_id` cruza folds em E1/E3;
- todas as 159.779 janelas recebem uma única previsão OOF;
- E0/E2 e E1/E3 reutilizam os mesmos arquivos de atribuição de folds;
- imputação causal é reajustada somente no treino de cada fold;
- E2/E3 usam a tabela causal de hash
  `8290e337fd2eacdf832ce2e65f09e256e7fe0753dba2efc7946844cb9207f7c8`.

Identificadores científicos das execuções:

- `e0-rf-retrospective-oof__c7a61d9f00`;
- `e1-rf-retrospective-oof__c0383f2b87`;
- `e2-rf-causal-oof__af2b9a92a9`;
- `e3-rf-causal-oof__619b3c8c1d`;
- `e0-xgboost-retrospective-oof__25d2e05d86`;
- `e1-xgboost-retrospective-oof__25ed36d758`;
- `e2-xgboost-causal-oof__be63c3106d`;
- `e3-xgboost-causal-oof__453a98252e`.

## Limites da evidência

Este é o piloto substantivo do PFC1, não o resultado final do PFC2:

1. foram avaliados dois ensembles de árvores, mas nenhuma família de modelo com
   hipótese representacional substancialmente diferente;
2. os hiperparâmetros vêm do R0 e não foram ajustados dentro de um protocolo
   aninhado por poço;
3. as classes têm forte desbalanceamento e suporte físico desigual;
4. janelas sobrepostas não são observações estatisticamente independentes;
5. ainda não há intervalos de incerteza ou análise de sensibilidade; a prova de
   conceito XAI cobre somente um fold de E3 com Random Forest;
6. o protocolo mede generalização no conjunto de 35 poços disponíveis, não em
   toda operação petrolífera possível;
7. a admissão dos episódios consulta o arquivo completo, embora as
   transformações causais sejam invariantes a mudanças futuras nos episódios
   admitidos;
8. o rótulo por moda representa o estado predominante nos cinco minutos da
   janela, sem medir o atraso até a detecção de um evento.

A conclusão defensável nesta etapa é: **o protocolo por episódio produz uma
estimativa muito mais otimista que o protocolo por poço; tornar o
pré-processamento causal melhora Random Forest e XGBoost, mas a generalização
física continua sendo o principal gargalo.**
