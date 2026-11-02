# Reprodução R0

O experimento R0 preserva deliberadamente o código, os identificadores e as
transformações do baseline de Gabriel Rabelo. Sua finalidade é verificar se o
ponto de partida pode ser reconstruído no ambiente atual antes de introduzir as
alterações metodológicas deste projeto.

## Insumos congelados

| Item | Identificação |
|---|---|
| Código do baseline | commit `49511b1caa82974410fb28892a49b54a13ca2f5f` |
| 3W Dataset | versão declarada `2.0.0` |
| Repositório do 3W | commit `4eaf746b2085ad76cade955545c33c67afe88bdf` |
| Fingerprint do inventário | `5e37d16f222c620698e22bf82d95450d3cc56ffd520be4446b4e555d078c0e65` |
| Execução | `r0-features-retrospective__ef5b392c24` |

O processamento manteve os oito sensores, janelas de 300 s, passo de 150 s,
normalização por instância e filtro gaussiano do baseline.

## Resultado do pré-processamento

| Verificação | Baseline documentado | Reprodução | Situação |
|---|---:|---:|---|
| Janelas | 449.397 | 449.397 | coincidência exata |
| Identificadores legados | 1.409 | 1.409 | coincidência exata |
| Features por janela | 88 | 88 | coincidência exata |
| Classes operacionais | 17 | 17 | coincidência exata |

As janelas reproduzidas estão distribuídas entre 269.862 observações simuladas,
159.779 reais e 19.756 desenhadas. O arquivo de features possui 93 colunas: cinco
metadados e 88 atributos estatísticos.

Os hashes SHA-256 dos artefatos locais são:

- `features_window_class.parquet`:
  `39c98d277fdc00fbddac58b6aaca054f8100c3f48ee079a505c28dbcd44b419b`;
- `cleaned.parquet`:
  `5566b1291d36e4a98cb9ff00c646f3f81151dc6e8c6dfe577e5c8f85bfc6f533`.

Os artefatos são produtos locais ignorados pelo Git. O script
`scripts/audit_r0_artifacts.py` recalcula contagens, integridade numérica e hashes
a partir do diretório de uma execução.

## Ausência estrutural de sensores

O conjunto de features não é uma matriz integralmente preenchida. Determinados
sensores não existem em parte das instâncias; nesses casos, todas as onze
estatísticas derivadas do sensor permanecem ausentes naquela janela.

| Sensor | Janelas ausentes | Proporção |
|---|---:|---:|
| `T-PDG` | 341.775 | 76,05% |
| `QGL` | 306.325 | 68,16% |
| `P-JUS-CKGL` | 285.820 | 63,60% |
| `T-JUS-CKP` | 61.294 | 13,64% |
| `T-TPT` | 45.912 | 10,22% |
| `P-MON-CKP` | 30.720 | 6,84% |
| `P-PDG` | 19.049 | 4,24% |
| `P-TPT` | 15 | <0,01% |

Isso não invalida a reprodução, mas exige uma etapa explícita de imputação antes
do classificador. A estratégia utilizada mudou durante o desenvolvimento do
baseline e essa mudança é relevante para interpretar as métricas.

## Duas versões do classificador no mesmo repositório

Os JSONs de métricas foram adicionados no commit `853b1cde`, de abril de 2026.
Naquela versão, `SimpleImputer(strategy="median")` era ajustado uma única vez
sobre todas as janelas, antes do `GroupKFold`. Portanto, a mediana incorporava o
conjunto de teste de cada fold.

No commit `6a8275d1`, de junho de 2026, os scripts foram refatorados para imputar
pela mediana da instância completa, com zero quando o sensor inteiro está
ausente. Os arquivos de métricas não foram regenerados: seus campos
`trained_at` permanecem em abril e não incluem o campo `filter` introduzido na
refatoração. Assim, a versão final do código e os números apresentados no README
não descrevem exatamente o mesmo pipeline.

Nenhuma das duas estratégias é estritamente causal. A primeira usa o teste
global; a segunda usa janelas futuras da própria instância de teste. Essa
cronologia passa a fazer parte do resultado de auditoria do R0.

## Reprodução das métricas publicadas

A busca aleatória completa foi iniciada para estimar seu custo. Uma dobra levou
cerca de dois minutos, projetando mais de três horas para 100 ajustes de Random
Forest. Como os melhores hiperparâmetros estão registrados nos JSONs originais,
a reprodução das métricas foi feita com esses parâmetros congelados e cinco
treinos OOF independentes. Isso reproduz a avaliação final; não constitui uma
nova execução da busca de hiperparâmetros nem verifica o campo `best_score_cv`.

| Modelo | Métrica | Publicada | Reprodução histórica | Diferença abs. |
|---|---|---:|---:|---:|
| Random Forest | F1-macro | 0,8716 | 0,8716 | 0,0000 |
| Random Forest | F1-weighted | 0,9308 | 0,9299 | 0,0009 |
| Random Forest | Acurácia | 0,9322 | 0,9312 | 0,0010 |
| XGBoost | F1-macro | 0,9082 | 0,9099 | 0,0017 |
| XGBoost | F1-weighted | 0,9361 | 0,9374 | 0,0013 |
| XGBoost | Acurácia | 0,9364 | 0,9376 | 0,0012 |

As diferenças máximas ficaram abaixo de 0,2 ponto percentual. Considerando que
o baseline especifica apenas versões mínimas das bibliotecas, sem arquivo de
lock, os resultados sustentam uma reprodução forte, embora não bit a bit.

Como controle, a mesma avaliação foi executada com a imputação por instância da
versão final dos scripts, sem refazer a busca de hiperparâmetros:

| Modelo | F1-macro | F1-weighted | Acurácia |
|---|---:|---:|---:|
| Random Forest | 0,8693 | 0,9284 | 0,9295 |
| XGBoost | 0,9022 | 0,9264 | 0,9271 |

Esses valores são um diagnóstico de sensibilidade à imputação, não uma
comparação definitiva entre modelos, pois os hiperparâmetros foram selecionados
sob o pipeline histórico.

As quatro execuções produziram 449.397 predições OOF. Uma verificação
independente confirmou que cada linha foi prevista uma vez, os cinco folds têm
aproximadamente 89,9 mil janelas e nenhum dos 1.409 `instance_id` legados cruza
folds.

## Conclusão do R0

O pré-processamento foi reproduzido com coincidência exata de contagens e as
métricas históricas foram reproduzidas com diferenças inferiores a 0,2 ponto
percentual. O R0 também revelou uma inconsistência de versionamento entre código
e resultados, além de três dependências não causais: imputação, normalização por
instância completa e filtro gaussiano bilateral. Esses achados justificam a
separação controlada entre reprodução histórica e experimentos E0–E3.
