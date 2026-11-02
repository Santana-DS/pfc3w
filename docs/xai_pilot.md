# Prova de conceito XAI no cenário E3

## O que está sendo explicado

A prova de conceito usa SHAP no Random Forest do fold 0 de E3, o protocolo
causal com poços fisicamente independentes. O modelo foi treinado com 127.808
janelas de 30 poços e testado em 31.971 janelas de cinco poços não vistos.

SHAP decompõe a saída do modelo em um valor de referência e contribuições de
cada feature. Para uma classe `c` e uma janela `x`, a interpretação usada é:

`probabilidade(c | x) ≈ valor-base(c) + soma das contribuições SHAP(c, x)`.

Uma contribuição positiva aumenta a saída da classe explicada em relação ao
valor-base; uma contribuição negativa a reduz. Isso descreve o comportamento
matemático do classificador. Não demonstra que alterar fisicamente um sensor
causaria a mudança prevista.

## Desenho do piloto

- até 25 janelas foram sorteadas sem reposição para cada estado verdadeiro
  presente no teste;
- a amostra total possui 269 janelas e 11 estados;
- a importância global é a média do valor absoluto da contribuição para a
  classe prevista em cada janela;
- as 88 features são agregadas também por sensor e por estatística;
- os casos locais incluem cinco erros e cinco acertos de alta confiança,
  priorizando instâncias distintas;
- o erro máximo da reconstrução aditiva foi `8,9 × 10⁻¹³`.

A amostragem estratificada impede que a classe normal domine a análise, mas
também significa que essas importâncias não representam a prevalência
operacional do dataset. Elas respondem quais entradas foram relevantes em uma
amostra deliberadamente balanceada para inspeção.

Das 269 janelas explicadas, 46 foram classificadas corretamente e 223
incorretamente. O ranking para a classe prevista reúne, assim, acertos e erros.
Ele descreve a utilização dos atributos pelo modelo nessa amostra; a causa dos
erros requer análise separada por correção, classe, poço e disponibilidade dos
sensores.

## Importância por sensor

| Sensor | Fração da importância SHAP |
|---|---:|
| `P-TPT` | 20,34% |
| `QGL` | 19,11% |
| `P-MON-CKP` | 13,13% |
| `T-TPT` | 12,46% |
| `T-JUS-CKP` | 11,88% |
| `P-PDG` | 8,45% |
| `P-JUS-CKGL` | 8,06% |
| `T-PDG` | 6,55% |

`P-TPT` e `QGL` somam 39,46% da importância na amostra. Isso mostra que o
modelo utilizou fortemente características da pressão em TPT e da vazão de gas
lift. A plausibilidade de cada relação e de seu sinal ainda deve ser avaliada
com conhecimento de garantia de escoamento; o ranking sozinho não valida uma
explicação física.

As features de `P-TPT` usam sinal previamente normalizado e filtrado; sua
importância não comprova uso de patamares absolutos de pressão. A importância de
`QGL` tampouco identifica, por si, um mecanismo de erro ou de imputação.

## Features mais influentes

| Feature | Fração da importância SHAP |
|---|---:|
| `P-TPT_mean` | 4,19% |
| `QGL_max` | 4,12% |
| `P-TPT_median` | 4,00% |
| `P-TPT_max` | 3,81% |
| `QGL_std` | 3,18% |
| `P-TPT_min` | 3,10% |
| `QGL_max_zscore` | 2,80% |
| `QGL_iqr` | 2,59% |
| `T-JUS-CKP_max` | 2,49% |
| `T-TPT_diff2_std` | 2,42% |

Média, extremos, dispersão e derivadas aparecem conjuntamente. Ao agregar por
família, `max`, `mean`, `median`, `diff2_std`, `std` e `diff1_std` respondem por
aproximadamente 73,3% da importância. Assimetria e curtose, juntas, ficam em
aproximadamente 2,6%.

## O que o piloto demonstra

1. o pipeline E3 é compatível com explicações SHAP globais e locais;
2. é possível voltar de uma atribuição às identidades `well_id`, `instance_id`
   e `window_start` da janela explicada;
3. a soma das atribuições reconstrói numericamente a saída do modelo;
4. sensores e famílias estatísticas podem ser apresentados em um nível mais
   inteligível que uma lista plana de 88 features;
5. erros de alta confiança podem ser selecionados para auditoria especializada.

## O que ainda não demonstra

- estabilidade do ranking nos cinco folds ou entre poços;
- concordância entre Random Forest e outros modelos;
- estabilidade diante de escolhas de amostragem e background;
- relação causal física entre sensores e estados operacionais;
- utilidade das explicações para decisão operacional;
- aprovação das relações por um especialista de domínio.

Esses itens pertencem ao protocolo ampliado do PFC2. No PFC1, o objetivo é
demonstrar viabilidade técnica, rastreabilidade e uma primeira interpretação sem
extrapolar a evidência.

## Rastreabilidade

Identificador científico da execução: `e3-rf-shap-fold0__904152b154`.

O run registra o commit `85e5144`, o fold, os hashes da tabela causal e das
atribuições de poço, a amostra explicada, versões de software, importâncias
globais, explicações locais e figura. Os artefatos volumosos permanecem em
`results/`, fora do Git; o script reproduzível é
`scripts/run_e3_shap_pilot.py`.
