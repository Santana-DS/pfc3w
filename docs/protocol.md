# Protocolo experimental

## Unidade de generalização

O protocolo principal estima desempenho em **poços físicos não vistos**. O
`instance_id` identifica um episódio/arquivo; o `well_id` identifica o sistema
físico. Ambos são preservados porque respondem a perguntas diferentes.

## Reprodução e matriz de ablação

`R0` reproduz o baseline com sua população e identidade originais. Ele não faz
parte do fatorial E0–E3, pois misturar fontes artificiais com dados reais somente
em uma célula confundiria o efeito do protocolo com o efeito da população.
Seu alvo histórico é o código que gerou os JSONs publicados: imputação por
mediana global antes do `GroupKFold`. A revisão posterior, com mediana por
instância, é mantida como controle separado.

| Código | População | Separação | Pré-processamento | Papel |
|---|---|---|---|---|
| R0 | todas as origens | identidade legada | histórico publicado | reprodução fiel |
| E0 | real | `instance_id` | retrospectivo | controle do fatorial |
| E1 | real | `well_id` | retrospectivo | efeito da independência física |
| E2 | real | `instance_id` | causal | efeito do pacote temporal e da imputação |
| E3 | real | `well_id` | causal | transferência interpoços com features temporais causais |

O uso de dados simulados ou desenhados no treino de E3 constituirá uma
intervenção adicional, denominada A3, e não será incorporado silenciosamente ao
experimento principal.

Em E0 e E1, “retrospectivo” corresponde à revisão final do baseline: z-score e
mediana pela instância completa e filtro gaussiano bilateral. A imputação global
que gerou as métricas históricas fica restrita a R0. A definição completa está
em [ADR 0003](decisions/0003_original_preprocessing_for_e0_e1.md).

Em E2 e E3, “causal” corresponde a z-score expansivo, filtro gaussiano
unilateral e imputação estimada somente no fold de treino. A janela representa
uma decisão emitida depois de sua última amostra. A definição e seus testes de
invariância a prefixos estão em
[ADR 0004](decisions/0004_causal_preprocessing_for_e2_e3.md).

A seleção dos episódios mantém a regra do baseline: descarta arquivos com mais
de 50% de `P-TPT` ausente após preenchimento limitado, considerando a trajetória
completa. Ela preserva a mesma população em E0--E3, mas não é uma decisão de
admissão disponível a cada instante. A avaliação interpoços é condicional aos
episódios aceitos por essa regra. O alvo por moda descreve o estado predominante
nos 300 segundos da janela, e não mede o atraso de detecção do evento.

## Protocolo por poço

O piloto usa cinco folds por `well_id`, equilibrados pelo número estimado de
janelas. Essa estimativa é calculada antes dos filtros de qualidade; as
atribuições definitivas serão congeladas depois da extração das janelas.

Todas as classes aparecem no treino dos cinco folds estruturais, mas nenhum fold
de teste contém as dez classes. Portanto:

- F1 macro global será calculado sobre as previsões out-of-fold agregadas;
- métricas por fold declararão quais classes têm suporte;
- desempenho por classe será acompanhado do número de poços que a sustentam;
- resultados por poço serão reportados separadamente;
- leave-one-well-out poderá ser usado como análise complementar de
  heterogeneidade, condicionado à cobertura estatística e à viabilidade
  computacional.

Depois da limpeza efetiva, o estado operacional 6 permanece em apenas um poço e
fica ausente do treino de um fold E1. Além do F1-macro conservador sobre todas as
classes, será reportado F1-macro sobre estados presentes em pelo menos dois
poços. Ambos serão acompanhados por suporte e cobertura; o segundo não substitui
nem oculta o primeiro.

## Métricas mínimas

- F1 macro e por classe;
- precisão, recall e suporte por classe;
- acurácia balanceada;
- matriz de confusão;
- cobertura de classes em cada fold;
- distribuição de desempenho por poço e por origem dos dados.

Nenhuma média será publicada sem sua população, cobertura e unidade de
agregação. Janelas sobrepostas não serão tratadas como observações estatísticas
independentes para intervalos de confiança.

## XAI

SHAP é o método principal por ser compatível com os modelos de árvores do núcleo.
O protocolo inclui:

1. explicações globais por classe;
2. explicações locais de casos representativos e erros importantes;
3. agregação por sensor e família de feature;
4. estabilidade de rankings entre folds e poços;
5. comparação E0 × E1 e E0 × E2;
6. análise de plausibilidade física sem alegar causalidade.

Regras ou modelos substitutos somente serão incluídos depois do núcleo E0–E3,
caso acrescentem uma forma de validação compreensível por especialistas.

## Critério de conclusão do PFC1

- ambiente e hashes registrados;
- R0 reproduzido dentro de tolerância declarada;
- matriz classe × poço validada;
- protocolo causal especificado;
- piloto comparável entre E0 e E3 executado;
- limitações e plano do PFC2 documentados.
