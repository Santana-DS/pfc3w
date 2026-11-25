# Auditoria do baseline

Esta auditoria descreve o comportamento do código no commit
`49511b1caa82974410fb28892a49b54a13ca2f5f`. Seu objetivo é definir uma
reprodução fiel e separar essa reprodução das alterações metodológicas deste
projeto.

## Unidade de agrupamento

O carregador define `instance_id` como o nome-base do arquivo. Os scripts de
treino usam esse campo no `GroupKFold`. Portanto, janelas do mesmo arquivo são
mantidas no mesmo fold, mas arquivos diferentes do mesmo poço real podem aparecer
em treino e teste.

Além disso, nomes de arquivos simulados e desenhados se repetem entre classes.
Ao ignorar o diretório da classe, o agrupamento associa artificialmente arquivos
distintos que compartilham nomes como `SIMULATED_00001`.

Para E0, esse comportamento será mantido porque faz parte do baseline. Nos
experimentos novos, cada arquivo terá identificador único e os dados reais serão
agrupados pelo poço físico.

## Transformações temporais

O forward-fill é aplicado na direção do tempo, limitado a 60 amostras e separado
por instância. Esse componente é causal desde que a ordem temporal seja
preservada.

A normalização z-score, por outro lado, calcula média e desvio-padrão usando a
instância completa antes de extrair janelas. Uma janela inicial passa a depender
de valores observados posteriormente. Essa escolha é adequada para análise
retrospectiva da instância completa, mas não representa inferência on-line sem
uma etapa adicional de calibração.

O filtro gaussiano também é bilateral: amostras futuras contribuem para o valor
filtrado no instante atual. Os experimentos causais precisarão substituí-lo ou
removê-lo.

## Imputação e validação

Há três implementações relevantes, considerando o histórico do repositório:

- no classificador diagnóstico, a mediana é ajustada sobre todo o conjunto antes
  da validação cruzada;
- no commit que gerou as métricas publicadas dos classificadores de estado, a
  mediana também era ajustada sobre todo o conjunto antes da validação;
- na versão final dos classificadores de estado, a mediana é calculada usando
  todas as janelas de cada instância, inclusive quando ela pertence ao teste.

A versão global mistura estatísticas de treino e teste. A versão por instância
não mistura instâncias entre folds, mas permite que o pré-processamento de uma
previsão utilize a distribuição completa da própria instância de teste. Nenhuma
é compatível com uma implantação estritamente causal. Nos protocolos novos,
transformações aprendidas serão ajustadas apenas no treino ou atualizadas
somente com o passado disponível.

Os JSONs de métricas foram gerados antes da refatoração para mediana por
instância e não foram atualizados depois dela. Portanto, código final e métricas
publicadas precisam ser tratados como dois artefatos históricos distintos.

## O que será reproduzido e o que será alterado

| Componente | E0: reprodução | E1–E3: protocolo novo |
|---|---|---|
| Identidade de arquivo | nome-base legado | `classe/nome-base` |
| Unidade física | `instance_id` | `well_id` para dados reais |
| Normalização | instância completa | treino ou histórico passado |
| Filtro gaussiano | mantido | removido ou tornado causal |
| Imputação | comportamento original | ajustada sem acesso ao teste/futuro |
| Dados artificiais | composição original | intervenção reportada separadamente |

Essa separação impede que uma melhoria metodológica seja apresentada como se
fosse uma reprodução e permite atribuir cada diferença de resultado a uma
mudança controlada.

## Ensaio de compatibilidade

Foi executado o pipeline canônico com a primeira instância de cada classe, sem
editar o repositório do baseline. A execução produziu 4.911 janelas e 88 features
por janela a partir de 745.847 observações limpas.

Embora dez arquivos — um por classe — tenham sido processados, restaram apenas
quatro valores distintos de `legacy_instance_id`. O ensaio confirma, em uma
execução real, que a repetição de nomes entre classes chega à variável de
agrupamento usada pelos modelos. A execução também confirmou que o pipeline de
features permanece compatível com o ambiente atual.
