# Thiago-TP — `flow-assurance-ml`

## Identificação

- Repositório: <https://github.com/Thiago-TP/flow-assurance-ml>
- Commit examinado: `f8b2484b1e6f156a46f34d6c480b438efa2fd22a`
- Relação: fork do repositório de Gabriel Rabelo, 52 commits à frente do commit
  `49511b1caa82974410fb28892a49b54a13ca2f5f`.
- Natureza da fonte: repositório de software em desenvolvimento, não revisado
  por pares.

## Pertinência

O fork implementa ou investiga vários temas diretamente relacionados ao PFC:

- agrupamento por `well_id` e avaliação leave-one-well-out;
- auditoria de composição dos splits;
- vazamento da normalização por instância completa;
- sobreposição temporal entre gravações do mesmo poço;
- sensores congelados e padrões de instrumentação;
- SHAP e árvores de decisão interpretáveis;
- proveniência de execuções e artefatos.

O repositório relata queda substancial de desempenho ao migrar da validação por
instância para poços não vistos e atribui parte do desempenho original à
identificação do poço. Esses resultados são relevantes, mas precisam ser
reproduzidos independentemente antes de serem usados como evidência do PFC.

## Relação com este estudo

- funciona como implementação relacionada e fonte de hipóteses verificáveis;
- permite comparar perguntas, protocolos e resultados;
- não constitui o baseline canônico, definido pelo commit publicado por
  Gabriel Rabelo;
- nenhum código do fork integra esta distribuição.

No commit examinado não foi localizado arquivo de licença. Por isso, o código é
considerado somente como fonte auditável para comparação, não como componente
reutilizável.

## Questões metodológicas abertas

O escopo examinado não consolida as seguintes questões:

- pré-processamento estritamente causal para cada instante de previsão, inclusive
  estatísticas acumuladas ou janelas de calibração somente passadas;
- estabilidade das explicações entre poços e protocolos, não apenas geração de
  gráficos SHAP;
- comparação controlada E0–E3 com atribuição do efeito de cada mudança;
- protocolo explícito para dados artificiais usados somente no treino;
- análise de incerteza e de cobertura de classes por unidade física.
