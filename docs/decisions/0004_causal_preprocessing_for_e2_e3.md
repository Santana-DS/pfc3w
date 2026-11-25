# ADR 0004 — pré-processamento causal de E2 e E3

## Contexto

O pré-processamento retrospectivo de E0 e E1 consulta a trajetória completa da
instância para calcular média e desvio e usa um filtro gaussiano bilateral. Essa
informação não estaria disponível no instante de uma previsão on-line. E2 e E3
devem retirar essa antecipação sem alterar sensores, janelas ou famílias de
features, permitindo atribuir a diferença observada ao regime temporal.

## Decisão

E2 e E3 compartilharão uma tabela causal de features reais com:

- limpeza e forward-fill limitado a 60 s, que já operam na direção do tempo;
- normalização z-score expansiva por sensor e instância: no instante `t`, média
  e desvio populacional são calculados somente com observações entre `0` e `t`;
- valor normalizado igual a zero durante a inicialização, quando há menos de
  duas observações válidas ou desvio numericamente nulo;
- filtro gaussiano unilateral com `sigma = 2`, truncado em quatro sigmas, cujos
  pesos abrangem somente a amostra presente e até oito amostras passadas;
- NaNs preservados na normalização e no filtro;
- janelas deslizantes de 300 s, passo de 150 s e as mesmas onze estatísticas por
  sensor usadas no baseline;
- previsão conceitualmente emitida no final da janela; portanto, todos os 300
  pontos da janela já estão disponíveis;
- rotulagem pela moda da coluna operacional `class` dentro da janela, usada
  somente como alvo supervisionado;
- imputação de features por mediana calculada apenas nas janelas do fold de
  treino; colunas inteiramente ausentes no treino recebem zero em treino e teste.

A normalização é expansiva e adaptativa, em vez de ajustada no conjunto de
treino, porque sua finalidade é remover a linha de base operacional de cada
instância sem exigir conhecimento de sua trajetória futura. A transformação
produz uma representação disponível no instante de cada previsão e
independente do fold, para os episódios admitidos. A regra anterior de exclusão
por ausência de `P-TPT` continua a considerar o arquivo completo. Por isso, o
experimento não simula integralmente uma política de admissão on-line.

E2 agrupa por `instance_id`; E3 agrupa por `well_id`. As atribuições de fold já
congeladas para E0 e E1 serão reutilizadas se a limpeza causal produzir o mesmo
conjunto de janelas e identidades, o que mantém comparações pareadas.

## Verificação

Testes automáticos comparam a transformação de uma série completa e de seus
prefixos. Também alteram artificialmente o futuro de uma instância e exigem que
todas as features de janelas já encerradas permaneçam iguais. Esse teste de
invariância a prefixos justifica o termo “causal” para as transformações e as
features dos episódios admitidos. Ele não abrange a regra de exclusão por
ausência no arquivo completo nem demonstra causalidade física entre sensores e
falhas.

## Consequências

- E0 × E2 estima o efeito do pré-processamento temporal sob separação por
  instância;
- E1 × E3 estima o mesmo efeito sob independência física por poço;
- E2 × E3 estima o efeito da separação física mantendo o pipeline causal;
- os primeiros instantes têm estatísticas menos estáveis, uma limitação real de
  inicialização que deverá ser declarada e poderá ser estudada no PFC2;
- a referência expansiva depende do início registrado do episódio e pode
  incorporar progressivamente uma alteração persistente;
- a política de admissão deve ser avaliada separadamente quanto à disponibilidade
  temporal, cobertura de previsões e composição da população resultante;
- “causal” neste documento significa ausência de consulta ao futuro, não prova
  de relação causal física.
