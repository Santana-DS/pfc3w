# ADR 0003 — pré-processamento retrospectivo de E0 e E1

## Contexto

O R0 histórico usa imputação global antes da validação cruzada. Essa operação
mistura estatísticas de treino e teste e não deve ser carregada para a matriz
E0–E3. A versão final dos scripts do baseline passou a usar a trajetória completa
da própria instância, ainda de forma retrospectiva e não causal.

## Decisão

E0 e E1 compartilharão uma única tabela de features reais com:

- `instance_id` globalmente único no formato `classe/nome-do-arquivo`;
- `well_id` físico preservado em todas as janelas;
- forward-fill limitado a 60 s;
- descarte quando `P-TPT` possui mais de 50% de ausência;
- normalização z-score pela instância completa;
- filtro gaussiano bilateral;
- janelas de 300 s com passo de 150 s;
- rotulagem pela moda do estado operacional na janela;
- imputação posterior pela trajetória completa da instância, com zero quando a
  feature inteira estiver ausente.

E0 agrupará por `instance_id`; E1 agrupará por `well_id`. A tabela de entrada e o
modelo permanecem iguais para que a diferença E0–E1 estime somente o efeito da
separação física.

## Consequências

- a imputação global histórica permanece restrita ao R0;
- E0 e E1 não misturam estatísticas entre instâncias, mas continuam
  retrospectivos e não causais;
- E2 e E3 substituirão normalização, filtro e imputação por operações ajustadas
  no treino ou dependentes apenas do passado;
- uma única extração de features serve a E0 e E1, reduzindo custo e eliminando
  diferenças acidentais entre suas entradas.
