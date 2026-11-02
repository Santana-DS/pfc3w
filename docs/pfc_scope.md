# Escopo científico e fases experimentais

Este documento delimita a progressão metodológica entre a avaliação preliminar
e a avaliação extensiva. O escopo é orientado pelas perguntas de pesquisa e
pelas evidências necessárias, não por uma lista cronológica de atividades.

## Fase I — validação da hipótese e do protocolo

A primeira fase estabelece se a pergunta científica é relevante e se o
protocolo proposto pode ser executado de forma auditável. Ela compreende:

1. fundamentação técnica sobre garantia de escoamento, classificação de estados
   operacionais, validação agrupada, pré-processamento temporal e XAI;
2. registro da proveniência do 3W Dataset, do baseline e das referências;
3. reprodução R0, incluindo a análise de divergências entre código e métricas;
4. definição das unidades físicas, dos riscos de vazamento e do critério de
   causalidade computacional;
5. implementação testável da matriz E0--E3;
6. avaliação preliminar com Random Forest e XGBoost, acompanhada de métricas por
   classe, suporte e cobertura dos folds;
7. prova de conceito de explicabilidade em ao menos um cenário E3;
8. documentação das limitações e das hipóteses que exigem avaliação extensiva.

Os experimentos preliminares podem empregar configuração computacional reduzida,
desde que preservem a população, as unidades de separação e as regras de
causalidade declaradas. Seus resultados caracterizam viabilidade e efeito
preliminar; não substituem a análise extensiva.

## Fase II — avaliação extensiva e consolidação

A segunda fase amplia a evidência necessária para responder às perguntas de
pesquisa. Ela compreende:

1. experimentos completos E0--E3 e intervenções adicionais explicitamente
   definidas;
2. análise de sensibilidade e variabilidade entre folds, poços e classes;
3. avaliação de uma política de admissão temporal e da sensibilidade à
   inicialização da normalização;
4. comparação de modelos e ajuste de hiperparâmetros com protocolo sem
   vazamento;
5. XAI global e local, incluindo estabilidade das explicações entre poços e
   protocolos;
6. análise de erros, incerteza, cobertura e, quando viável, atraso de detecção;
7. avaliação de plausibilidade das explicações por conhecimento especializado,
   quando aplicável;
8. consolidação da discussão e das conclusões.

## Fronteira de interpretação

A Fase I demonstra a relevância da pergunta, a consistência da implementação e
a viabilidade do protocolo. A Fase II fornece cobertura experimental e análise
de incerteza suficientes para conclusões mais abrangentes. Resultados da Fase I
devem permanecer identificados como preliminares quando essas análises ainda não
tiverem sido executadas.
