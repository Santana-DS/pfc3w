# lundberg2017shap

## Contribuição da fonte

Apresenta SHAP como estrutura de atribuição aditiva de importância para explicar
previsões individuais de modelos.

## Uso no PFC

- explicações locais de janelas representativas;
- agregação de valores absolutos para análises globais por classe;
- comparação de rankings de features entre protocolos e folds;
- discussão da estabilidade das explicações.

## Limite interpretativo adotado

Uma atribuição SHAP descreve como o modelo utilizou suas entradas em relação a um
valor de referência. Não demonstra que a feature seja causa física do evento nem
substitui a validação do protocolo experimental.

## Evidência auditada

Seções 2 e 3, páginas 4766--4770 dos anais: classe de modelos aditivos de
atribuição, valor-base esperado e contribuições locais por feature.
