# Varma e Simon (2006)

## Contribuição da fonte

Demonstra por simulação que escolher hiperparâmetros e estimar o erro com a
mesma validação produz viés otimista, enquanto um laço externo separado reduz
substancialmente esse viés.

## Uso no PFC

Fundamenta a validação aninhada por poço planejada para eventual busca de
hiperparâmetros no PFC2. O agrupamento físico deverá ser preservado nos laços
interno e externo.

## Evidência auditada

Seções de métodos e resultados, páginas internas 3--7: simulações mostram viés
quando a mesma validação seleciona o modelo e estima seu erro; o laço externo
separado reduz esse viés.
