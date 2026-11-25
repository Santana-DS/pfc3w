# Marins et al. (2021)

## Pergunta e método

Detecção e classificação de sete eventos indesejáveis e normalidade no 3W por
meio de Random Forest e atributos estatísticos. O artigo organiza três
configurações: detecção binária, detectores por falha e classificação
multiclasse.

## Resultado usado

Na configuração multiclasse, o estudo reporta acurácia global de 94%. A
avaliação no nível de eventos reais é apresentada separadamente, assim como a
antecedência média da detecção durante o transiente. Esses resultados
caracterizam a literatura, mas não são comparados diretamente ao F1-macro
interpoços do PFC.

## Evidência auditada

Seções 4.1--4.3 e Tabelas 6--8: cenários binário, detectores específicos por
falha e classificação multiclasse; discussão final: avaliação por eventos reais
e durante o período transiente. A redação anterior, que associava diretamente
os 94% à detecção de eventos reais, foi corrigida.

## Implicação

É o antecedente supervisionado central do 3W e fundamenta o uso de features
estatísticas e árvores. A unidade física de generalização não é o foco da
contribuição descrita no resumo e, por isso, o número publicado não substitui o
protocolo E1/E3.
