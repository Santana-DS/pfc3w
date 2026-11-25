# vargas2026threew

## Contribuição da fonte

Descrição primária do 3W Dataset 2.0.0, incluindo evolução em relação à versão
inicial, estrutura dos dados, origem das instâncias e procedimentos de validação
técnica adotados pelos responsáveis pelo dataset.

## Evidências relevantes

- versão pública 2.0.0;
- combinação de instâncias reais, simuladas e desenhadas;
- identificação persistente dos poços reais;
- validação das instâncias reais por especialistas;
- cenário físico de poços satélites offshore, instrumentação e posição das
  variáveis;
- distinção entre os rótulos `class` (normal, transiente e evento estabelecido)
  e `state` (configuração operacional do poço);
- descrição dos nove tipos de evento indesejável e seus códigos;
- licença CC BY 4.0;
- disponibilidade pública do dataset e de seus metadados.

## Divergência registrada

O artigo informa 42 poços reais considerados na versão 2.0.0. O inventário dos
1.119 arquivos Parquet no commit congelado identifica 40 valores distintos de
`WELL-XXXXX`; não há arquivos associados a `WELL-00017` ou `WELL-00018`. O
relatório distingue, por isso, a estatística publicada do conteúdo efetivamente
processado e identificado por hash.

## Uso no PFC

Fonte principal para a caracterização do dataset, do sistema físico, dos sinais,
dos rótulos, da interpretação de `well_id` e da separação dos resultados por
origem dos dados.

## Evidência auditada

- páginas 2--5, Tabelas 1--5: topologia, 27 variáveis, classes de evento e
  estados operacionais;
- página 11, seção de estrutura dos dados: convenção `WELL-XXXXX`, identidade
  persistente do poço real e múltiplas instâncias por poço;
- páginas 11--12, Tabela 6 e Figura 9: 1.119 instâncias reais, 1.089 simuladas,
  20 desenhadas e 42 poços descritos pelos autores.
