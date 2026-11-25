# Inventário e identidade dos dados

## Insumos congelados

| Item | Identificador |
|---|---|
| 3W Dataset | `2.0.0` |
| Repositório 3W | `4eaf746b2085ad76cade955545c33c67afe88bdf` |
| Repositório do baseline | `49511b1caa82974410fb28892a49b54a13ca2f5f` |
| Arquivos Parquet | 2.228 |
| Tamanho total | 1.873.131.359 bytes |
| Observações declaradas nos metadados Parquet | 76.587.318 |
| Impressão digital dos arquivos | `5e37d16f222c620698e22bf82d95450d3cc56ffd520be4446b4e555d078c0e65` |

A impressão digital é o SHA-256 da sequência ordenada formada pelo caminho
relativo e pelo SHA-256 de cada Parquet. Ela identifica o conteúdo efetivamente
utilizado, independentemente do nome da versão.

## Instâncias por classe e origem

| Classe | Real | Simulada | Desenhada | Total |
|---:|---:|---:|---:|---:|
| 0 | 594 | 0 | 0 | 594 |
| 1 | 4 | 114 | 10 | 128 |
| 2 | 22 | 16 | 0 | 38 |
| 3 | 32 | 74 | 0 | 106 |
| 4 | 343 | 0 | 0 | 343 |
| 5 | 11 | 439 | 0 | 450 |
| 6 | 6 | 215 | 0 | 221 |
| 7 | 36 | 0 | 10 | 46 |
| 8 | 14 | 81 | 0 | 95 |
| 9 | 57 | 150 | 0 | 207 |
| **Total** | **1.119** | **1.089** | **20** | **2.228** |

## Cobertura por poço físico

Os arquivos reais identificam 40 poços. A disponibilidade por classe é muito
desigual:

| Classe | Poços reais | Instâncias reais |
|---:|---:|---:|
| 0 | 9 | 594 |
| 1 | 3 | 4 |
| 2 | 7 | 22 |
| 3 | 2 | 32 |
| 4 | 7 | 343 |
| 5 | 3 | 11 |
| 6 | 2 | 6 |
| 7 | 6 | 36 |
| 8 | 9 | 14 |
| 9 | 15 | 57 |

Portanto, um `GroupKFold(5)` por poço não pode garantir que todas as classes
apareçam no teste de todos os folds. Classes 3 e 6 existem em somente dois poços;
classes 1 e 5, em três. A cobertura de classes deve ser reportada por fold e
resultados sem suporte de teste não podem ser interpretados como desempenho
estimado daquela classe.

## Contrato de identidade

Três identificadores têm papéis diferentes:

- `legacy_instance_id`: nome-base do arquivo, mantido apenas para reproduzir o
  baseline;
- `instance_id`: combinação `classe/nome-base`, única para cada arquivo;
- `well_id`: prefixo `WELL-XXXXX`, disponível somente para dados reais.

Dados simulados e desenhados não recebem `well_id`, pois o nome do arquivo não
fornece evidência de que representem um poço físico específico. A avaliação
principal em poços não vistos será calculada sobre dados reais. O uso de dados
artificiais no treino será tratado como uma intervenção separada.

Foram encontrados 225 valores repetidos de `legacy_instance_id`, afetando 885
registros. Por exemplo, `SIMULATED_00001` aparece em sete classes. Assim, o nome
do arquivo sem o diretório da classe não é um identificador globalmente único.

## Reprodução

O inventário é gerado por:

```powershell
python scripts/build_inventory.py --hash
```

Os arquivos detalhados são gravados em `results/inventory/`, diretório local
ignorado pelo Git. O script não carrega as séries completas: usa metadados
Parquet e leitura em blocos para os hashes.
