# Disponibilidade dos dados

Este repositório não redistribui o 3W Dataset nem suas tabelas intermediárias.
Os experimentos usam a versão 2.0.0 do conjunto publicado pela Petrobras, cuja
obtenção, licença e citação estão descritas no [repositório oficial do
3W](https://github.com/petrobras/3W) e em [`NOTICE`](../NOTICE).

Para reprodução, [`config/paths.example.yaml`](../config/paths.example.yaml)
documenta os caminhos esperados. O inventário da versão efetivamente usada,
incluindo origens, classes, volumes e identificação dos poços, está em
[`docs/data_inventory.md`](../docs/data_inventory.md).

Somente resumos derivados necessários à verificação das figuras e tabelas são
versionados em [`reports/`](../reports/). Dados brutos, tabelas de features,
previsões individuais e modelos treinados permanecem fora do repositório.
