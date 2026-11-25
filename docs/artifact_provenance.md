# Proveniência de tabelas e figuras

Toda tabela ou figura candidata ao relatório recebe um identificador estável no
arquivo `reports/artifact_manifest.csv`. O manifesto é a ligação entre o objeto
mostrado ao leitor e os dados, a execução e o código que o produziram.

## Classes de origem

- `original`: produzida neste projeto a partir de resultados rastreados;
- `adapted`: redesenhada a partir de uma fonte, com citação e descrição das
  alterações;
- `reproduced`: reconstruída a partir de dados ou método publicado;
- `borrowed`: reutilizada sem alteração analítica, somente quando a licença e a
  citação permitem.

Figuras de terceiros não são copiadas por conveniência. Diagramas conceituais
devem ser preferencialmente redesenhados e marcados como adaptados. Resultados
numéricos do projeto devem ser gerados por código, nunca ajustados manualmente
em editor gráfico.

## Registro mínimo

O manifesto registra: rótulo no LaTeX, descrição, classe de origem, `run_id`,
arquivo de entrada, código gerador, commit, formato de publicação e estado da
verificação. Um artefato só pode ser promovido a `verified` depois de:

1. ser regenerado a partir dos arquivos declarados;
2. ter valores e população conferidos contra o manifesto da execução;
3. possuir eixos, unidades, legenda e definição da incerteza quando aplicáveis;
4. passar por inspeção visual no PDF compilado.

## Padrão visual

- paleta acessível a daltonismo e também distinguível em escala de cinza;
- mesma cor para o mesmo experimento em todas as figuras;
- tipografia compatível com o relatório e tamanho legível na largura final;
- exportação vetorial (`PDF` ou `SVG`) para o relatório e `PNG` a 300 dpi para
  apresentações;
- títulos curtos; população, protocolo e incerteza explicados na legenda.

O manifesto contém somente proveniência técnica e pode ser versionado.

## Regeneração das tabelas E0--E3

As tabelas T004, T005 e T006 são declaradas na configuração pública
`config/reporting/public/results.yaml` e regeneradas com:

```powershell
python scripts/build_report_tables.py
```

O comando lê os `metrics.json` e `run_manifest.json` dos oito runs congelados,
confere a identidade do experimento e do modelo, o estado da execução, a
população avaliada e a igualdade das métricas nos dois arquivos. Para T006,
também verifica os 35 identificadores físicos, o suporte de janelas e rótulos e
calcula média e mediana sem ponderação pelo número de janelas. Os CSVs e os
fragmentos LaTeX são gravados em `reports/tables/`; os hashes das entradas, da
configuração e das saídas ficam em
`reports/tables/tab_e0_e3_provenance.json`. Quando o manuscrito local está
presente, os fragmentos LaTeX também são copiados para sua pasta `tables/`.

O mesmo comando produz T001 a partir das configurações experimentais, T002 a
partir dos runs históricos R0, T003 a partir da tabela causal de features e T007 a
partir do piloto SHAP. A proveniência desses conjuntos fica em
`reports/tables/tab_method_r0_provenance.json` e
`reports/tables/tab_support_shap_provenance.json`. T007 descreve somente o fold e
a amostra declarados no manifesto; a automação não amplia sua interpretação para
outros folds ou modelos.

## Regeneração da figura F001

A figura de importância global do piloto SHAP é regenerada com:

```powershell
python scripts/build_report_figures.py
```

O gerador confere a identidade do run E3, o fold, a unicidade das features e a
soma unitária das frações de importância. A saída seleciona as 15 features mais
influentes sem repetir a agregação por sensor de T007. O PDF vetorial e o PNG a
300 dpi ficam em `reports/figures/`; os hashes das entradas e saídas ficam em
`reports/figures/fig_f001_shap_global_e3_provenance.json`.
