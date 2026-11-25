# grabelo2026flowassuranceml

## Natureza da fonte

Repositório público associado ao TG de Gabriel Rabelo. A proveniência de código
reutilizado será preservada por commit, caminho original e histórico de mudanças.

## Pontos de auditoria

- construção de `instance_id` a partir do nome do arquivo;
- ordem de limpeza, normalização, filtragem e janelamento;
- ajuste de transformações antes ou depois da criação dos folds;
- geração das 88 features;
- implementação dos grupos de validação;
- sementes, hiperparâmetros e artefatos de resultados;
- amostras e modelos usados nos gráficos SHAP.

## Uso no PFC

Base executável para E0 e referência de comparação para as alterações introduzidas
em E1, E2 e E3. Mudanças locais serão documentadas como autoria deste projeto.

## Evidência auditada

Commit local auditado: `49511b1caa82974410fb28892a49b54a13ca2f5f`.

- `config.py`: `WINDOW_SIZE = 300` e `STEP_SIZE = 150`;
- `src/data_loader.py`: `instance_id` derivado do nome do arquivo;
- `src/feature_engineering.py`: onze estatísticas por sensor;
- scripts de treinamento: `GroupKFold` aplicado aos grupos `instance_id`.
