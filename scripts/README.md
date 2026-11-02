# Pontos de entrada

- `build_inventory.py`: inventário e fingerprint do 3W Dataset;
- `audit_well_splits.py`: cobertura estrutural dos folds por poço;
- `run_baseline_smoke.py`: compatibilidade mínima com o baseline;
- `run_r0_preprocessing.py`: pré-processamento histórico completo;
- `audit_r0_artifacts.py`: integridade dos artefatos R0;
- `reproduce_r0_oof.py`: reprodução das métricas OOF;
- `run_r0_model.py`: ajuste isolado dos modelos do cenário R0;
- `build_original_real_features.py`: tabela retrospectiva real de E0/E1;
- `make_e0_e1_splits.py`: folds e cobertura efetivos de E0/E1;
- `compare_feature_tables.py`: comparação estrutural entre tabelas de features;
- `run_e0_e1_oof.py`: execução OOF dos cenários E0--E3;
- `run_e3_shap_pilot.py`: prova de conceito SHAP no cenário E3;
- `build_report_tables.py`: valida os runs congelados e gera as tabelas
  T001--T007, seus CSVs e os registros de proveniência.
- `build_report_figures.py`: valida o piloto SHAP e gera a F001 em PDF vetorial,
  PNG a 300 dpi e JSON de proveniência.

Os scripts atuam como pontos de entrada para funções testadas em `src/pfc3w`.
As rotinas de relatório consomem artefatos congelados; as rotinas OOF e de SHAP
correspondem à reprodução experimental de maior custo.
