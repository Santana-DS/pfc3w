# Auditoria estrutural dos folds por poço

## Escopo

A auditoria usa somente as 1.119 instâncias reais e os 40 `well_id` encontrados
no inventário bruto. O número de janelas é uma estimativa estrutural com janela
de 300 amostras e passo 150, antes dos filtros de qualidade e da rotulagem por
estado operacional.

Foram estimadas 217.398 janelas. O `GroupKFold(5)` equilibrou os folds entre
43.454 e 43.517 janelas de teste.

## Cobertura

| Fold | Poços no teste | Janelas estimadas | Classes no teste | Classes ausentes no teste |
|---:|---:|---:|---|---|
| 0 | 6 | 43.454 | 0, 1, 2, 4, 5, 6, 9 | 3, 7, 8 |
| 1 | 8 | 43.459 | 0, 2, 7, 8, 9 | 1, 3, 4, 5, 6 |
| 2 | 8 | 43.468 | 0, 1, 2, 5, 7, 8, 9 | 3, 4, 6 |
| 3 | 9 | 43.517 | 0, 2, 3, 4, 6, 7, 8, 9 | 1, 5 |
| 4 | 9 | 43.500 | 0, 1, 2, 3, 4, 5, 7, 8, 9 | 6 |

Todas as classes estão presentes no treino de todos os folds. A ausência no
teste é consequência da concentração física dos eventos e não deve ser ocultada
por redistribuição de janelas do mesmo poço.

## Implicações para as métricas

Uma média macro calculada separadamente em cada fold depende de quais classes
estão presentes naquele fold. Por isso, ela não é diretamente comparável entre
folds sem um relatório de suporte. O protocolo adotará:

1. predições out-of-fold agregadas para a métrica global;
2. precisão, recall, F1 e suporte por classe;
3. desempenho por poço;
4. cobertura classe × poço;
5. distribuição dos resultados, sem tratar janelas sobrepostas como réplicas
   físicas independentes.

## Leave-one-well-out

O inventário bruto permite 40 folds leave-one-well-out. Após limpeza e remoção
de gravações inadequadas, esse número pode diminuir. O protocolo será usado para
examinar heterogeneidade entre poços; a busca completa de hiperparâmetros em cada
fold somente será executada se acrescentar informação proporcional ao custo.

## Reprodução

```powershell
python scripts/audit_well_splits.py
```

As atribuições e tabelas detalhadas são gravadas em `results/splits/`, ignorado
pelo Git.
