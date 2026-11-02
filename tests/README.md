# Garantias verificadas automaticamente

1. nenhum `instance_id` cruza treino e teste;
2. nenhum `well_id` cruza treino e teste em E1/E3;
3. estatísticas de treino não são recalculadas no teste;
4. transformações causais não acessam amostras posteriores ao instante de decisão;
5. sementes reproduzem os mesmos grupos;
6. a cobertura por classe acompanha cada métrica;
7. contagens do pipeline reproduzem o baseline dentro da tolerância registrada.
