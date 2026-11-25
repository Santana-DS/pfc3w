# Base causal real de E2 e E3

## Finalidade

E2 e E3 usam a mesma população real, os mesmos sensores, rótulos, tamanhos de
janela e estatísticas de E0 e E1. A intervenção experimental é somente temporal:
normalização e filtragem deixam de consultar a continuação futura da instância.
A especificação completa está no [ADR 0004](decisions/0004_causal_preprocessing_for_e2_e3.md).

## Piloto de integração

A execução `features-real-causal-pilot__840f8853a1` selecionou uma
instância real válida por classe, segundo a mesma regra do piloto retrospectivo
`features-real-retrospective-pilot__9f3bbb3e34`.

| Verificação | Resultado |
|---|---:|
| Instâncias processadas | 10 |
| Poços físicos | 5 |
| Janelas | 3.567 |
| Features por janela | 88 |
| Metadados por janela | 9 |
| Metadados iguais ao controle retrospectivo | sim |
| Padrão de valores ausentes igual ao controle | sim |
| Células finitas de features alteradas | 236.844 |
| Diferença absoluta média nas células comparáveis | 0,3550 |

As 3.567 linhas possuem exatamente os mesmos `instance_id`, `well_id`,
`window_start`, `window_label` e demais metadados do piloto retrospectivo. A
igualdade também inclui a posição dos valores ausentes. Logo, as diferenças
numéricas não decorrem de seleção de arquivos, descarte de janelas ou alteração
de rótulos.

O valor máximo de diferença absoluta entre células foi 254,2505. Esse número
isolado não é uma medida de desempenho nem deve ser interpretado fisicamente:
as 88 colunas têm escalas e significados distintos. Sua utilidade aqui é apenas
confirmar que a intervenção causal não é uma operação nula.

## Teste da propriedade causal

A suíte automatizada verifica duas formas equivalentes de ausência de consulta
ao futuro:

1. transformar a série completa ou somente um prefixo produz o mesmo resultado
   dentro desse prefixo;
2. modificar drasticamente as amostras após um instante de corte não altera as
   features de janelas encerradas antes do corte.

Esses testes cobrem normalização expansiva, filtro gaussiano unilateral e a
extração das janelas dos episódios admitidos. A regra de descarte por ausência
de `P-TPT` consulta o episódio completo e não é coberta por eles. O termo causal
continua restrito às transformações computacionais/temporais, sem alegação de
causalidade física ou de admissão inteiramente on-line.

## Validação da base completa

A execução completa `features-real-causal__bb8b5f6f66` confirmou:

| Verificação | Resultado |
|---|---:|
| Arquivos examinados | 1.119 |
| Instâncias processadas | 960 |
| Instâncias descartadas | 159 |
| Poços físicos | 35 |
| Janelas | 159.779 |
| Metadados iguais à base retrospectiva | sim |
| Hash SHA-256 da tabela causal | `8290e337...9207f7c8` |

Esquema, ordem, identidades, rótulos e 159.779 janelas coincidem exatamente com
a base retrospectiva. Assim, os folds congelados de E0 e E1 podem ser
reutilizados respectivamente em E2 e E3.

### Diferença de ausência em sensores constantes

A máscara de NaNs das **features** não é integralmente igual. Há 68.442 células
em que o controle retrospectivo possui zero finito e a versão causal possui
NaN; não há nenhuma célula na direção inversa. Elas correspondem a 6.222
combinações janela–sensor e, como cada sensor gera onze features, totalizam
68.299 células em `P-PDG`/`T-PDG` e 143 células em `QGL`.

A inspeção dos dados brutos confirmou que esses sensores têm um único valor
distinto, portanto desvio zero, mas contêm lacunas. A normalização original
atribui zero à coluna inteira quando detecta constância e, com isso, também
substitui inadvertidamente amostras ausentes. A transformação causal atribui
zero somente às observações disponíveis e preserva a ausência. Essa diferença
é deliberadamente mantida porque preencher a lacuna com base na constância da
trajetória completa consultaria o futuro. As features ausentes serão tratadas
pela mediana do fold de treino, com fallback zero para coluna inteiramente
ausente no treino.

Entre as 10.972.885 células finitas comparáveis, 8.318.146 mudaram de valor. A
diferença absoluta média foi 0,2743. Essas grandezas apenas verificam que a
intervenção foi aplicada; não são métricas de classificação.
