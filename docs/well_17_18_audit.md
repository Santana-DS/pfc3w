# Auditoria dos poços `WELL-00017` e `WELL-00018`

## Conclusão

Os poços 17 e 18 não foram omitidos pelo pipeline do PFC. Eles não estão nos
arquivos publicados na versão 2.0.0 congelada para os experimentos. O histórico
oficial do repositório mostra que oito arquivos CSV desses poços existiam na
versão 1.1.1 e foram excluídos no commit
`8abe820e1804c320fb9d908965913ed535cd8e9f`, que atualizou os arquivos do dataset
para a versão 2.0.0 em 25 de julho de 2024.

O `dataset/README.md` da versão 2.0.0 afirma simultaneamente que 42 poços reais
foram cobertos e que algumas instâncias reais foram removidas. O conteúdo
publicado contém 40 identificadores: `00001` a `00016` e `00019` a `00042`.
Assim, o número 42 descreve o processo de composição ou permaneceu como contagem
documental, mas não corresponde ao número de identificadores presentes nos
arquivos da versão publicada. O motivo específico da exclusão dos poços 17 e 18
não é individualizado nas notas de versão.

## Evidência histórica

| Poço | Classe nominal antiga | Arquivos excluídos | Observações antigas | Máximo de janelas 300/150 |
|---|---:|---:|---:|---:|
| `WELL-00017` | 5 | 6 | 423.093 | 2.812 |
| `WELL-00018` | 7 | 2 | 87.788 | 582 |
| **Total** | — | **8** | **510.881** | **3.394** |

O máximo de janelas é calculado por arquivo antes dos filtros de qualidade. O
valor real após limpeza poderia ser menor. Se todas fossem aprovadas, as 3.394
janelas acrescentariam aproximadamente 2,12% às 159.779 janelas atuais.

## Incompatibilidade de esquema

Os CSVs antigos não são apenas oito arquivos faltantes da versão atual. Seu
esquema contém os oito sinais antigos
`P-PDG`, `P-TPT`, `T-TPT`, `P-MON-CKP`, `T-JUS-CKP`, `P-JUS-CKGL`,
`T-JUS-CKGL` e `QGL`. O pipeline atual utiliza `T-PDG` no lugar de
`T-JUS-CKGL`; portanto, `T-PDG` estaria inteiramente ausente nos dois poços.
A versão 2.0.0 também possui variáveis e o rótulo de estado adicionais, além de
correções de unidades e de anotações.

Reintroduzir esses arquivos criaria um dataset híbrido 1.1.1/2.0.0. Isso exigiria
uma decisão metodológica explícita de harmonização, e o padrão de ausência de
`T-PDG` poderia tornar a versão do dado — e, indiretamente, a identidade dos
poços — uma pista para o classificador.

## Custo estimado de uma repetição hipotética

Os tempos medidos na máquina do projeto foram:

| Etapa | Tempo observado |
|---|---:|
| Extração retrospectiva e causal | 34,64 min |
| Oito treinos OOF de E0–E3 | 39,84 min |
| Piloto SHAP | 1,66 min |
| **Total E0–E3 + SHAP** | **76,14 min** |

Aplicando o aumento observado de volume, a repetição computacional seria de
aproximadamente 78 minutos. Regeneração das tabelas, figuras, testes e PDF
acrescentaria cerca de 10 a 20 minutos. Se a reprodução R0 também fosse refeita
para uma população híbrida, o total computacional se aproximaria de 2 h 50 min.

O custo dominante seria humano: definir a harmonização dos esquemas, documentar
a proveniência, testar a conversão e revisar a nova composição dos folds. Uma
estimativa prudente é de quatro a oito horas de engenharia e auditoria, além do
processamento.

## Decisão para o PFC1

Manter a versão 2.0.0 exatamente como publicada. Os resultados atuais abrangem
todos os arquivos reais disponíveis no commit congelado e não precisam ser
refeitos. Uma reconstrução da versão 1.1.1 pode ser realizada apenas como estudo
histórico separado, sem misturar seus arquivos às conclusões principais.
