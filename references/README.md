# Corpus bibliográfico e mapa de evidências

Esta pasta apresenta as fontes que fundamentam o problema, o desenho
experimental e a interpretação dos resultados. Ela permite percorrer o caminho
entre uma afirmação do estudo, a evidência que a sustenta e o registro
bibliográfico correspondente.

| Recurso | Finalidade |
|---|---|
| [`library.bib`](library.bib) | Metadados BibTeX das obras citadas e das fontes diretamente relacionadas. |
| [`claims.csv`](claims.csv) | Mapa entre afirmações, tipo de evidência, fontes e condição de suporte. |
| [`notes/`](notes/) | Fichas analíticas por fonte, identificadas pela mesma chave usada no BibTeX. |

## Como consultar

1. localize a afirmação em `claims.csv`;
2. use `source_keys` para abrir as fichas correspondentes em `notes/`;
3. consulte `evidence_location` para identificar artigo, seção, página,
   código ou experimento relevante;
4. recupere DOI, URL e dados editoriais em `library.bib`.

O campo `status` distingue suporte direto, reprodução independente, resultados
restritos à população estudada, discrepâncias documentadas e afirmações ainda
não usadas como evidência consolidada. Assim, fontes publicadas, repositórios de
software, decisões metodológicas e hipóteses experimentais não recebem o mesmo
peso epistemológico.

As fichas são sínteses analíticas autorais e não substituem as obras originais.
Os textos integrais não são redistribuídos neste repositório; DOI e URLs permitem
consultá-los nas fontes autorizadas.
