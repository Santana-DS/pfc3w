# Fernandes, Komati e Gazolli (2024)

## Pergunta e método

Detecção binária de novidade com classificadores de uma classe. O estudo
compara Isolation Forest, OCSVM, LOF, Elliptical Envelope e autoencoders, com e
sem extração de features, analisando dados reais e simulados separadamente.

## Protocolo relevante

Em cada rodada, uma instância fornece amostras normais de treinamento e teste;
as amostras anômalas entram somente no teste. A divisão normal é 60/40 e a
métrica central é F1. Logo, o protocolo avalia detecção dentro da mesma
instância, não transferência para poço físico inédito.

## Implicação

O resultado não é contraditório com a queda interpoços: alvo, dados de treino e
unidade de avaliação são diferentes.

## Evidência auditada

Páginas 349--354, metodologia experimental: os primeiros 60% das observações
normais alimentam o treinamento; os 40% normais restantes e as observações
anômalas compõem a validação; um classificador é ajustado por instância. O
artigo foi publicado on-line em 2023 e integra o volume 14, páginas 343--363,
de 2024; o BibTeX foi corrigido para o ano do volume.
