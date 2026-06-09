# Relatorio tecnico curto - CardioIA Vision PBL Fase 4

## 1. Objetivo

O projeto CardioIA Vision tem como objetivo demonstrar, em contexto academico, como tecnicas de Visao Computacional podem apoiar a triagem de achados em radiografias de torax. A solucao foi construida sobre o dataset publico NIH Chest X-rays, disponivel no Kaggle, e organizada como um pipeline completo: analise exploratoria, pre-processamento, treinamento de modelos, avaliacao, fairness, prototipo de inferencia e integracao com app em React Native.

O problema foi modelado como classificacao multi-label para quatro patologias: `Infiltration`, `Effusion`, `Atelectasis` e `Pneumothorax`. Essa escolha e coerente com o dataset, pois uma mesma imagem pode apresentar mais de um achado simultaneamente. O projeto nao tem finalidade diagnostica e nao substitui avaliacao medica.

## 2. Dataset e pre-processamento

O dataset utilizado foi o NIH Chest X-rays, composto por radiografias toracicas e metadados associados. A EDA identificou forte presenca de `No Finding`, desbalanceamento entre patologias e casos com multiplas labels. Entre as quatro patologias selecionadas, a distribuicao observada foi:

| Patologia | Imagens | Percentual das linhas |
|---|---:|---:|
| Infiltration | 19.891 | 17,74% |
| Effusion | 13.316 | 11,88% |
| Atelectasis | 11.558 | 10,31% |
| Pneumothorax | 5.301 | 4,73% |

O pipeline aplicou redimensionamento das imagens para `224x224`, conversao para RGB, normalizacao com medias e desvios do ImageNet e data augmentation no treino, com flip horizontal e rotacao leve. Os conjuntos de treino, validacao e teste foram criados com separacao por `Patient ID`, reduzindo risco de vazamento de imagens do mesmo paciente entre treino e teste.

## 3. Modelos treinados

Foram avaliadas seis abordagens:

- CNN base treinada do zero.
- CNN melhorada, com regularizacao e normalizacao.
- `ResNet50` com transfer learning.
- `VGG16` com transfer learning.
- `EfficientNetB0` com transfer learning.
- `Vision Transformer ViT-B/16`.

Todos os modelos retornam quatro logits, um para cada patologia. A funcao de perda utilizada foi `BCEWithLogitsLoss`, adequada para classificacao multi-label. Para tratar o desbalanceamento, foi usado peso positivo por classe no treino. A avaliacao considerou F1 macro/micro, precision, recall, AUC-ROC, PR-AUC, hamming loss, matrizes de confusao por patologia, tempo medio de inferencia e tamanho dos checkpoints.

## 4. Resultados e modelo final

O ranking final combinou desempenho multi-label e eficiencia operacional, com pesos de 40% para F1 macro, 30% para recall macro, 20% para AUC-ROC macro e 10% para tempo medio de inferencia. O modelo selecionado foi o `Vision Transformer ViT`.

| Modelo | F1 macro | Recall macro | AUC-ROC macro | PR-AUC macro | Tempo s/img | Score |
|---|---:|---:|---:|---:|---:|---:|
| Vision Transformer ViT | 0,5426 | 0,6605 | 0,7201 | 0,5323 | 0,0100 | 1,9 |
| VGG16 | 0,5392 | 0,6826 | 0,7157 | 0,5256 | 0,0100 | 2,0 |
| ResNet50 | 0,5277 | 0,6505 | 0,6998 | 0,4979 | 0,0102 | 4,0 |
| EfficientNetB0 | 0,5205 | 0,6516 | 0,6962 | 0,4896 | 0,0101 | 4,2 |
| CNN melhorada | 0,4821 | 0,8146 | 0,6483 | 0,4291 | 0,0102 | 4,4 |
| CNN base | 0,4935 | 0,6824 | 0,6667 | 0,4452 | 0,0110 | 4,5 |

No teste, o ViT obteve F1 macro de `0,5426`, recall macro de `0,6605`, AUC-ROC macro de `0,7201`, PR-AUC macro de `0,5323`, precision macro de `0,4651` e hamming loss de `0,3328`. A acuracia exata foi `0,2116`, valor esperado em um problema multi-label, pois essa metrica exige que todas as quatro decisoes da imagem estejam corretas simultaneamente.

O `VGG16` teve desempenho muito proximo e maior recall macro, mas perdeu no ranking ponderado por pequena diferenca de F1, AUC e eficiencia. A `CNN melhorada` teve recall elevado, mas com custo de mais falsos positivos e menor especificidade, mostrando que recall isolado nao e suficiente para selecionar um modelo em contexto de saude.

## 5. Fairness, governanca e limitacoes

A analise de fairness avaliou genero, faixa etaria e posicao da imagem (`AP`/`PA`). Foram calculadas metricas por subgrupo, incluindo precision, recall, F1, AUC-ROC, specificity, taxa de falsos positivos e taxa de falsos negativos. Os maiores gaps apareceram principalmente por faixa etaria e posicao da imagem, com destaque para `Pneumothorax` por idade e `Infiltration` por `View Position`.

Esses resultados devem ser interpretados como alertas de governanca, nao como conclusoes clinicas definitivas. Alguns subgrupos possuem poucas amostras, o que pode gerar metricas instaveis. Para uso real, seriam necessarias validacao externa, revisao por especialistas, calibracao de probabilidades, monitoramento continuo de vies e protocolo formal de governanca.

## 6. Prototipo e entrega

O notebook principal esta em `notebooks/CardioIA_Vision_PBL_Fase4.ipynb`. Ele contem o fluxo completo do desafio: download/organizacao, EDA, pre-processamento, DataLoaders, modelos, comparativo, fairness e conclusao.

Para o IR ALEM 2, foi criado um prototipo com backend Flask e app React Native/Expo:

- Backend: `backend/app.py`.
- App: `mobile-app/App.js`.
- Docker Compose: `docker-compose.yml`.
- Relatorio do app: `Docs/ir_alem_2_mobile.md`.

O backend carrega o checkpoint local `vision_transformer_vit.pt` e expoe endpoints para saude, metricas e predicao. O app permite selecionar uma imagem, enviar ao backend e visualizar probabilidades para as quatro patologias.

Os checkpoints `.pt` nao foram enviados ao GitHub comum porque alguns ultrapassam 100 MB. Eles devem ser gerados pelo notebook ou disponibilizados separadamente via Git LFS/Drive. Essa decisao mantem o repositorio versionavel sem perder a reprodutibilidade do fluxo.

## 7. Conclusao

O projeto atende aos requisitos principais do PBL ao implementar pre-processamento, CNN do zero, transfer learning, avaliacao com metricas classicas, comparacao de modelos e prototipo de apresentacao dos resultados. Tambem cobre os pontos de governanca/fairness e integra um app React Native com backend Flask, atendendo ao IR ALEM 2. Como limitacao central, trata-se de uma prova de conceito academica em dataset publico, sem validacao clinica externa.
