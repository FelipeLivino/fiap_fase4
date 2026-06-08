# Relatorio Tecnico - CardioIA Vision

## 1. Introducao

O CardioIA Vision e um prototipo academico desenvolvido para a Fase 4 do PBL da FIAP. A proposta e aplicar Visao Computacional em imagens medicas publicas, usando redes neurais convolucionais, Transfer Learning e modelo baseado em Transformer para classificar radiografias de torax.

O projeto utiliza o dataset publico NIH Chest X-rays, recomendado no enunciado da atividade. O problema inicial foi definido como classificacao binaria:

- Classe 0: `No Finding`
- Classe 1: `Cardiomegaly`

A escolha de `Cardiomegaly` aproxima o projeto do contexto cardiologico do CardioIA, mantendo o escopo inicial suficientemente controlado para avaliacao tecnica, documentacao e prototipacao.

## 2. Objetivo

Construir um prototipo capaz de:

1. Baixar e organizar o dataset NIH Chest X-rays.
2. Pre-processar imagens medicas de raio-X.
3. Criar datasets binarios limpo e realista.
4. Aplicar balanceamento e splits por paciente.
5. Treinar e avaliar uma CNN propria.
6. Treinar e avaliar uma CNN padrao.
7. Treinar e avaliar modelos de Transfer Learning.
8. Treinar e avaliar um modelo baseado em Vision Transformer.
9. Comparar desempenho e eficiencia.
10. Escolher um modelo final.
11. Apresentar resultados em prototipo simples.
12. Discutir governanca, fairness e limitacoes clinicas.

## 3. Dataset Utilizado

Dataset: NIH Chest X-rays

Fonte: <https://www.kaggle.com/datasets/nih-chest-xrays/data>

O dataset contem radiografias de torax e metadados clinicos, incluindo identificador da imagem, labels de achados, ID do paciente, idade, sexo e posicao da imagem.

O download e organizacao dos dados sao feitos pelo notebook:

- `notebooks/00_download_dataset_kaggle.ipynb`

Esse notebook:

1. Tenta baixar o dataset com `kagglehub`.
2. Usa Kaggle API como fallback.
3. Documenta fallback manual.
4. Organiza `Data_entry_2017.csv`.
5. Organiza imagens em `data/raw/images/`.
6. Gera `data/raw/image_paths.csv`.
7. Valida imagens encontradas e ausentes.

## 4. Analise Exploratoria

A analise exploratoria e realizada no notebook:

- `notebooks/01_eda_dataset.ipynb`

A EDA cobre:

1. Validacao das colunas principais.
2. Nulos e duplicatas.
3. Distribuicao geral das labels.
4. Frequencia de `No Finding`.
5. Frequencia de `Cardiomegaly`.
6. Casos de `Cardiomegaly` isolada.
7. Casos multi-label.
8. Distribuicao por sexo.
9. Distribuicao por idade.
10. Distribuicao por posicao da imagem.
11. Visualizacao de amostras.

Artefatos esperados apos execucao:

- `reports/tabelas/eda_distribuicao_labels.csv`
- `reports/tabelas/eda_resumo_no_finding_cardiomegaly.csv`
- `reports/tabelas/eda_distribuicao_quantidade_labels_por_imagem.csv`
- `reports/figuras/eda_distribuicao_labels.png`
- `reports/figuras/eda_no_finding_vs_cardiomegaly.png`

## 5. Pre-processamento

O pre-processamento esta implementado em:

- `src/preprocessing.py`
- `src/datasets.py`
- `notebooks/03_validacao_preprocessamento_dataloaders.ipynb`

Etapas aplicadas:

1. Carregamento da imagem.
2. Conversao para RGB.
3. Redimensionamento para `224x224`.
4. Conversao para tensor.
5. Normalizacao com media e desvio padrao do ImageNet.
6. Augmentations leves apenas no treino.

Augmentations de treino:

1. Rotacao pequena.
2. Translacao pequena.
3. Ajuste leve de brilho.
4. Ajuste leve de contraste.

O horizontal flip fica desativado por padrao porque pode alterar lateralidade anatomica.

## 6. Criacao dos Splits

A criacao dos datasets binarios, balanceamento e splits e feita no notebook:

- `notebooks/02_preprocessamento_splits.ipynb`

Foram criadas duas versoes:

### Dataset limpo

- Classe 0: imagens exatamente `No Finding`.
- Classe 1: imagens exatamente `Cardiomegaly`.

### Dataset realista

- Classe 0: imagens exatamente `No Finding`.
- Classe 1: imagens que contem `Cardiomegaly`, mesmo com outras labels.

O split e feito por `Patient ID`, evitando que imagens do mesmo paciente aparecam simultaneamente em treino, validacao e teste.

Arquivos principais:

- `data/splits/dataset_binary_clean.csv`
- `data/splits/dataset_binary_realistic.csv`
- `data/splits/train.csv`
- `data/splits/val.csv`
- `data/splits/test.csv`

## 7. Balanceamento

O projeto implementa:

1. Balanceamento por undersampling.
2. Pesos de classe para treino com dataset desbalanceado.

Arquivos esperados:

- `data/splits/dataset_binary_clean_balanced_undersampled.csv`
- `data/splits/dataset_binary_realistic_balanced_undersampled.csv`
- `data/splits/class_weights_clean.json`
- `data/splits/class_weights_realistic.json`

Essa abordagem permite comparar uma estrategia simples e interpretavel com estrategias que preservam mais dados.

## 8. Modelos Treinados

O projeto contempla os seguintes modelos:

### CNN propria

Arquivo:

- `src/models/custom_cnn.py`

Notebook:

- `notebooks/05_treinamento_cnn_propria.ipynb`

A arquitetura usa blocos convolucionais com BatchNorm, ReLU, MaxPooling, Dropout e Global Average Pooling.

### CNN padrao

Arquivo:

- `src/models/standard_cnn.py`

Notebook:

- `notebooks/06_treinamento_cnn_padrao.ipynb`

Serve como baseline simples para comparar com a CNN propria.

### Transfer Learning

Arquivo:

- `src/models/transfer_learning.py`

Notebook:

- `notebooks/07_treinamento_transfer_learning.ipynb`

Modelos:

1. ResNet50.
2. EfficientNetB0.
3. EfficientNetB3.
4. DenseNet121.

Cada modelo e treinado em duas fases:

1. Backbone congelado e treino da cabeca classificadora.
2. Fine-tuning parcial dos blocos finais.

### Vision Transformer

Arquivo:

- `src/models/vision_transformer.py`

Notebook:

- `notebooks/08_treinamento_transformer_vision.ipynb`

Modelo principal:

- ViT-B/16.

Opcao adicional:

- Swin Tiny.

## 9. Metricas de Avaliacao

As metricas sao implementadas em:

- `src/training/metrics.py`
- `src/training/evaluate.py`
- `src/training/train.py`

Metricas principais:

1. Accuracy.
2. Precision.
3. Recall.
4. F1-score.
5. AUC-ROC.
6. Matriz de confusao.

Metricas de eficiencia:

1. Tempo total de treino.
2. Tempo medio por epoca.
3. Tempo medio de inferencia por imagem.
4. Total de parametros.
5. Parametros treinaveis.
6. Tamanho do checkpoint.

## 10. Comparacao de Eficiencia

A comparacao final e feita no notebook:

- `notebooks/09_comparacao_modelos.ipynb`

Esse notebook:

1. Consolida arquivos `*_metrics.csv`.
2. Gera ranking dos modelos.
3. Compara desempenho e eficiencia.
4. Gera graficos de F1, AUC, inferencia e parametros.
5. Escolhe o modelo final.
6. Exporta o checkpoint final para `models/exported/`.

O criterio de escolha prioriza:

1. F1-score.
2. Recall.
3. AUC-ROC.
4. Tempo de inferencia.

Essa priorizacao e adequada ao contexto de saude porque falsos negativos podem ser mais graves que falsos positivos.

## 11. Prototipo

O projeto possui dois prototipos.

### Notebook de inferencia

- `notebooks/10_prototipo_inferencia.ipynb`

Funcionalidades:

1. Carrega o modelo final exportado.
2. Recebe caminho de imagem.
3. Aplica pre-processamento.
4. Retorna classe prevista.
5. Retorna probabilidade de `Cardiomegaly`.
6. Exibe aviso de uso academico.

### Flask

- `src/app/flask_app.py`
- `src/app/templates/index.html`
- `src/app/templates/result.html`
- `src/app/static/css/styles.css`

Funcionalidades:

1. Upload de imagem.
2. Inferencia pelo modelo final.
3. Exibicao da imagem.
4. Exibicao da classe prevista.
5. Exibicao das probabilidades.
6. Aviso de uso academico.

Comando:

```bash
python -m src.app.flask_app
```

URL:

```text
http://127.0.0.1:5000
```

## 12. Etica, Governanca e Limitacoes

A analise de governanca e fairness esta em:

- `src/training/fairness.py`
- `notebooks/11_governanca_fairness.ipynb`

Ela avalia:

1. Representatividade por classe.
2. Representatividade por sexo.
3. Representatividade por faixa etaria.
4. Representatividade por posicao da imagem.
5. Metricas por subgrupo.
6. Gaps de desempenho entre subgrupos.

Riscos e limitacoes:

1. O modelo nao substitui avaliacao medica.
2. Os labels podem conter ruido.
3. Falsos negativos podem atrasar investigacao clinica.
4. Falsos positivos podem gerar preocupacao ou exames desnecessarios.
5. O dataset pode nao representar todos os grupos populacionais.
6. O prototipo nao foi validado clinicamente.
7. O uso real exigiria revisao medica, validacao prospectiva e governanca formal.

## 13. Checklist dos Criterios FIAP

Notebook principal de entrega:

- `notebooks/CardioIA_Vision_PBL_Fase4.ipynb`

Os notebooks separados permanecem como apoio tecnico e execucao detalhada de cada etapa.

| Criterio | Pontos | Onde foi atendido |
|---|---:|---|
| Pipeline de pre-processamento implementado | 3 | `src/preprocessing.py`, `src/datasets.py`, `notebooks/03_validacao_preprocessamento_dataloaders.ipynb` |
| Treinamento e avaliacao de CNN do zero | 2 | `src/models/custom_cnn.py`, `notebooks/05_treinamento_cnn_propria.ipynb` |
| Transfer Learning funcional | 2 | `src/models/transfer_learning.py`, `notebooks/07_treinamento_transfer_learning.ipynb` |
| Prototipo simples | 2 | `notebooks/10_prototipo_inferencia.ipynb`, `src/app/flask_app.py` |
| Documentacao clara | 1 | `README.md`, `Docs/plano_implementacao_pbl_fase4.md`, `Docs/relatorio_tecnico.md` |
| Trabalho em grupo | Extra | Preencher com nomes dos integrantes |
| Ir Alem: etica e governanca | Extra | `notebooks/11_governanca_fairness.ipynb`, `src/training/fairness.py` |
| Ir Alem: mobile | Extra | Nao implementado nesta versao |

## 14. Resultados

Esta secao deve ser atualizada apos executar os notebooks de treinamento com o dataset real.

Arquivos esperados:

- `reports/metricas/model_comparison_all_metrics.csv`
- `reports/metricas/model_comparison_ranked.csv`
- `reports/metricas/final_model_selection.csv`
- `reports/justificativa_modelo_final.txt`

Tabela sugerida para preencher:

| Modelo | Accuracy | Precision | Recall | F1 | AUC-ROC | Tempo inferencia | Parametros |
|---|---:|---:|---:|---:|---:|---:|---:|
| CNN propria | A preencher | A preencher | A preencher | A preencher | A preencher | A preencher | A preencher |
| CNN padrao | A preencher | A preencher | A preencher | A preencher | A preencher | A preencher | A preencher |
| ResNet50 | A preencher | A preencher | A preencher | A preencher | A preencher | A preencher | A preencher |
| EfficientNetB0 | A preencher | A preencher | A preencher | A preencher | A preencher | A preencher | A preencher |
| EfficientNetB3 | A preencher | A preencher | A preencher | A preencher | A preencher | A preencher | A preencher |
| DenseNet121 | A preencher | A preencher | A preencher | A preencher | A preencher | A preencher | A preencher |
| Vision Transformer | A preencher | A preencher | A preencher | A preencher | A preencher | A preencher | A preencher |

## 15. Conclusao

O projeto implementa um pipeline completo de Visao Computacional para analise academica de imagens medicas. A solucao cobre desde download e organizacao do dataset ate treinamento, avaliacao, comparacao de modelos, prototipacao e discussao de governanca.

O principal valor tecnico esta na comparacao entre CNN propria, CNN padrao, Transfer Learning e Vision Transformer usando os mesmos splits e metricas. O principal cuidado metodologico esta no split por paciente e na discussao de limitacoes clinicas, evitando apresentar o prototipo como diagnostico real.

## 16. Referencias

1. NIH Chest X-rays Dataset. Kaggle. <https://www.kaggle.com/datasets/nih-chest-xrays/data>
2. PyTorch Documentation. <https://pytorch.org/docs/stable/index.html>
3. Torchvision Models. <https://pytorch.org/vision/stable/models.html>
4. Scikit-learn Metrics. <https://scikit-learn.org/stable/modules/model_evaluation.html>
5. Flask Documentation. <https://flask.palletsprojects.com/>
