# CardioIA Vision - PBL Fase 4

Projeto academico da Fase 4 do PBL FIAP para aplicar Visao Computacional em imagens medicas do dataset NIH Chest X-rays.

## Objetivo

Construir um prototipo de Assistente Cardiologico Virtual capaz de:

1. Baixar e organizar o dataset NIH Chest X-rays.
2. Pre-processar imagens de raio-X de torax.
3. Criar splits de treino, validacao e teste, preferencialmente separados por paciente.
4. Treinar uma CNN propria.
5. Treinar uma CNN padrao de referencia.
6. Treinar modelos com Transfer Learning: ResNet50, EfficientNetB0, EfficientNetB3 e DenseNet121.
7. Treinar ou adaptar um modelo baseado em Vision Transformer.
8. Comparar desempenho e eficiencia dos modelos.
9. Apresentar o resultado em um prototipo simples.
10. Documentar limitacoes, governanca e riscos eticos.

## Dataset

Dataset recomendado no enunciado:

<https://www.kaggle.com/datasets/nih-chest-xrays/data>

Problema inicial recomendado:

- Classe 0: `No Finding`
- Classe 1: `Cardiomegaly`

O dataset nao deve ser versionado no Git. Os arquivos baixados devem ficar em:

```text
data/raw/
  Data_entry_2017.csv
  images/
```

## Estrutura do Projeto

```text
Docs/                 Documentos do PBL, plano e relatorio tecnico
data/raw/             Dataset original baixado do Kaggle
data/processed/       Dados derivados ou imagens pre-processadas
data/splits/          CSVs de treino, validacao e teste
notebooks/            Notebooks de download, EDA, treino e avaliacao
src/                  Codigo Python reutilizavel
src/models/           Arquiteturas de modelos
src/training/         Rotinas de treino, avaliacao e metricas
src/app/              Prototipo Flask
models/checkpoints/   Pesos salvos durante treinamento
models/exported/      Modelo final exportado
reports/              Figuras, tabelas e metricas para o relatorio
```

## Ordem de Implementacao

A fonte de verdade do projeto e:

```text
Docs/plano_implementacao_pbl_fase4.md
```

Ordem resumida:

1. Estrutura do projeto e configuracoes globais.
2. Notebook para download e organizacao do dataset Kaggle.
3. EDA do dataset.
4. Dataset binario, balanceamento e splits por paciente.
5. Pre-processamento e DataLoaders.
6. Pipeline de treino e avaliacao.
7. CNN propria e CNN padrao.
8. Transfer Learning.
9. Vision Transformer.
10. Comparacao de modelos.
11. Prototipo.
12. Governanca e fairness.
13. Relatorio tecnico e checklist FIAP.

## Notebooks

Notebook principal para entrega:

```text
notebooks/CardioIA_Vision_PBL_Fase4.ipynb
```

Ele consolida o fluxo completo em uma narrativa unica para avaliacao: download, EDA, pre-processamento, splits, CNN propria, CNN padrao, Transfer Learning, Vision Transformer, comparacao, prototipo e governanca.

Notebooks modulares de apoio:

O primeiro notebook operacional e:

```text
notebooks/00_download_dataset_kaggle.ipynb
```

Ele baixa ou orienta o download manual do dataset NIH Chest X-rays, organiza `Data_entry_2017.csv`, extrai/localiza imagens e gera `data/raw/image_paths.csv` para as proximas etapas.

Depois do download, execute:

```text
notebooks/01_eda_dataset.ipynb
```

Ele analisa distribuicao de labels, `No Finding`, `Cardiomegaly`, casos multi-label, pacientes, sexo, idade, posicao da imagem e salva tabelas/graficos em `reports/`.

Depois da EDA, execute:

```text
notebooks/02_preprocessamento_splits.ipynb
```

Ele cria os datasets binarios limpo e realista, aplica balanceamento por undersampling, calcula pesos de classe e gera splits por paciente em `data/splits/`.

Para validar pre-processamento e DataLoaders, execute:

```text
notebooks/03_validacao_preprocessamento_dataloaders.ipynb
```

Ele verifica transforms, carrega os splits canonicos, inspeciona o shape de um batch e mostra amostras pre-processadas.

Para validar o pipeline de treino e avaliacao, execute:

```text
notebooks/04_validacao_pipeline_treino_avaliacao.ipynb
```

Ele treina um modelo pequeno por uma epoca, salva checkpoint, historico, metricas, matriz de confusao e curva ROC. Esse modelo e apenas uma validacao tecnica do pipeline, nao o modelo final do PBL.

Para treinar as CNNs iniciais do PBL, execute:

```text
notebooks/05_treinamento_cnn_propria.ipynb
notebooks/06_treinamento_cnn_padrao.ipynb
```

A CNN propria atende ao criterio de rede criada pelo grupo. A CNN padrao serve como baseline simples para comparacao usando os mesmos splits, loss e metricas.

Para treinar os modelos de Transfer Learning, execute:

```text
notebooks/07_treinamento_transfer_learning.ipynb
```

Ele treina ResNet50, EfficientNetB0, EfficientNetB3 e DenseNet121 em duas fases: cabeca classificadora congelando o backbone e fine-tuning parcial dos blocos finais.

Para treinar o modelo baseado em Transformer, execute:

```text
notebooks/08_treinamento_transformer_vision.ipynb
```

Ele treina ViT-B/16 por padrao, com opcao de Swin Tiny no codigo, e salva uma comparacao inicial com metricas ja existentes das CNNs.

Para consolidar metricas e escolher o modelo final, execute:

```text
notebooks/09_comparacao_modelos.ipynb
```

Ele junta os arquivos `*_metrics.csv`, gera graficos comparativos, ranqueia desempenho/eficiencia e exporta o checkpoint escolhido para `models/exported/`.

Para testar o prototipo de inferencia, use:

```text
notebooks/10_prototipo_inferencia.ipynb
```

Para rodar a interface Flask:

```text
python -m src.app.flask_app
```

A interface abre em `http://127.0.0.1:5000` e permite upload de imagem para classificacao pelo modelo final exportado.

Para gerar a analise de governanca e fairness, execute:

```text
notebooks/11_governanca_fairness.ipynb
```

Ele calcula representatividade por subgrupo, metricas por sexo/faixa etaria/posicao da imagem, gaps de desempenho e uma discussao tecnica para o relatorio.

## Configuracoes Globais

As configuracoes compartilhadas ficam em:

```text
src/config.py
```

Esse arquivo centraliza caminhos, tamanho de imagem, batch size, seed, nomes de classes e parametros iniciais de experimento.

## Observacao de Uso

Este projeto tem finalidade exclusivamente academica e educacional. Os modelos treinados nao devem ser usados como ferramenta diagnostica real.

## Relatorio e Checklist FIAP

Documentos finais:

```text
Docs/relatorio_tecnico.md
Docs/checklist_criterios_fiap.md
```

Antes da entrega, execute os notebooks com o dataset real e atualize a secao de resultados do relatorio com as metricas geradas em `reports/metricas/`.
