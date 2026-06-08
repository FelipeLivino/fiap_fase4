# Plano de Implementacao - PBL Fase 4 - CardioIA Vision

## 1. Visao Geral do Projeto

### 1.1 Contexto

O projeto da Fase 4 do PBL da FIAP propõe a evolução da solução CardioIA para um prototipo de Visao Computacional aplicado a imagens medicas simuladas ou publicas. O objetivo principal e transformar imagens medicas em informacoes interpretaveis, apoiando uma tomada de decisao clinica simulada.

Neste projeto, sera utilizado o dataset publico NIH Chest X-rays, recomendado no enunciado:

<https://www.kaggle.com/datasets/nih-chest-xrays/data>

O dataset contem radiografias de torax e metadados com rotulos clinicos. A proposta inicial e construir um classificador para identificar sinais associados a cardiomegalia em imagens de raio-X de torax, comparando diferentes arquiteturas de redes neurais.

### 1.2 Objetivo Principal

Construir um prototipo chamado CardioIA Vision, capaz de:

1. Organizar e pre-processar imagens medicas do dataset NIH Chest X-rays.
2. Criar conjuntos de treino, validacao e teste de forma controlada.
3. Treinar uma CNN propria criada pelo grupo.
4. Treinar uma CNN padrao de referencia.
5. Treinar modelos de Transfer Learning com arquiteturas consolidadas.
6. Treinar ou adaptar um modelo baseado em Vision Transformer.
7. Comparar os modelos usando metricas quantitativas e criterios de eficiencia.
8. Apresentar os resultados em uma interface simples e interpretavel.
9. Documentar decisoes tecnicas, limitacoes, riscos eticos e possibilidades de melhoria.

### 1.3 Problema Inicial Recomendado

O problema inicial sera uma classificacao binaria:

- Classe 0: `No Finding`
- Classe 1: `Cardiomegaly`

Essa escolha e recomendada porque:

1. Cardiomegalia tem relacao direta com o tema cardiologico do CardioIA.
2. Classificacao binaria permite uma primeira versao mais clara e avaliavel.
3. O dataset NIH e originalmente multi-label, o que aumenta a complexidade. Comecar com um problema binario permite validar o pipeline completo antes de expandir.
4. As metricas de avaliacao ficam mais simples de explicar no relatorio e na apresentacao.

### 1.4 Possivel Expansao

Depois que a classificacao binaria estiver funcionando, o projeto pode evoluir para:

1. Classificacao multi-label com varias patologias.
2. Comparacao entre cardiomegalia isolada e cardiomegalia acompanhada de outras doencas.
3. Analise de fairness por sexo, idade e posicao da imagem.
4. Integracao com aplicativo mobile ou interface web mais completa.

---

## 2. Estrutura Recomendada do Projeto

Criar a seguinte estrutura de pastas:

```text
fase4_pbl/
  Docs/
    enunciado_pbl.md
    plano_implementacao_pbl_fase4.md
    relatorio_tecnico.md

  data/
    raw/
      Data_entry_2017.csv
      images/
    processed/
    splits/
      train.csv
      val.csv
      test.csv

  notebooks/
    01_eda_dataset.ipynb
    02_preprocessamento_splits.ipynb
    03_treinamento_cnn_propria.ipynb
    04_treinamento_cnn_padrao.ipynb
    05_transfer_learning_resnet_efficientnet_densenet.ipynb
    06_treinamento_transformer_vision.ipynb
    07_comparacao_modelos.ipynb
    08_prototipo_inferencia.ipynb

  src/
    config.py
    data_loader.py
    preprocessing.py
    datasets.py
    utils.py

    models/
      custom_cnn.py
      standard_cnn.py
      transfer_learning.py
      vision_transformer.py

    training/
      train.py
      evaluate.py
      metrics.py
      experiment_logger.py

    app/
      flask_app.py
      templates/
        index.html
        result.html
      static/
        css/
        uploads/

  models/
    checkpoints/
    exported/

  reports/
    figuras/
    metricas/
    tabelas/
```

### 2.1 Finalidade de Cada Pasta

`Docs/`

Guarda documentos textuais do projeto, como o enunciado, plano de implementacao e relatorio tecnico final.

`data/raw/`

Guarda os arquivos originais baixados do Kaggle. Esses arquivos nao devem ser alterados manualmente.

`data/processed/`

Guarda dados pre-processados, se for necessario salvar imagens redimensionadas, normalizadas ou convertidas.

`data/splits/`

Guarda arquivos CSV com a divisao oficial de treino, validacao e teste. Isso garante reprodutibilidade dos experimentos.

`notebooks/`

Guarda os notebooks usados para analise, pre-processamento, treinamento, avaliacao e demonstracao.

`src/`

Guarda codigo Python reutilizavel. A ideia e evitar que todo o projeto fique preso apenas em notebooks.

`models/checkpoints/`

Guarda pesos salvos durante o treinamento.

`models/exported/`

Guarda modelos finais exportados para uso no prototipo.

`reports/`

Guarda graficos, metricas, tabelas e outros artefatos usados no relatorio final.

---

## 3. Fase 0 - Preparacao Inicial do Projeto

### Tarefa 0.1 - Criar Estrutura de Pastas

Objetivo:

Criar a organizacao base do repositorio para separar dados, notebooks, codigo, modelos e relatorios.

Passos:

1. Criar as pastas `data`, `notebooks`, `src`, `models` e `reports`.
2. Criar subpastas internas conforme a estrutura recomendada.
3. Criar arquivos `.gitkeep` em pastas vazias, se necessario, para que elas sejam versionadas.
4. Atualizar o `README.md` explicando a estrutura.

Entregaveis:

1. Estrutura de diretorios criada.
2. `README.md` inicial com resumo do projeto.

Validacao:

1. Conferir se todas as pastas existem.
2. Conferir se o repositorio esta organizado e facil de navegar.

### Tarefa 0.2 - Definir Configuracoes Globais

Objetivo:

Criar um arquivo central de configuracao para evitar valores duplicados nos notebooks e scripts.

Passos:

1. Criar `src/config.py`.
2. Definir caminhos principais do projeto.
3. Definir tamanho padrao das imagens, inicialmente `224x224`.
4. Definir `batch_size`.
5. Definir `seed` para reprodutibilidade.
6. Definir numero de epocas inicial para testes rapidos.
7. Definir dispositivo de treino: GPU CUDA, se disponivel, ou CPU.

Exemplo de parametros:

```python
IMAGE_SIZE = 224
BATCH_SIZE = 32
SEED = 42
NUM_WORKERS = 4
```

Entregaveis:

1. Arquivo `src/config.py`.

Validacao:

1. Importar o arquivo em um notebook.
2. Confirmar que os caminhos e parametros sao carregados corretamente.

---

## 4. Fase 1 - Analise Exploratoria do Dataset

### Tarefa 1.1 - Baixar e Organizar o Dataset Pelo Notebook

Objetivo:

Garantir que o dataset NIH Chest X-rays esteja disponivel localmente em uma estrutura conhecida, usando um notebook reprodutivel para baixar, localizar, organizar e validar os arquivos.

Arquivo:

`notebooks/00_download_dataset_kaggle.ipynb`

Motivo:

O download deve ficar documentado no projeto. Isso ajuda a demonstrar reprodutibilidade para a FIAP e evita depender de uma explicacao verbal sobre como os dados foram obtidos.

Pre-requisitos:

1. Ter uma conta no Kaggle.
2. Aceitar os termos de uso do dataset na pagina do Kaggle, se solicitado.
3. Ter credenciais do Kaggle configuradas no ambiente:
   - opcao A: arquivo `kaggle.json`;
   - opcao B: variaveis de ambiente `KAGGLE_USERNAME` e `KAGGLE_KEY`;
   - opcao C: uso de `kagglehub`, quando disponivel.
4. Nunca versionar credenciais no GitHub.

Passos:

1. Criar o notebook `notebooks/00_download_dataset_kaggle.ipynb`.
2. Criar uma celula inicial explicando:
   - nome do dataset;
   - link do dataset;
   - finalidade academica;
   - necessidade de credenciais do Kaggle.
3. Instalar ou importar dependencias de download, preferencialmente `kagglehub` ou `kaggle`.
4. Tentar baixar o dataset pelo notebook.
5. Registrar no notebook o caminho retornado pelo Kaggle.
6. Criar as pastas:
   - `data/raw/`
   - `data/raw/images/`
7. Localizar o arquivo `Data_entry_2017.csv` dentro do download.
8. Copiar ou referenciar o CSV em `data/raw/Data_entry_2017.csv`.
9. Localizar os arquivos de imagem.
10. Se as imagens vierem em arquivos `.zip`, extrair para `data/raw/images/`.
11. Se as imagens ja vierem extraidas, criar uma rotina para mapear todos os arquivos `.png`, `.jpg` ou `.jpeg`.
12. Criar um indice local `data/raw/image_paths.csv` com:
    - nome da imagem;
    - caminho absoluto ou relativo;
    - tamanho do arquivo;
    - flag indicando se o arquivo existe.
13. Ler `Data_entry_2017.csv`.
14. Verificar se os nomes de imagens no CSV aparecem no indice local.
15. Contar:
    - total de linhas no CSV;
    - total de imagens referenciadas;
    - total de imagens encontradas;
    - total de imagens ausentes.
16. Mostrar uma amostra de imagens carregadas diretamente do caminho final.
17. Criar uma celula de fallback manual explicando onde colocar os arquivos caso o download automatico falhe:
    - `data/raw/Data_entry_2017.csv`
    - `data/raw/images/`

Implementacao recomendada:

1. Preferir `kagglehub.dataset_download("nih-chest-xrays/data")`, se a biblioteca estiver disponivel e funcionar no ambiente.
2. Se `kagglehub` nao funcionar, usar a Kaggle API com comando equivalente a `kaggle datasets download -d nih-chest-xrays/data`.
3. Se nenhuma opcao automatica funcionar, manter o notebook com instrucoes claras para download manual e validacao local.

Observacao importante:

O dataset completo e grande. O notebook deve separar duas ideias:

1. Download e organizacao dos dados.
2. Treinamento dos modelos.

O download pode demorar e ocupar bastante espaco em disco. O treino nao deve depender de rebaixar o dataset toda vez.

Entregaveis:

1. Notebook `notebooks/00_download_dataset_kaggle.ipynb`.
2. Dataset organizado em `data/raw/`.
3. Arquivo `data/raw/Data_entry_2017.csv`.
4. Pasta `data/raw/images/` com as imagens.
5. Arquivo `data/raw/image_paths.csv`.
6. Celula no notebook indicando quantas imagens foram localizadas e quantas ficaram ausentes.

Validacao:

1. O numero de imagens disponiveis deve ser compativel com o dataset baixado.
2. O CSV deve ser lido sem erros.
3. Uma amostra de imagens deve ser visualizada corretamente.
4. O notebook deve poder ser reexecutado sem duplicar desnecessariamente os dados.
5. O notebook deve falhar com mensagem clara caso as credenciais do Kaggle nao estejam configuradas.

### Tarefa 1.2 - Criar Notebook de EDA

Objetivo:

Entender a composicao do dataset antes de treinar modelos.

Arquivo:

`notebooks/01_eda_dataset.ipynb`

Passos:

1. Importar bibliotecas principais: `pandas`, `numpy`, `matplotlib`, `seaborn`, `PIL` ou `opencv`.
2. Ler `Data_entry_2017.csv`.
3. Exibir as primeiras linhas do dataset.
4. Conferir colunas disponiveis:
   - `Image Index`
   - `Finding Labels`
   - `Patient ID`
   - `Patient Age`
   - `Patient Gender`
   - `View Position`
   - dimensoes originais da imagem
5. Verificar valores nulos.
6. Verificar duplicatas.
7. Separar a coluna `Finding Labels` em listas de labels.
8. Contar quantas imagens existem por label.
9. Contar quantas imagens sao `No Finding`.
10. Contar quantas imagens contem `Cardiomegaly`.
11. Contar quantas imagens possuem multiplas labels.
12. Gerar graficos de distribuicao das patologias.
13. Gerar grafico de distribuicao por sexo.
14. Gerar grafico de distribuicao por idade.
15. Gerar grafico de distribuicao por posicao da imagem (`PA` e `AP`).
16. Visualizar amostras de imagens `No Finding`.
17. Visualizar amostras de imagens com `Cardiomegaly`.

Entregaveis:

1. Notebook `01_eda_dataset.ipynb`.
2. Graficos salvos em `reports/figuras/`.
3. Tabela com distribuicao de labels salva em `reports/tabelas/`.

Validacao:

1. O notebook deve executar do inicio ao fim.
2. Os graficos devem ser legiveis.
3. A analise deve deixar claro se ha desbalanceamento.
4. A analise deve deixar claro se ha muitos casos multi-label.

### Tarefa 1.3 - Registrar Conclusoes da EDA

Objetivo:

Transformar a analise exploratoria em decisoes tecnicas justificadas.

Passos:

1. Escrever um resumo dentro do notebook com os principais achados.
2. Responder:
   - O dataset e balanceado?
   - Quantas imagens de `Cardiomegaly` existem?
   - Quantas imagens de `No Finding` existem?
   - Existem muitos casos multi-label?
   - Ha diferenca relevante entre `PA` e `AP`?
   - Ha algum possivel vies por idade ou sexo?
3. Salvar essas conclusoes tambem no relatorio tecnico.

Entregaveis:

1. Secao de conclusoes no notebook.
2. Texto inicial para o relatorio.

Validacao:

1. As escolhas futuras de balanceamento e split devem estar apoiadas nesses resultados.

---

## 5. Fase 2 - Pre-processamento e Criacao dos Splits

### Tarefa 2.1 - Definir Versoes do Dataset Binario

Objetivo:

Criar versoes controladas do problema `No Finding` vs `Cardiomegaly`.

Versao A - Dataset limpo:

1. Classe 0: imagens com label exatamente `No Finding`.
2. Classe 1: imagens com label exatamente `Cardiomegaly`.
3. Excluir imagens com `Cardiomegaly` acompanhada de outras patologias.

Vantagem:

Essa versao reduz ambiguidade e facilita interpretar se o modelo realmente aprendeu diferencas entre normalidade e cardiomegalia.

Limitacao:

Ela e menos realista, porque na pratica pacientes podem apresentar multiplas condicoes ao mesmo tempo.

Versao B - Dataset realista:

1. Classe 0: imagens com label exatamente `No Finding`.
2. Classe 1: imagens que contem `Cardiomegaly`, mesmo se houver outras labels.

Vantagem:

Essa versao representa melhor o carater multi-label do dataset NIH.

Limitacao:

O modelo pode aprender sinais associados a outras doencas junto com cardiomegalia.

Entregaveis:

1. `data/splits/dataset_binary_clean.csv`
2. `data/splits/dataset_binary_realistic.csv`

Validacao:

1. Conferir contagem de cada classe nas duas versoes.
2. Conferir exemplos de linhas de cada classe.

### Tarefa 2.2 - Implementar Balanceamento

Objetivo:

Tratar o desbalanceamento entre `No Finding` e `Cardiomegaly`.

Estrategias a comparar:

1. Undersampling:
   - Reduzir a classe majoritaria para ficar com tamanho parecido com a classe minoritaria.
   - Vantagem: simples e facil de explicar.
   - Desvantagem: descarta muitas imagens.

2. Class weights:
   - Manter todas as imagens e ajustar o peso da funcao de perda.
   - Vantagem: preserva mais dados.
   - Desvantagem: pode exigir ajuste fino.

3. Weighted sampler:
   - Amostrar exemplos de forma balanceada durante o treino.
   - Vantagem: util em PyTorch.
   - Desvantagem: adiciona complexidade ao DataLoader.

Passos:

1. Calcular distribuicao original das classes.
2. Criar uma versao balanceada por undersampling.
3. Calcular pesos de classe para a versao desbalanceada.
4. Salvar as duas configuracoes para comparacao.

Entregaveis:

1. CSV balanceado.
2. Pesos de classe registrados no notebook.
3. Grafico antes/depois do balanceamento.

Validacao:

1. A distribuicao do dataset balanceado deve estar correta.
2. Os pesos de classe devem fazer sentido matematicamente.

### Tarefa 2.3 - Criar Split por Paciente

Objetivo:

Evitar vazamento de dados entre treino, validacao e teste.

Problema:

O mesmo paciente pode ter varias imagens. Se imagens do mesmo paciente aparecerem em treino e teste, o modelo pode parecer melhor do que realmente e.

Passos:

1. Usar a coluna `Patient ID`.
2. Separar pacientes, nao apenas imagens.
3. Criar proporcao:
   - 70% treino
   - 15% validacao
   - 15% teste
4. Garantir que nenhum `Patient ID` apareca em mais de um conjunto.
5. Manter a distribuicao de classes o mais equilibrada possivel.

Entregaveis:

1. `data/splits/train.csv`
2. `data/splits/val.csv`
3. `data/splits/test.csv`

Validacao:

1. Confirmar que nao ha intersecao de pacientes entre treino, validacao e teste.
2. Mostrar distribuicao das classes em cada split.
3. Mostrar numero de pacientes por split.

### Tarefa 2.4 - Implementar Pre-processamento de Imagens

Objetivo:

Padronizar as imagens para entrada nos modelos.

Passos:

1. Carregar imagem original.
2. Converter para RGB quando necessario.
3. Redimensionar para `224x224`.
4. Normalizar valores dos pixels.
5. Aplicar augmentations apenas no treino.
6. Nao aplicar augmentations aleatorias em validacao e teste.

Augmentations sugeridas:

1. Rotacao pequena, por exemplo ate 10 graus.
2. Ajuste leve de brilho e contraste.
3. Pequena translacao.
4. Horizontal flip deve ser avaliado com cuidado, pois pode alterar lateralidade anatomica.

Entregaveis:

1. `src/preprocessing.py`
2. Transformacoes de treino, validacao e teste.

Validacao:

1. Visualizar imagens antes e depois do pre-processamento.
2. Confirmar que a forma final dos tensores esta correta.
3. Confirmar que normalizacao esta compativel com modelos pre-treinados.

---

## 6. Fase 3 - Pipeline Reutilizavel de Treinamento

### Tarefa 3.1 - Escolher Framework Principal

Recomendacao:

Usar PyTorch.

Justificativa:

1. PyTorch oferece boa flexibilidade para criar CNN propria.
2. Facilita Transfer Learning com `torchvision`.
3. Facilita uso de Vision Transformers com `timm` ou `torchvision`.
4. Permite uso eficiente da GPU NVIDIA RTX 5070 Ti com CUDA, desde que o ambiente esteja configurado.

Entregaveis:

1. Ambiente Python documentado.
2. Lista de dependencias no `README.md` ou `requirements.txt`.

Validacao:

1. Rodar um teste simples verificando `torch.cuda.is_available()`.
2. Mostrar nome da GPU detectada.

### Tarefa 3.2 - Criar Dataset e DataLoader

Objetivo:

Criar uma classe reutilizavel para carregar imagens e labels.

Arquivo:

`src/datasets.py`

Passos:

1. Ler um CSV de split.
2. Receber caminho base das imagens.
3. Carregar imagem pelo nome em `Image Index`.
4. Aplicar transformacao correta.
5. Retornar imagem e label.
6. Criar DataLoaders para treino, validacao e teste.

Entregaveis:

1. Classe `ChestXrayBinaryDataset`.
2. Funcao para criar DataLoaders.

Validacao:

1. Carregar um batch.
2. Verificar shape das imagens.
3. Verificar shape dos labels.
4. Visualizar algumas imagens do batch.

### Tarefa 3.3 - Criar Loop de Treinamento

Objetivo:

Padronizar o treino para todos os modelos.

Arquivo:

`src/training/train.py`

Passos:

1. Receber modelo, DataLoaders, otimizador, loss function e numero de epocas.
2. Executar treino por epoca.
3. Calcular perda de treino.
4. Calcular perda e metricas de validacao.
5. Salvar melhor modelo baseado em F1-score ou AUC.
6. Registrar tempo por epoca.
7. Registrar historico de metricas.

Entregaveis:

1. Funcao `train_model`.
2. Historico de treinamento salvo em CSV ou JSON.
3. Checkpoint do melhor modelo.

Validacao:

1. Rodar treino curto de 1 epoca.
2. Confirmar que loss diminui ou que o loop executa corretamente.
3. Confirmar que checkpoint e historico sao salvos.

### Tarefa 3.4 - Criar Avaliacao Padronizada

Objetivo:

Avaliar todos os modelos com as mesmas metricas.

Arquivo:

`src/training/evaluate.py`

Metricas obrigatorias:

1. Accuracy.
2. Precision.
3. Recall.
4. F1-score.
5. Matriz de confusao.
6. AUC-ROC.

Metricas de eficiencia:

1. Tempo medio de inferencia por imagem.
2. Tempo total de treino.
3. Tempo medio por epoca.
4. Numero de parametros treinaveis.
5. Numero total de parametros.
6. Tamanho do arquivo do modelo.

Passos:

1. Rodar predicao no conjunto de teste.
2. Coletar probabilidades.
3. Aplicar threshold padrao de 0.5.
4. Calcular metricas.
5. Gerar matriz de confusao.
6. Gerar curva ROC.
7. Salvar resultados.

Entregaveis:

1. Arquivo de metricas por modelo.
2. Figuras da matriz de confusao.
3. Figuras da curva ROC.

Validacao:

1. Todas as metricas devem ser calculadas para todos os modelos.
2. As figuras devem estar salvas em `reports/figuras/`.

---

## 7. Fase 4 - Modelos a Implementar

### Tarefa 4.1 - CNN Propria

Objetivo:

Construir uma CNN criada pelo grupo, cumprindo diretamente o requisito do PBL.

Arquivo:

`src/models/custom_cnn.py`

Arquitetura sugerida:

1. Entrada: imagem `224x224x3`.
2. Bloco 1:
   - Convolucao.
   - Batch Normalization.
   - ReLU.
   - Max Pooling.
3. Bloco 2:
   - Convolucao.
   - Batch Normalization.
   - ReLU.
   - Max Pooling.
4. Bloco 3:
   - Convolucao.
   - Batch Normalization.
   - ReLU.
   - Max Pooling.
5. Bloco 4 opcional:
   - Convolucao.
   - Batch Normalization.
   - ReLU.
   - Dropout.
6. Global Average Pooling.
7. Camada fully connected final.
8. Saida binaria.

Passos:

1. Implementar a classe da rede.
2. Testar forward pass com um batch fake.
3. Treinar por poucas epocas para validar.
4. Treinar experimento completo.
5. Salvar metricas.

Entregaveis:

1. Codigo da CNN propria.
2. Notebook `03_treinamento_cnn_propria.ipynb`.
3. Checkpoint do melhor modelo.
4. Metricas e graficos.

Validacao:

1. O modelo deve treinar sem erros.
2. A saida deve ter dimensao correta.
3. O resultado deve ser comparado aos demais modelos.

### Tarefa 4.2 - CNN Padrao de Referencia

Objetivo:

Criar uma CNN classica de referencia para comparar com a CNN propria.

Possibilidades:

1. Arquitetura inspirada em LeNet adaptada para imagens maiores.
2. Arquitetura inspirada em AlexNet simplificada.
3. Modelo CNN pequeno com menos camadas que a CNN propria.

Justificativa:

Uma CNN padrao permite avaliar se a arquitetura propria realmente agrega desempenho ou se uma arquitetura simples ja e suficiente.

Arquivo:

`src/models/standard_cnn.py`

Passos:

1. Implementar CNN padrao.
2. Testar forward pass.
3. Treinar com os mesmos splits.
4. Avaliar com as mesmas metricas.

Entregaveis:

1. Codigo da CNN padrao.
2. Notebook `04_treinamento_cnn_padrao.ipynb`.
3. Checkpoint.
4. Metricas.

Validacao:

1. Usar mesmo conjunto de treino, validacao e teste.
2. Usar metricas identicas as da CNN propria.

### Tarefa 4.3 - ResNet50 com Transfer Learning

Objetivo:

Treinar uma ResNet50 pre-treinada para classificar `No Finding` vs `Cardiomegaly`.

Arquivo:

`src/models/transfer_learning.py`

Passos:

1. Carregar ResNet50 pre-treinada.
2. Substituir a camada final para saida binaria.
3. Congelar inicialmente as camadas convolucionais.
4. Treinar apenas a cabeca classificadora.
5. Fazer fine-tuning parcial liberando ultimos blocos.
6. Avaliar no teste.

Entregaveis:

1. Funcao para criar ResNet50.
2. Metricas de treino congelado.
3. Metricas de fine-tuning.

Validacao:

1. Comparar resultado com CNN propria.
2. Registrar tempo de treino e inferencia.

### Tarefa 4.4 - EfficientNetB0

Objetivo:

Avaliar uma arquitetura eficiente em termos de parametros e desempenho.

Passos:

1. Carregar EfficientNetB0 pre-treinada.
2. Adaptar classificador final.
3. Treinar cabeca classificadora.
4. Fazer fine-tuning parcial.
5. Medir metricas e eficiencia.

Entregaveis:

1. Funcao para criar EfficientNetB0.
2. Checkpoint.
3. Metricas.

Validacao:

1. Comparar F1-score e tempo de inferencia com ResNet50 e CNN propria.

### Tarefa 4.5 - EfficientNetB3

Objetivo:

Comparar uma EfficientNet maior com a EfficientNetB0.

Passos:

1. Carregar EfficientNetB3 pre-treinada.
2. Ajustar tamanho de imagem se necessario.
3. Adaptar classificador final.
4. Treinar e avaliar.
5. Comparar ganhos de desempenho contra custo computacional.

Entregaveis:

1. Funcao para criar EfficientNetB3.
2. Checkpoint.
3. Metricas.

Validacao:

1. Verificar se EfficientNetB3 melhora F1 ou AUC em relacao a B0.
2. Medir se o aumento de custo compensa.

### Tarefa 4.6 - DenseNet121

Objetivo:

Avaliar uma arquitetura muito usada em tarefas de raio-X medico.

Justificativa:

DenseNet121 aparece com frequencia em estudos de classificacao de imagens medicas, especialmente radiografias, por reutilizar bem caracteristicas aprendidas em diferentes profundidades da rede.

Passos:

1. Carregar DenseNet121 pre-treinada.
2. Trocar classificador final.
3. Treinar cabeca classificadora.
4. Fazer fine-tuning parcial.
5. Avaliar com as mesmas metricas.

Entregaveis:

1. Funcao para criar DenseNet121.
2. Checkpoint.
3. Metricas.

Validacao:

1. Comparar DenseNet121 com ResNet50, EfficientNetB0 e EfficientNetB3.

### Tarefa 4.7 - Vision Transformer

Objetivo:

Comparar arquiteturas CNN com uma arquitetura baseada em Transformer para imagens.

Opcoes:

1. ViT-B/16.
2. Swin Transformer Tiny.
3. DeiT, se for conveniente.

Recomendacao inicial:

Comecar com ViT-B/16 ou Swin Tiny usando pesos pre-treinados.

Passos:

1. Escolher biblioteca: `torchvision` ou `timm`.
2. Carregar modelo pre-treinado.
3. Adaptar a cabeca classificadora.
4. Ajustar transformacoes de imagem conforme o modelo.
5. Treinar cabeca classificadora.
6. Fazer fine-tuning parcial.
7. Avaliar resultados.

Entregaveis:

1. `src/models/vision_transformer.py`.
2. Notebook `06_treinamento_transformer_vision.ipynb`.
3. Checkpoint.
4. Metricas.

Validacao:

1. Comparar Transformer com CNNs tradicionais.
2. Avaliar se o ganho de desempenho compensa custo computacional.

---

## 8. Fase 5 - Comparacao dos Modelos

### Tarefa 5.1 - Consolidar Resultados

Objetivo:

Juntar todas as metricas em uma tabela unica.

Arquivo:

`notebooks/07_comparacao_modelos.ipynb`

Modelos a comparar:

1. CNN propria.
2. CNN padrao.
3. ResNet50.
4. EfficientNetB0.
5. EfficientNetB3.
6. DenseNet121.
7. Vision Transformer.

Metricas principais:

1. Accuracy.
2. Precision.
3. Recall.
4. F1-score.
5. AUC-ROC.

Metricas de eficiencia:

1. Tempo total de treino.
2. Tempo medio por epoca.
3. Tempo medio de inferencia por imagem.
4. Numero total de parametros.
5. Numero de parametros treinaveis.
6. Tamanho do modelo salvo.

Entregaveis:

1. Tabela comparativa em CSV.
2. Tabela comparativa no notebook.
3. Graficos comparativos.

Validacao:

1. Todos os modelos devem aparecer na tabela.
2. As metricas devem vir do mesmo conjunto de teste.
3. O notebook deve deixar claro qual modelo foi melhor em desempenho e qual foi melhor em eficiencia.

### Tarefa 5.2 - Gerar Graficos Comparativos

Objetivo:

Facilitar a interpretacao visual dos resultados.

Graficos sugeridos:

1. F1-score por modelo.
2. AUC-ROC por modelo.
3. Tempo de inferencia por modelo.
4. Numero de parametros por modelo.
5. F1-score versus tempo de inferencia.
6. Matriz de confusao do melhor modelo.

Entregaveis:

1. Figuras salvas em `reports/figuras/`.
2. Explicacao textual no notebook.

Validacao:

1. Os graficos devem ter titulo, legenda e eixos claros.
2. Os graficos devem ajudar a escolher o melhor modelo.

### Tarefa 5.3 - Escolher Modelo Final

Objetivo:

Selecionar o modelo que sera usado no prototipo.

Criterios:

1. Melhor F1-score.
2. Bom recall para `Cardiomegaly`, pois falso negativo em saude e critico.
3. Boa AUC-ROC.
4. Tempo de inferencia aceitavel.
5. Facilidade de explicar no relatorio.

Passos:

1. Comparar todos os modelos.
2. Identificar melhor desempenho geral.
3. Identificar modelo mais eficiente.
4. Justificar escolha final.
5. Exportar o melhor checkpoint para `models/exported/`.

Entregaveis:

1. Modelo final exportado.
2. Justificativa no notebook.
3. Justificativa no relatorio.

Validacao:

1. Deve existir um modelo final claro.
2. A escolha deve estar apoiada nas metricas.

---

## 9. Fase 6 - Prototipo de Apresentacao dos Resultados

### Tarefa 6.1 - Prototipo em Notebook

Objetivo:

Criar uma demonstracao simples da inferencia do modelo.

Arquivo:

`notebooks/08_prototipo_inferencia.ipynb`

Passos:

1. Carregar modelo final.
2. Permitir selecionar ou carregar uma imagem.
3. Aplicar o mesmo pre-processamento usado no teste.
4. Executar inferencia.
5. Mostrar a imagem.
6. Mostrar classe prevista.
7. Mostrar probabilidade.
8. Mostrar mensagem de aviso educacional.

Mensagem sugerida:

```text
Este prototipo possui finalidade exclusivamente academica e educacional.
Ele nao deve ser usado como ferramenta diagnostica real.
```

Entregaveis:

1. Notebook interativo de inferencia.

Validacao:

1. Testar com imagem `No Finding`.
2. Testar com imagem `Cardiomegaly`.
3. Confirmar que o modelo retorna resultado e probabilidade.

### Tarefa 6.2 - Prototipo Web com Flask

Objetivo:

Criar uma interface simples para upload de imagem e exibicao do resultado.

Arquivo:

`src/app/flask_app.py`

Funcionalidades:

1. Tela inicial com upload de imagem.
2. Botao para enviar imagem.
3. Pre-processamento automatico.
4. Inferencia usando modelo final.
5. Tela de resultado com:
   - imagem enviada;
   - classe prevista;
   - probabilidade;
   - modelo utilizado;
   - aviso educacional.

Passos:

1. Criar rota `/`.
2. Criar rota `/predict`.
3. Salvar imagem temporariamente em `src/app/static/uploads/`.
4. Carregar modelo uma unica vez na inicializacao.
5. Aplicar transformacao de validacao/teste.
6. Retornar resultado para template HTML.

Entregaveis:

1. Aplicacao Flask funcional.
2. Templates HTML.
3. CSS simples.

Validacao:

1. Rodar a aplicacao localmente.
2. Fazer upload de uma imagem.
3. Conferir se o resultado aparece.
4. Conferir se a pagina e clara e compreensivel.

---

## 10. Fase 7 - Etica, Governanca e Limitacoes

### Tarefa 7.1 - Analisar Desbalanceamento e Representatividade

Objetivo:

Atender ao tema de Governanca e ao item "Ir Alem" do enunciado.

Passos:

1. Avaliar distribuicao das classes.
2. Avaliar distribuicao por sexo.
3. Avaliar distribuicao por idade.
4. Avaliar distribuicao por posicao da imagem.
5. Discutir se o dataset representa bem diferentes grupos de pacientes.

Entregaveis:

1. Secao no relatorio tecnico.
2. Graficos de apoio.

Validacao:

1. A discussao deve estar conectada aos dados observados na EDA.

### Tarefa 7.2 - Avaliar Metricas por Subgrupo

Objetivo:

Investigar se o modelo se comporta de forma diferente em subgrupos.

Subgrupos possiveis:

1. Sexo: masculino e feminino.
2. Faixas etarias.
3. Posicao da imagem: `PA` e `AP`.

Passos:

1. Rodar predicoes no conjunto de teste.
2. Juntar predicoes com metadados.
3. Calcular precision, recall e F1 por subgrupo.
4. Comparar diferencas.
5. Discutir riscos e limitacoes.

Entregaveis:

1. Tabela de metricas por subgrupo.
2. Discussao no relatorio.

Validacao:

1. Deve ficar claro se ha diferenca relevante de desempenho entre subgrupos.

### Tarefa 7.3 - Discutir Riscos Clinicos

Objetivo:

Mostrar responsabilidade no uso de IA em saude.

Pontos obrigatorios:

1. O modelo nao substitui avaliacao medica.
2. Labels do dataset podem conter ruido, pois foram extraidos de laudos.
3. Falsos negativos podem atrasar investigacao clinica.
4. Falsos positivos podem gerar preocupacao ou exames desnecessarios.
5. O prototipo deve ser tratado como ferramenta academica.

Entregaveis:

1. Secao de limitacoes no relatorio.
2. Aviso no prototipo.

Validacao:

1. O relatorio deve explicitar que o sistema nao e diagnostico real.

---

## 11. Fase 8 - Relatorio Tecnico

### Tarefa 8.1 - Criar Relatorio Principal

Objetivo:

Produzir um relatorio claro e bem estruturado para avaliacao da FIAP.

Arquivo:

`Docs/relatorio_tecnico.md`

Estrutura sugerida:

1. Introducao.
2. Objetivo.
3. Dataset utilizado.
4. Analise exploratoria.
5. Pre-processamento.
6. Criacao dos splits.
7. Balanceamento.
8. Modelos treinados.
9. Metricas de avaliacao.
10. Comparacao de eficiencia.
11. Prototipo.
12. Etica, governanca e limitacoes.
13. Conclusao.
14. Referencias.

Entregaveis:

1. Relatorio tecnico em Markdown.
2. Opcionalmente exportar para PDF.

Validacao:

1. O relatorio deve cobrir todos os criterios do enunciado.
2. O texto deve explicar as decisoes, nao apenas mostrar resultados.
3. As figuras e tabelas devem estar referenciadas.

### Tarefa 8.2 - Mapear Criterios da FIAP para Entregaveis

Objetivo:

Garantir que todos os pontos do enunciado foram cobertos.

Checklist:

1. Pipeline de pre-processamento implementado - 3 pontos.
2. Treinamento e avaliacao de CNN do zero - 2 pontos.
3. Transfer Learning funcional - 2 pontos.
4. Prototipo simples - 2 pontos.
5. Documentacao clara - 1 ponto.
6. Trabalho em grupo - 1 ponto extra.
7. Ir Alem: etica e governanca.
8. Ir Alem: possivel integracao mobile.

Entregaveis:

1. Checklist no relatorio.
2. Checklist no `README.md`.

Validacao:

1. Cada criterio deve apontar para um arquivo, notebook ou secao do projeto.

---

## 12. Ordem Recomendada de Implementacao

Para evitar retrabalho, a implementacao deve seguir esta ordem. Cada item deve ser tratado como uma tarefa pequena, validada antes da proxima.

1. Criar estrutura do projeto.
2. Criar configuracoes globais.
3. Criar notebook para baixar e organizar dataset pelo Kaggle.
4. Criar notebook de EDA.
5. Analisar distribuicao das labels.
6. Criar dataset binario limpo e realista.
7. Implementar balanceamento.
8. Criar splits por paciente.
9. Implementar pre-processamento.
10. Criar Dataset e DataLoader.
11. Criar pipeline de treino.
12. Criar pipeline de avaliacao.
13. Treinar CNN propria.
14. Treinar CNN padrao.
15. Treinar ResNet50.
16. Treinar EfficientNetB0.
17. Treinar EfficientNetB3.
18. Treinar DenseNet121.
19. Treinar Vision Transformer.
20. Consolidar metricas.
21. Comparar desempenho e eficiencia.
22. Escolher modelo final.
23. Criar prototipo em notebook.
24. Criar prototipo Flask.
25. Fazer analise de governanca e fairness.
26. Escrever relatorio tecnico.
27. Revisar checklist dos criterios da FIAP.

### 12.1 Como Pedir a Implementacao Para o Codex

Use os pedidos abaixo para executar o projeto em blocos. Cada bloco preserva contexto, reduz risco de retrabalho e facilita validar o que foi feito.

#### Bloco 1 - Base do Projeto

Pedido sugerido:

```text
Implemente as tarefas 1 e 2 da ordem recomendada do plano: criar estrutura do projeto e configuracoes globais. Use o arquivo Docs/plano_implementacao_pbl_fase4.md como fonte de verdade.
```

Tarefas cobertas:

1. Criar estrutura do projeto.
2. Criar configuracoes globais.

O que deve ser entregue:

1. Pastas do projeto.
2. Arquivos base.
3. `README.md` inicial.
4. `src/config.py`.

Validacao esperada:

1. Conferir arvore de pastas.
2. Conferir se `src/config.py` pode ser importado.

#### Bloco 2 - Download e Organizacao do Dataset

Pedido sugerido:

```text
Implemente a tarefa 3 da ordem recomendada: criar notebook para baixar e organizar o dataset NIH Chest X-rays pelo Kaggle, com validacao dos arquivos baixados.
```

Tarefa coberta:

3. Criar notebook para baixar e organizar dataset pelo Kaggle.

O que deve ser entregue:

1. `notebooks/00_download_dataset_kaggle.ipynb`.
2. Rotina de download via Kaggle ou `kagglehub`.
3. Fallback documentado para download manual.
4. Checagem de `Data_entry_2017.csv`.
5. Checagem das imagens.
6. `data/raw/image_paths.csv`.

Validacao esperada:

1. Notebook mostra se o Kaggle esta autenticado.
2. Notebook mostra caminho do dataset.
3. Notebook conta imagens encontradas e ausentes.
4. Notebook exibe amostra de imagens.

Observacao:

Se o ambiente nao tiver credenciais do Kaggle, a implementacao deve deixar o notebook pronto e explicar claramente onde colocar `kaggle.json` ou os arquivos baixados manualmente.

#### Bloco 3 - EDA do Dataset

Pedido sugerido:

```text
Implemente as tarefas 4 e 5 da ordem recomendada: criar notebook de EDA e analisar a distribuicao das labels, pacientes, sexo, idade, posicao da imagem e casos multi-label.
```

Tarefas cobertas:

4. Criar notebook de EDA.
5. Analisar distribuicao das labels.

O que deve ser entregue:

1. `notebooks/01_eda_dataset.ipynb`.
2. Graficos em `reports/figuras/`.
3. Tabelas em `reports/tabelas/`.
4. Conclusoes textuais no notebook.

Validacao esperada:

1. Notebook roda com o dataset local.
2. Distribuicao de `No Finding` e `Cardiomegaly` fica clara.
3. Desbalanceamento e multi-label ficam documentados.

#### Bloco 4 - Dataset Binario, Balanceamento e Splits

Pedido sugerido:

```text
Implemente as tarefas 6, 7 e 8 da ordem recomendada: criar dataset binario limpo e realista, implementar balanceamento e criar splits por paciente.
```

Tarefas cobertas:

6. Criar dataset binario limpo e realista.
7. Implementar balanceamento.
8. Criar splits por paciente.

O que deve ser entregue:

1. CSV do dataset binario limpo.
2. CSV do dataset binario realista.
3. CSV balanceado, se aplicavel.
4. `train.csv`, `val.csv` e `test.csv`.
5. Validacao de que nao ha paciente repetido entre splits.

Validacao esperada:

1. Distribuicao das classes antes e depois do balanceamento.
2. Distribuicao das classes por split.
3. Intersecao zero de `Patient ID` entre treino, validacao e teste.

#### Bloco 5 - Pre-processamento e DataLoader

Pedido sugerido:

```text
Implemente as tarefas 9 e 10 da ordem recomendada: pre-processamento das imagens, transforms, Dataset PyTorch e DataLoaders.
```

Tarefas cobertas:

9. Implementar pre-processamento.
10. Criar Dataset e DataLoader.

O que deve ser entregue:

1. `src/preprocessing.py`.
2. `src/datasets.py`.
3. Transformacoes de treino, validacao e teste.
4. Teste de carregamento de batch.

Validacao esperada:

1. Batch de imagens carrega sem erro.
2. Shapes dos tensores estao corretos.
3. Labels estao no formato correto.
4. Amostras pre-processadas podem ser visualizadas.

#### Bloco 6 - Pipeline de Treino e Avaliacao

Pedido sugerido:

```text
Implemente as tarefas 11 e 12 da ordem recomendada: pipeline de treino e pipeline de avaliacao com metricas, checkpoints e logs.
```

Tarefas cobertas:

11. Criar pipeline de treino.
12. Criar pipeline de avaliacao.

O que deve ser entregue:

1. `src/training/train.py`.
2. `src/training/evaluate.py`.
3. `src/training/metrics.py`.
4. Salvamento de checkpoints.
5. Salvamento de historico de metricas.

Validacao esperada:

1. Treino curto de uma epoca executa.
2. Avaliacao calcula accuracy, precision, recall, F1, AUC e matriz de confusao.
3. Checkpoint e logs sao salvos.

#### Bloco 7 - CNNs Baseline

Pedido sugerido:

```text
Implemente as tarefas 13 e 14 da ordem recomendada: treinar CNN propria e CNN padrao usando o pipeline ja criado.
```

Tarefas cobertas:

13. Treinar CNN propria.
14. Treinar CNN padrao.

O que deve ser entregue:

1. `src/models/custom_cnn.py`.
2. `src/models/standard_cnn.py`.
3. Notebooks de treinamento.
4. Checkpoints.
5. Metricas.

Validacao esperada:

1. Os dois modelos usam os mesmos splits.
2. As metricas sao calculadas no mesmo conjunto de teste.
3. Resultados sao salvos em formato comparavel.

#### Bloco 8 - Transfer Learning

Pedido sugerido:

```text
Implemente as tarefas 15, 16, 17 e 18 da ordem recomendada: treinar ResNet50, EfficientNetB0, EfficientNetB3 e DenseNet121.
```

Tarefas cobertas:

15. Treinar ResNet50.
16. Treinar EfficientNetB0.
17. Treinar EfficientNetB3.
18. Treinar DenseNet121.

O que deve ser entregue:

1. Funcoes em `src/models/transfer_learning.py`.
2. Notebooks ou scripts de treinamento.
3. Checkpoints.
4. Metricas de desempenho e eficiencia.

Validacao esperada:

1. Todos os modelos usam os mesmos splits.
2. Todos reportam as mesmas metricas.
3. Tempo de treino e inferencia sao registrados.

#### Bloco 9 - Vision Transformer

Pedido sugerido:

```text
Implemente a tarefa 19 da ordem recomendada: treinar Vision Transformer e comparar com as CNNs.
```

Tarefa coberta:

19. Treinar Vision Transformer.

O que deve ser entregue:

1. `src/models/vision_transformer.py`.
2. Notebook ou script de treinamento.
3. Checkpoint.
4. Metricas de desempenho e eficiencia.

Validacao esperada:

1. O Transformer usa o mesmo split de teste.
2. As transformacoes sao compativeis com o modelo pre-treinado.
3. Resultado fica pronto para entrar na tabela comparativa.

#### Bloco 10 - Comparacao e Escolha do Modelo Final

Pedido sugerido:

```text
Implemente as tarefas 20, 21 e 22 da ordem recomendada: consolidar metricas, comparar desempenho e eficiencia, e escolher o modelo final.
```

Tarefas cobertas:

20. Consolidar metricas.
21. Comparar desempenho e eficiencia.
22. Escolher modelo final.

O que deve ser entregue:

1. `notebooks/07_comparacao_modelos.ipynb`.
2. Tabela comparativa em CSV.
3. Graficos comparativos.
4. Justificativa do modelo escolhido.
5. Modelo final em `models/exported/`.

Validacao esperada:

1. Todos os modelos aparecem na tabela.
2. O melhor modelo e escolhido com base em F1, recall, AUC e eficiencia.
3. A justificativa esta pronta para o relatorio.

#### Bloco 11 - Prototipos

Pedido sugerido:

```text
Implemente as tarefas 23 e 24 da ordem recomendada: criar prototipo de inferencia em notebook e prototipo Flask.
```

Tarefas cobertas:

23. Criar prototipo em notebook.
24. Criar prototipo Flask.

O que deve ser entregue:

1. `notebooks/08_prototipo_inferencia.ipynb`.
2. `src/app/flask_app.py`.
3. Templates HTML.
4. CSS simples.
5. Aviso de uso academico.

Validacao esperada:

1. Upload ou selecao de imagem funciona.
2. Modelo retorna classe e probabilidade.
3. Interface mostra aviso de que nao e diagnostico medico real.

#### Bloco 12 - Governanca e Fairness

Pedido sugerido:

```text
Implemente a tarefa 25 da ordem recomendada: fazer analise de governanca e fairness com metricas por subgrupo.
```

Tarefa coberta:

25. Fazer analise de governanca e fairness.

O que deve ser entregue:

1. Tabelas de desempenho por sexo, idade e posicao da imagem, se os dados permitirem.
2. Discussao de vieses.
3. Discussao de falsos positivos e falsos negativos.
4. Texto pronto para o relatorio.

Validacao esperada:

1. Analise usa predicoes reais do conjunto de teste.
2. Limitacoes do dataset ficam explicitas.
3. O texto evita qualquer promessa de diagnostico real.

#### Bloco 13 - Relatorio e Revisao Final

Pedido sugerido:

```text
Implemente as tarefas 26 e 27 da ordem recomendada: escrever relatorio tecnico e revisar o projeto contra os criterios da FIAP.
```

Tarefas cobertas:

26. Escrever relatorio tecnico.
27. Revisar checklist dos criterios da FIAP.

O que deve ser entregue:

1. `Docs/relatorio_tecnico.md`.
2. Checklist dos criterios da FIAP.
3. Referencias.
4. Pontos de melhoria ou lacunas restantes.

Validacao esperada:

1. Cada criterio do enunciado aponta para um notebook, codigo, figura ou secao do relatorio.
2. O relatorio explica as decisoes tecnicas.
3. A conclusao indica claramente o modelo final e os limites de uso.

### 12.2 Regra de Continuidade

Sempre que uma nova etapa for solicitada, o arquivo `Docs/plano_implementacao_pbl_fase4.md` deve ser usado como fonte de verdade.

Antes de implementar uma etapa, verificar:

1. Quais tarefas anteriores ja existem no repositorio.
2. Quais arquivos ja foram criados.
3. Se ha mudancas manuais feitas pelo grupo.
4. Se a etapa depende do dataset ja baixado.
5. Se a etapa depende de resultados de treinamento anteriores.

Depois de implementar uma etapa, registrar no `README.md` ou no proprio notebook:

1. O que foi implementado.
2. Como executar.
3. Quais arquivos foram gerados.
4. Quais validacoes foram feitas.
5. Quais proximas tarefas devem ser executadas.

---

## 13. Criterios de Qualidade para Nota Maxima

### 13.1 Qualidade Tecnica

O projeto deve mostrar:

1. Uso correto de dataset publico.
2. Pre-processamento bem documentado.
3. Separacao correta entre treino, validacao e teste.
4. Evitar vazamento de dados por paciente.
5. Comparacao justa entre modelos.
6. Uso de metricas adequadas.
7. Analise de eficiencia, nao apenas desempenho.

### 13.2 Qualidade Cientifica

O projeto deve explicar:

1. Por que o problema foi definido como binario.
2. Por que cardiomegalia foi escolhida.
3. Como o desbalanceamento foi tratado.
4. Por que as metricas escolhidas importam.
5. Por que recall e F1 sao especialmente importantes em saude.
6. Quais limitacoes impedem uso clinico real.

### 13.3 Qualidade de Apresentacao

O projeto deve incluir:

1. Graficos claros.
2. Tabelas comparativas.
3. Matriz de confusao.
4. Curva ROC.
5. Prints do prototipo.
6. Explicacao textual dos resultados.
7. Conclusao objetiva dizendo qual modelo foi escolhido e por que.

### 13.4 Qualidade de Governanca

O projeto deve discutir:

1. Desbalanceamento de classes.
2. Representatividade por sexo, idade e posicao da imagem.
3. Risco de labels ruidosos.
4. Risco de falso negativo.
5. Risco de falso positivo.
6. Necessidade de validacao clinica antes de qualquer uso real.
7. Transparencia na comunicacao dos resultados.

---

## 14. Resultado Esperado

Ao final, o projeto deve entregar:

1. Um pipeline completo de pre-processamento de imagens medicas.
2. Uma CNN propria treinada e avaliada.
3. Uma CNN padrao treinada e avaliada.
4. Modelos de Transfer Learning treinados e avaliados.
5. Um modelo baseado em Transformer treinado e avaliado.
6. Uma tabela comparativa de desempenho e eficiencia.
7. Um prototipo simples para inferencia.
8. Um relatorio tecnico bem explicado.
9. Uma discussao de etica e governanca.
10. Um projeto organizado e facil de apresentar.

O foco nao deve ser apenas obter a maior acuracia. O foco deve ser demonstrar um processo completo, responsavel, reprodutivel e bem explicado de aplicacao de Visao Computacional em imagens medicas.
