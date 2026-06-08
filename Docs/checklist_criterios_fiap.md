# Checklist de Criterios FIAP - CardioIA Vision

## Criterios Principais

Notebook principal de entrega:

- `notebooks/CardioIA_Vision_PBL_Fase4.ipynb`

Os demais notebooks sao modulares e servem como apoio tecnico para executar etapas especificas.

| Criterio | Pontos | Status | Evidencias |
|---|---:|---|---|
| Pipeline de pre-processamento implementado | 3 | Implementado | `src/preprocessing.py`, `src/datasets.py`, `notebooks/03_validacao_preprocessamento_dataloaders.ipynb` |
| Treinamento e avaliacao de CNN do zero | 2 | Implementado | `src/models/custom_cnn.py`, `notebooks/05_treinamento_cnn_propria.ipynb` |
| Transfer Learning funcional | 2 | Implementado | `src/models/transfer_learning.py`, `notebooks/07_treinamento_transfer_learning.ipynb` |
| Prototipo simples | 2 | Implementado | `notebooks/10_prototipo_inferencia.ipynb`, `src/app/flask_app.py` |
| Documentacao clara | 1 | Implementado | `README.md`, `Docs/plano_implementacao_pbl_fase4.md`, `Docs/relatorio_tecnico.md` |

## Pontos Extras e Ir Alem

| Item | Status | Evidencias |
|---|---|---|
| Trabalho em grupo | A preencher | Informar integrantes no README e no relatorio final |
| Etica e governanca | Implementado | `src/training/fairness.py`, `notebooks/11_governanca_fairness.ipynb` |
| Metricas por subgrupo | Implementado | `reports/metricas/fairness_metricas_por_subgrupo.csv` apos execucao do notebook |
| Integracao mobile | Nao implementado | Opcional para etapa futura |

## Ordem Recomendada de Execucao Para Gerar Resultados

1. `notebooks/00_download_dataset_kaggle.ipynb`
2. `notebooks/01_eda_dataset.ipynb`
3. `notebooks/02_preprocessamento_splits.ipynb`
4. `notebooks/03_validacao_preprocessamento_dataloaders.ipynb`
5. `notebooks/04_validacao_pipeline_treino_avaliacao.ipynb`
6. `notebooks/05_treinamento_cnn_propria.ipynb`
7. `notebooks/06_treinamento_cnn_padrao.ipynb`
8. `notebooks/07_treinamento_transfer_learning.ipynb`
9. `notebooks/08_treinamento_transformer_vision.ipynb`
10. `notebooks/09_comparacao_modelos.ipynb`
11. `notebooks/10_prototipo_inferencia.ipynb`
12. `notebooks/11_governanca_fairness.ipynb`

## Pendencias Antes da Entrega Final

1. Executar todos os notebooks com o dataset real.
2. Preencher resultados numericos em `Docs/relatorio_tecnico.md`.
3. Inserir nomes dos integrantes.
4. Conferir figuras em `reports/figuras/`.
5. Conferir tabelas em `reports/tabelas/` e `reports/metricas/`.
6. Exportar ou converter o relatorio para PDF, se solicitado pela FIAP.
