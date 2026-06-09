# Governanca, Fairness e Limitacoes

A analise foi realizada no conjunto de teste para o modelo final `Vision Transformer ViT`. Foram avaliados genero, faixa etaria e posicao da imagem para qualquer patologia selecionada e para cada patologia individual.

## Representatividade

Subgrupos com menos de 10 amostras foram marcados com `skipped_low_sample`, pois metricas com poucas observacoes podem oscilar muito e nao devem sustentar conclusoes fortes.

## Metricas por subgrupo

Foram calculadas accuracy, precision, recall, F1-score, AUC-ROC, specificity, false positive rate e false negative rate. O recall e o false negative rate merecem destaque porque falsos negativos em um contexto de saude podem atrasar investigacao clinica.

## Maiores gaps observados

      target     group_col                   metric_gap      gap
Pneumothorax     age_group                      fnr_gap 0.795455
Pneumothorax     age_group equal_opportunity_recall_gap 0.795455
Pneumothorax     age_group                       f1_gap 0.654206
Infiltration View Position                      fnr_gap 0.619340
Infiltration View Position equal_opportunity_recall_gap 0.619340
Pneumothorax     age_group                precision_gap 0.555556
Infiltration View Position       demographic_parity_gap 0.545593
Infiltration     age_group equal_opportunity_recall_gap 0.511905
Infiltration     age_group                      fnr_gap 0.511905
Infiltration View Position                      fpr_gap 0.471406

## Limites de uso

Este prototipo nao substitui avaliacao medica. O dataset NIH ChestX-ray14 possui labels derivados de laudos e pode conter ruido de anotacao. Antes de uso real seriam necessarias validacao clinica externa, revisao por especialistas, auditoria de vies, governanca de dados e monitoramento continuo.
