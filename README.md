# Churn Prediction — Telco Customer Dataset

Modelo preditivo para identificação de clientes com alto risco de cancelamento, com foco em métricas de negócio e interpretabilidade dos resultados.

---

## Sobre o projeto

A taxa de churn é uma das métricas mais críticas em empresas de serviços. Identificar, com antecedência, quais clientes tem maior probabilidade de cancelar permite ações direcionadas de retenção — mais eficientes e menos custosas do que campanhas genéricas.

Este projeto cobre o ciclo completo de um modelo de classificação: análise exploratória, engenharia de features, comparação de algoritmos e interpretação dos resultados via SHAP values.

**Dataset:** IBM Telco Customer Churn — 7.043 clientes, 21 variáveis  
**Fonte:** [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

---

## Estrutura

```
churn-prediction/
├── churn_model.ipynb       # Notebook principal
├── plots/                  # Gráficos gerados na execução
│   ├── churn_distribution.png
│   ├── numeric_distributions.png
│   ├── categorical_churn_rates.png
│   ├── roc_pr_curves.png
│   ├── confusion_matrix.png
│   ├── shap_summary.png
│   └── shap_feature_importance.png
├── requirements.txt
└── README.md
```

---

## Metodologia

**EDA** — distribuição do target, análise por variável numérica e categórica, identificação de padrões de comportamento antes do cancelamento.

**Feature Engineering** — correção de tipo em `TotalCharges`, criação de features derivadas (`avg_monthly_spend`, `services_count`, `is_new_customer`, `is_month_to_month`), encoding de variáveis categóricas.

**Modelagem** — três algoritmos comparados com validação cruzada estratificada (5 folds):
- Logistic Regression
- Random Forest
- XGBoost

**Avaliação** — AUC-ROC, Average Precision, curvas ROC e Precision-Recall, matriz de confusão. A escolha dessas métricas leva em conta o desbalanceamento do dataset (~26% de churn).

**Interpretabilidade** — SHAP values para identificar os fatores com maior influência individual nas predições do melhor modelo.

---

## Principais achados

O tipo de contrato é o driver mais forte: clientes com plano mensal têm taxa de churn cerca de 14x maior do que clientes com contratos de dois anos. O tempo de relacionamento também tem peso relevante — os primeiros meses concentram a maior parte dos cancelamentos.

No lado financeiro, mensalidades mais altas estão associadas a maior risco, enquanto clientes com mais serviços ativos tendem a cancelar menos. Ausência de suporte técnico e segurança online aparece consistentemente entre os fatores que elevam o risco.

---

## Dependências

```
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
shap
jupyter
```

```bash
pip install -r requirements.txt
```

---

## Como executar

```bash
git clone https://github.com/samelaqueiroz/churn-prediction.git
cd churn-prediction
pip install -r requirements.txt
jupyter notebook churn_model.ipynb
```

O dataset precisa ser baixado do Kaggle (`WA_Fn-UseC_-Telco-Customer-Churn.csv`) e colocado na raiz do projeto antes de executar.

---

## Tecnologias

Python · Pandas · Scikit-learn · XGBoost · SHAP · Matplotlib · Seaborn
