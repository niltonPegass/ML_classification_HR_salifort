# HR Salifort Motors - Churn Prediction Project
[ 🇧🇷 Português | 🇺🇸 English ]

---

## 🇧🇷 Apresentação do Projeto (Português)

Este repositório contém a versão modularizada do projeto de Machine Learning para prever o desligamento de funcionários (Churn Prediction) na **Salifort Motors**. O projeto utiliza dados comportamentais e de desempenho dos colaboradores para identificar padrões que indicam maior probabilidade de demissão voluntária, auxiliando na tomada de decisões estratégicas de Recursos Humanos e em programas de retenção de talentos.

### Objetivos
*   **Identificar colaboradores em risco de saída** com base em variáveis de satisfação, avaliação de desempenho, tempo de casa, número de projetos e horas mensais trabalhadas.
*   **Analisar fatores de engajamento e sobrecarga de trabalho** para apoiar políticas internas de gestão de clima e alocação de tarefas.
*   **Treinar e comparar múltiplos modelos de classificação** supervisada (`Regressão Logística`, `Árvore de Decisão`, `Random Forest` e `XGBoost`) para obter a melhor performance preditiva.

### Principais Descobertas
*   O modelo **XGBoost** atingiu a melhor performance com **95,2% de acurácia** e excelente área sob a curva ROC (AUC).
*   **Carga de trabalho extrema** (mais de 250 horas mensais divididas em 6 ou 7 projetos) está diretamente associada a níveis baixíssimos de satisfação (<10%) e risco altíssimo de saída do funcionário.
*   O **nível de satisfação individual** foi identificado como a feature com maior importância (Gini Importance) na previsão de desligamento.
*   Departamentos com maiores taxas de desligamento em risco alto: **Recursos Humanos, Técnico e Marketing**.

---

## 🇺🇸 Project Overview (English)

This repository contains the modularized version of the Machine Learning project to predict employee turnover (Churn Prediction) at **Salifort Motors**. The project leverages employee behavioral and performance data to identify patterns leading to voluntary departure, supporting Human Resources in strategic decision-making and talent retention programs.

### Objectives
*   **Predict employee turnover risk** based on job satisfaction, performance evaluation scores, tenure, project load, and average monthly hours.
*   **Analyze engagement and workload factors** to support climate management and task allocation policies.
*   **Train and evaluate multiple classification models** (`Logistic Regression`, `Decision Tree`, `Random Forest`, and `XGBoost`) to achieve optimal predictive performance.

### Key Insights
*   The **XGBoost** model achieved the best performance with **95.2% accuracy** and a high Area Under the ROC Curve (AUC).
*   **Extreme workloads** (250+ hours/month across 6 or 7 projects) are linked to critically low satisfaction scores (<10%) and extremely high turnover risk.
*   **Individual satisfaction level** is the most significant predictive feature (highest Gini Importance).
*   Departments with the highest high-risk turnover rates: **HR, Technical, and Marketing**.

---

## 📁 Estrutura de Pastas / Project Structure

O projeto foi estruturado seguindo as melhores práticas de desenvolvimento de software e ciência de dados para repositórios no GitHub:

```
google_coursera_ml_class/
│
├── README.md                          # Documentação principal (este arquivo)
├── HR Salifort Motors.ipynb           # Notebook com a descrição do desenvolvimento e com a documentação
├── requirements.txt                   # Dependências e bibliotecas do projeto
├── .gitignore                         # Regras de arquivos ignorados no Git
├── CHANGELOG.md                       # Log detalhado das refatorações efetuadas
├── main.py                            # Script principal/orquestrador do pipeline
│
├── data/
│   └── processed/
│       └── HR_Sailfort_dataset_Processed.xls  # Dataset tratado (formato CSV)
│
├── notebooks/
│   └── HR_Salifort_Motors_Churn_Prediction.ipynb  # Notebook original do Kaggle
│
├── src/                               # Módulos de código-fonte
│   ├── __init__.py                    # Inicialização do pacote python
│   ├── config.py                      # Constantes e caminhos de arquivos
│   ├── data_loader.py                 # Funções de carregamento e overview dos dados
│   ├── eda.py                         # Gráficos de Análise Exploratória (EDA)
│   ├── model_training.py              # Split, escala, tuning e treinamento de modelos
│   ├── model_evaluation.py            # Avaliação de métricas, ROC e curvas de aprendizado
│   └── insights.py                    # Segmentação de risco e análise departamental
│
├── models/                            # Pasta para salvamento dos modelos serializados (.pkl)
└── outputs/
    └── figures/                       # Pasta onde os gráficos gerados são salvos automaticamente
```

---

## 🚀 Como Executar o Projeto / How to Run the Project

### Pré-requisitos
Certifique-se de possuir o Python 3.8+ instalado em sua máquina.

### 1. Clonar o repositório
```bash
git clone https://github.com/niltonpegass/ML_classification_HR_salifort.git
cd ML_classification_HR_salifort
```

### 2. Instalar as dependências
Recomenda-se o uso de um ambiente virtual (venv):
```bash
# Criar e ativar o ambiente virtual (Windows)
python -m venv venv
venv\Scripts\activate

# Criar e ativar o ambiente virtual (Linux/macOS)
python3 -m venv venv
source venv/bin/activate

# Instalar pacotes
pip install -r requirements.txt
```

### 3. Rodar o pipeline completo
Execute o orquestrador `main.py`. O pipeline irá realizar a carga de dados, gerar as visualizações da análise exploratória (EDA), realizar a divisão de treino/teste, treinar os modelos comparativos otimizados com GridSearchCV, salvar os arquivos `.pkl` em `models/` e gerar os relatórios de insights e gráficos finais na pasta `outputs/figures/`.

```bash
python main.py
```

---

## 🛠️ Tecnologias Utilizadas / Tech Stack
*   **Language:** Python 3.11
*   **Data Manipulation:** Pandas, NumPy
*   **Visualization:** Matplotlib, Seaborn
*   **Machine Learning:** Scikit-learn, XGBoost
*   **Hyperparameter Tuning:** GridSearchCV, StratifiedKFold
