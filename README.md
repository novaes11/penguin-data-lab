# 🐧 Penguin Data Lab

Este repositório contém um **projeto de análise exploratória de dados e classificação de espécies de pinguins**, desenvolvido como parte das atividades do **Grupo de Inteligência Artificial da faculdade**.  

O objetivo foi praticar:
- Limpeza e preparação de dados.  
- Análise estatística e visualização.  
- Identificação de outliers, dados faltantes e inconsistentes.  
- Treinamento de um modelo de **árvore de decisão** para classificar espécies de pinguins.  

---

## 📊 Conjunto de Dados
O dataset utilizado é o **Palmer Penguins**, disponível diretamente no **Seaborn** (`sns.load_dataset("penguins")`).  
Ele contém medidas físicas de três espécies de pinguins, como comprimento do bico, profundidade, comprimento da nadadeira e massa corporal.

---

## 🛠️ Tecnologias Utilizadas
- [Python 3.12](https://www.python.org/)  
- [Pandas](https://pandas.pydata.org/)  
- [NumPy](https://numpy.org/)  
- [Matplotlib](https://matplotlib.org/)  
- [Seaborn](https://seaborn.pydata.org/)  
- [Scikit-learn](https://scikit-learn.org/)  

---

## 📂 Estrutura do Projeto
  ```bash 
  penguin-data-lab/
  │── data_analysis.py # Script principal de análise e classificação
  │── requirements.txt # Dependências do projeto
  │── README.md # Documentação
  ```
---

## ▶️ Como Executar

1. Clone este repositório:
     ```bash
     git clone https://github.com/seu-usuario/penguin-data-lab.git
     cd penguin-data-lab
     ```
2. Crie e ative um ambiente virtual:
    ```bash
      python -m venv .venv
      .\.venv\Scripts\activate   # Windows
      source .venv/bin/activate # Linux/Mac
    ```
3. Instale as dependências:
    ```bash
    pip install -r requirements.txt
    ```
4. Execute o script:
    ```bash
    python data_analysis.py
    ```
---
## 📈 Resultados

- Identificação de **dados faltantes** e **linhas duplicadas**.  
- Visualizações com **boxplots** e **pairplots** para detectar outliers e padrões.  
- Treinamento de uma **árvore de decisão** para classificar espécies.  
- Obtenção da **acurácia do modelo** no conjunto de teste.  
- Visualização da **árvore de decisão treinada**.  

---

## 🚀 Aprendizado

Esse projeto foi desenvolvido para **praticar análise de dados e machine learning** em Python dentro do grupo de IA da faculdade.  
Ele serviu como base para entender o fluxo completo: **carregar dados → limpar → explorar → visualizar → modelar → avaliar**.  

