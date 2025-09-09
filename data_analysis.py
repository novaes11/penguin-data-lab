#Carregando conjunto de dados tips.
import seaborn as sns
pinguins = sns.load_dataset('penguins')
print(pinguins)


#Renomear a classificação de dados
import pandas as pd
#pinguins.rename(columns={'species' : 'espécies', 'island' : 'ilha', 'bill_length_mm' : 'comprimento do bico', 'bill_depth_mm': 'profundidade do bico (mm)', 'flipper_length_mm': 'omprimento da nadadeira (mm)', 'body_mass_g' : 'massa_corporal_(g)', 'sex' : 'sexo'}, inplace = True)
print(pinguins)


#Verificação de dados faltantes
import pandas as pd
#pinguins.iloc[0] = pd.NA faz com que a primeira linha dos dados seja apagada.
#pinguins.at[0,'island' ] = pd.NA faz com que na linha 0, coluna 'island' o dado seja apagado.

print(pinguins)
pinguins.dropna()
print('-----------------------------------------------------')
dadosFaltantes = pinguins.isnull().sum() #serve para mostrar uma tabela onde falará se há dados faltantes.
print(dadosFaltantes)


# Identificação de Dados Ruidosos
# Dados ruidosos são aqueles que não seguem a distribuição esperada dos dados, como valores que são extremamente altos ou baixos comparados ao resto do conjunto de dados. Existem várias técnicas para identificar dados ruidosos:

# Análise Estatística Descritiva: Use o método describe() para obter estatísticas resumidas que podem ajudar a identificar valores extremos.
# Visualização de Dados: Gráficos como boxplots, histogramas e scatter plots podem ajudar a visualizar a distribuição dos dados e identificar dados ruídosos (outliers).

#Describe para verificar dados estatísticos

pinguins.describe()

# Boxplot para visualizar outliers
import matplotlib.pyplot as plt

pinguins.boxplot()
plt.show()

#Verificação de Dados Redundantes / Linhas duplicadas
#Para encontrar e tratar dados redundantes, geralmente buscamos por linhas duplicadas. Essas instâncias podem ocorrer devido a erros de entrada de dados ou ao combinar múltiplos conjuntos de dados.
#Identificar Redundâncias: Verifique se existem linhas duplicadas no seu DataFrame. O método duplicated() retorna uma série booleana, onde True indica uma linha duplicada.

duplicadas = pinguins.duplicated()
print(duplicadas.any())  # Verifica se há alguma linha duplicada

qtd_duplicadas = duplicadas.sum()
print(f"Existem {qtd_duplicadas} linhas duplicadas.")

print(pinguins[duplicadas])
print('-----------------------------------------------------')


#Verificação de Dados Inconsistentes
#Dados inconsistentes podem ser categorias mal escritas, valores impossíveis (como um comprimento de bico negativo). Para verificar inconsistências:
#Valores Únicos e Categorias: Verifique os valores únicos para colunas categóricas usando unique() para identificar categorias mal escritas ou inconsistentes.
#Verificação de Valores Impossíveis: Certifique-se de que os valores numéricos estão dentro de intervalos esperados. Por exemplo, valores negativos para medidas físicas podem ser impossíveis.

sns.pairplot(pinguins, hue='species', height=2)

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import seaborn as sns

#Carregando conjunto de dados penguins
pinguins = pinguins.dropna()
print(pinguins)

print('-----------------------------------------------------')

x = pinguins[["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]]# Separa os atributos (X)
y = pinguins["species"]# Separa o alvo (y) - a espécie
print(x)
print('-----------------------------------------------------')
print(y)
print('-----------------------------------------------------')
y_names = y.unique() # Obtém os nomes únicos das espécies
print(y_names)


from sklearn.model_selection import train_test_split
#Obtém um embaralhamento (shuffle) dos dados aleatóriamente para definirmos o conjunto de dados para treinamento
conjunto_dados = np.random.permutation(len(x))

#x_train = x.iloc[conjunto_dados[:250]]  #x_train: O array de caracteristicas da base de dados separados para treino.
#x_test = x.iloc[conjunto_dados[251:]] #x_test: O array de caracteristicas da base de dados separados para teste.

#y_train = y.iloc[conjunto_dados[:250]]   #y_train: O array de rotulo(s) da base de dados separados para treino.
#y_test = y.iloc[conjunto_dados[251:]] #y_test: O array de rotulo(s) da base de dados separados para teste.

#classificação usando uma árvore de decisão
clf = DecisionTreeClassifier()

corte_automatico = int(0.8 * len(x)) # CORTE AUTOMATICO para separar dados de treino e dados de teste.

x_train = x.iloc[conjunto_dados[:corte_automatico]]
y_train = y.iloc[conjunto_dados[:corte_automatico]]

x_test = x.iloc[conjunto_dados[corte_automatico:]]
y_test = y.iloc[conjunto_dados[corte_automatico:]]

#treinando e ajustando (fitting) o classificador com Qo conjunto de treinamento
clf.fit(x_train, y_train)


#previsões no conjunto de dados de teste
#y^ = predito (usualmente utilizado inves de "prev")
prev = clf.predict(x_test)

print(prev) #rótulos que foram previstos
print('--------------------------------------------------------------')
print(y_test) #rótulos reais
precisao = (accuracy_score(prev, y_test))*100
print('Porcentagem de precisão: {:.2f}%'.format(precisao)) #acurácia de predição
print(y_names)

from sklearn.tree import plot_tree
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt


plt.rcParams["figure.figsize"] = (20,15)
plt.figure()
clf = DecisionTreeClassifier().fit(pinguins[["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]], pinguins["species"])
plot_tree(clf, filled=True)
plt.title("Decision tree trained on all the penguins features")
plt.show()