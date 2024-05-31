import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Bora ler os dados do CSV 🐧
df = pd.read_csv('palmerpenguins.csv')

# Mapeando os valores das colunas categóricas pra números inteiros 🤓
island_map = {'Biscoe': 0, 'Dream': 1, 'Torgersen': 2}
sex_map = {'FEMALE': 0, 'MALE': 1}
species_map = {'Adelie': 0, 'Chinstrap': 1, 'Gentoo': 2}

# Trocando as strings por números nas colunas 🔄
df['island'] = df['island'].map(island_map)
df['sex'] = df['sex'].map(sex_map)
df['species'] = df['species'].map(species_map)

# Convertendo as colunas para o tipo int64 💪
df['island'] = df['island'].astype('int64')
df['sex'] = df['sex'].astype('int64')
df['species'] = df['species'].astype('int64')

# Reorganizando as colunas, que nem no PDF 📄
column_order = ['island', 'sex', 'culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g', 'species']
df = df[column_order]

# Solicitando dados de entrada ao usuário
island_input = input("Digite o número da ilha (0 para Biscoe, 1 para Dream, 2 para Torgersen): ")
sex_input = input("Digite o número do sexo (0 para Fêmea, 1 para Macho): ")
culmen_length_mm = float(input("Digite o comprimento do bico (mm): "))
culmen_depth_mm = float(input("Digite a profundidade do bico (mm): "))
flipper_length_mm = float(input("Digite o comprimento da nadeira (mm): "))
body_mass_g = float(input("Digite a massa corporal (g): "))

# Visualizando os dados com gráficos de dispersão 📈
colors = {0: 'blue', 1: 'orange', 2: 'green'}  # Corrigido para usar os valores numéricos de species_map

# Gráfico 1: Comprimento do bico vs. Profundidade do bico
plt.figure(figsize=(10, 6))
for species, group in df.groupby('species'):
    plt.scatter(group['culmen_length_mm'], group['culmen_depth_mm'], c=colors[species], label=species)
plt.scatter(culmen_length_mm, culmen_depth_mm, c='red', marker='x', label='Pinguim a ser previsto')
plt.xlabel('Comprimento do Bico (mm)')
plt.ylabel('Profundidade do Bico (mm)')
plt.title('Comprimento vs. Profundidade do Bico por Espécie')
plt.legend()
plt.show()

# Gráfico 2: Comprimento do bico vs. Comprimento da nadadeira
plt.figure(figsize=(10, 6))
for species, group in df.groupby('species'):
    plt.scatter(group['culmen_length_mm'], group['flipper_length_mm'], c=colors[species], label=species)
plt.scatter(culmen_length_mm, flipper_length_mm, c='red', marker='x', label='Pinguim a ser previsto')
plt.xlabel('Comprimento do Bico (mm)')
plt.ylabel('Comprimento da Nadeira (mm)')
plt.title('Comprimento do Bico vs. Comprimento da Nadeira por Espécie')
plt.legend()
plt.show()

# Gráfico 3: Comprimento do bico vs. Massa corporal
plt.figure(figsize=(10, 6))
for species, group in df.groupby('species'):
    plt.scatter(group['culmen_length_mm'], group['body_mass_g'], c=colors[species], label=species)
plt.scatter(culmen_length_mm, body_mass_g, c='red', marker='x', label='Pinguim a ser previsto')
plt.xlabel('Comprimento do Bico (mm)')
plt.ylabel('Massa Corporal (g)')
plt.title('Comprimento do Bico vs. Massa Corporal por Espécie')
plt.legend()
plt.show()

# Separando os atributos (X) da classe (y) 🎯
X = df.drop('species', axis=1)
y = df['species']

# Dividindo os dados em treino e teste ⚖️
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criando os modelos de ML 😎
clf_dt = DecisionTreeClassifier()
clf_knn = KNeighborsClassifier(n_neighbors=3)
clf_mlp = MLPClassifier(random_state=42)
clf_nb = GaussianNB()

# Treinando os modelos 💪
clf_dt.fit(X_train, y_train)
clf_knn.fit(X_train, y_train)
clf_mlp.fit(X_train, y_train)
clf_nb.fit(X_train, y_train)

# Função pra prever a espécie do pinguim 🐧
def predict_penguin_with_input(island, sex, culmen_length_mm, culmen_depth_mm, flipper_length_mm, body_mass_g):
    # Fazendo a previsão 🔮
    prediction = clf_dt.predict([[island, sex, culmen_length_mm, culmen_depth_mm, flipper_length_mm, body_mass_g]])[0]

    # Traduzindo o número da previsão para o nome da espécie 🧐
    inverse_species_map = {v: k for k, v in species_map.items()}
    predicted_species = inverse_species_map[prediction]

    # Mostrando o resultado 🚀
    print(f"A espécie prevista é: {predicted_species}")

# Chamando a função para prever a espécie
predict_penguin_with_input(int(island_input), int(sex_input), culmen_length_mm, culmen_depth_mm, flipper_length_mm, body_mass_g)
