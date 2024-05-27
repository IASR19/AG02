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

# Visualizando os dados com gráficos de dispersão 📈
colors = {'Adelie': 'blue', 'Chinstrap': 'orange', 'Gentoo': 'green'}

# Gráfico 1: Comprimento do bico vs. Profundidade do bico
plt.figure(figsize=(10, 6))
for species, group in df.groupby('species'):
    plt.scatter(group['culmen_length_mm'], group['culmen_depth_mm'], c=colors[species_map[species]], label=species_map[species])
plt.xlabel('Comprimento do Bico (mm)')
plt.ylabel('Profundidade do Bico (mm)')
plt.title('Comprimento vs. Profundidade do Bico por Espécie')
plt.legend()
plt.show()

# Gráfico 2: Comprimento do bico vs. Comprimento da nadadeira
plt.figure(figsize=(10, 6))
for species, group in df.groupby('species'):
    plt.scatter(group['culmen_length_mm'], group['flipper_length_mm'], c=colors[species_map[species]], label=species_map[species])
plt.xlabel('Comprimento do Bico (mm)')
plt.ylabel('Comprimento da Nadeira (mm)')
plt.title('Comprimento do Bico vs. Comprimento da Nadeira por Espécie')
plt.legend()
plt.show()

# Gráfico 3: Comprimento do bico vs. Massa corporal
plt.figure(figsize=(10, 6))
for species, group in df.groupby('species'):
    plt.scatter(group['culmen_length_mm'], group['body_mass_g'], c=colors[species_map[species]], label=species_map[species])
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

# Fazendo as predições 🔮
y_pred_dt = clf_dt.predict(X_test)
y_pred_knn = clf_knn.predict(X_test)
y_pred_mlp = clf_mlp.predict(X_test)
y_pred_nb = clf_nb.predict(X_test)

# Avaliando os modelos e vendo qual mandou melhor 🏆
print("Árvore de Decisão:")
print(f'Acurácia: {accuracy_score(y_test, y_pred_dt):.2f}')
print("Relatório de Classificação:\n", classification_report(y_test, y_pred_dt))

print("\nk Vizinhos Mais Próximos:")
print(f'Acurácia: {accuracy_score(y_test, y_pred_knn):.2f}')
print("Relatório de Classificação:\n", classification_report(y_test, y_pred_knn))

print("\nPerceptron Multicamadas:")
print(f'Acurácia: {accuracy_score(y_test, y_pred_mlp):.2f}')
print("Relatório de Classificação:\n", classification_report(y_test, y_pred_mlp))

print("\nNaive Bayes:")
print(f'Acurácia: {accuracy_score(y_test, y_pred_nb):.2f}')
print("Relatório de Classificação:\n", classification_report(y_test, y_pred_nb))

# Função pra prever a espécie do pinguim 🐧
def predict_penguin_with_input(island, sex, culmen_length_mm, culmen_depth_mm, flipper_length_mm, body_mass_g):
    # Criando um DataFrame com os dados de entrada 📝
    new_data = pd.DataFrame({
        'island': [island],
        'sex': [sex],
        'culmen_length_mm': [culmen_length_mm],
        'culmen_depth_mm': [culmen_depth_mm],
        'flipper_length_mm': [flipper_length_mm],
        'body_mass_g': [body_mass_g]
    })

    # Fazendo a previsão 🔮
    prediction = clf_dt.predict(new_data)[0]

    # Traduzindo o número da previsão para o nome da espécie 🧐
    inverse_species_map = {v: k for k, v in species_map.items()}
    predicted_species = inverse_species_map[prediction]

    # Mostrando o resultado 🚀
    print(f"A espécie prevista é: {predicted_species}")

# Exemplo de uso da função com dados de um pinguim 🐧
predict_penguin_with_input(2, 1, 45.0, 19.0, 200.0, 4000.0)
