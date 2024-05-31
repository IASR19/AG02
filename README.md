## Classificador de Espécies de Pinguins 🐧

#### Este projeto tem como objetivo construir um modelo de aprendizado de máquina capaz de classificar a espécie de um pinguim com base em suas características físicas. Para isso, utilizamos o conjunto de dados Palmer Penguins, que contém informações sobre pinguins de três espécies diferentes (Adelie, Chinstrap e Gentoo) coletadas em três ilhas da Antártica.

### Funcionalidades
- Leitura e pré-processamento dos dados: O código carrega os dados do arquivo CSV, converte as variáveis categóricas em numéricas e reorganiza as colunas.

- Visualização dos dados: Plota gráficos de dispersão para explorar as relações entre as características dos pinguins e suas espécies.

- Treinamento e avaliação de modelos: Quatro modelos de classificação são treinados e avaliados:

### Árvore de Decisão
- k Vizinhos Mais Próximos (k-NN)
- Perceptron Multicamadas (MLP)
- Naive Bayes
- Predição da espécie: Uma função permite que você insira as características de um pinguim e obtenha a previsão da espécie com base no modelo de Árvore de Decisão.

### Como usar

#### Instalação:

- Certifique-se de ter o Python instalado em seu sistema.
- Instale as bibliotecas necessárias:

``
pip install pandas scikit-learn matplotlib
``

#### Dados:

Baixe o conjunto de dados Palmer Penguins em formato CSV de https://github.com/allisonhorst/palmerpenguins.

Salve o arquivo CSV (palmerpenguins.csv) na mesma pasta do código Python.

### Execução:

Execute o código Python:

``
python AG2.py
``


### Visualização:

Os gráficos de dispersão serão exibidos automaticamente após a execução do código.


### Predição:

Ao final da execução, você será solicitado a inserir as características de um pinguim.

Siga as instruções e insira os valores para:

- Ilha (0: Biscoe, 1: Dream, 2: Torgersen)
- Sexo (0: Fêmea, 1: Macho)
- Comprimento do bico (mm)
- Profundidade do bico (mm)
- Comprimento da nadadeira (mm)
- Massa corporal (g)

O código irá prever e exibir a espécie do pinguim.

#### Por exemplo, se você fornecer dados como:

- Ilha: Torgersen (código 2)
- Sexo: Macho (código 1)
- Comprimento do bico: 45.0 mm
- Profundidade do bico: 19.0 mm
- Comprimento da nadadeira: 200.0 mm
- Massa corporal: 4000.0 g


### Observações
O modelo de Árvore de Decisão foi usado para fazer as predições. 

(Podemos alterar o modelo na função predict_penguin_with_input para usar outro modelo)

Os resultados da avaliação dos modelos (acurácia e relatório de classificação) são exibidos no console.

### Grupo:

    Itamar Augusto Silva Ribeiro - 91 - GES
    Eduardo Karpfenstein - 77 - GES
