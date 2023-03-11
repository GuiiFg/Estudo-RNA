import pandas as pd

from src.perceptron.Perceprton import Neuron, ActivationFunctions

databaseTreino = pd.read_excel('database/Dados.ods', sheet_name='Treinamento')
databaseTeste = pd.read_excel('database/Dados.ods', sheet_name='Teste')

train_x = databaseTreino.drop(columns=['d'])
train_y = databaseTreino.drop(columns=['x1', 'x2', 'x3'])

perceptron = Neuron(fg = ActivationFunctions.SigmoidBipolar, teta= -1)

perceptron.printW()

perceptron.train(train_x, train_y, 999, lerning_rate= 0.01, verbose= True)

perceptron.predict(databaseTeste)

