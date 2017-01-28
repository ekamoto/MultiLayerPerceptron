# Alunos: Leandro Shindi Ekamoto, Diego Takaki
# Leia readme
#

import math
import random
import numpy as np

def criar_linha():
    print "-"*80

def rand(a, b):
    return (b-a) * random.random() + a

# nossa funcao sigmoide - gera graficos em forma de S
# funcao tangente hiperbolica
def funcao_ativacao_tang_hip(x):
    return math.tanh(x)

# derivada da tangente hiperbolica
def derivada_funcao_ativacao(x):
    t = funcao_ativacao_tang_hip(x)
    return 1 - t**2

# Normal logistic function.
# saida em [0, 1]
def funcao_ativacao_log(x):
    return 1 / ( 1 + math.exp(-x))

# derivada da funcao
def derivada_funcao_ativacao_log(x):
    ret = -1*math.log(x) * (1 - math.log(x))
    return ret

# Logistic function with output in [-1, 1].
def funcao_ativacao_log2(x):
    return 1 - 2 * log(x)

# derivada da funcao
def derivada_funcao_ativacao_log2(x):
    return  -5*math.log(x) * (1 - math.log(x))

class RedeNeural:
    def __init__(self, nos_entrada, nos_ocultos, nos_saida):
        # camada de entrada
        self.nos_entrada = nos_entrada + 1 # +1 por causa do no do bias
        # camada oculta
        self.nos_ocultos = nos_ocultos
        # camada de saida
        self.nos_saida = nos_saida
        # hisamoto
        # quantidade maxima de max_interacoes
        self.max_interacoes = 10
        # taxa de aprendizado
        self.taxa_aprendizado = 0.011
        # momentum Normalmente eh ajustada entre 0.5 e 0.9
        self.momentum = 0.1
        self.teste = 0

        # activations for nodes
        # cria uma matriz, preenchida com uns, de uma linha pela quantidade de nos
        self.ativacao_entrada = np.ones(self.nos_entrada)
        self.ativacao_ocultos = np.ones(self.nos_ocultos)
        self.ativacao_saida = np.ones(self.nos_saida)

        # contem os resultados das ativacoes de saida
        self.resultados_ativacao_saida = np.ones(self.nos_saida)

        # criar a matriz de pesos, preenchidas com zeros
        self.wi = np.zeros((self.nos_entrada, self.nos_ocultos))
        self.wo = np.zeros((self.nos_ocultos, self.nos_saida))

        # adicionar os valores dos pesos
        # vetor de pesos da camada de entrada - intermediaria
        for i in range(self.nos_entrada):
            for j in range(self.nos_ocultos):
                self.wi[i][j] = 0.01

        # vetor de pesos da camada intermediaria - saida
        for j in range(self.nos_ocultos):
            for k in range(self.nos_saida):
                self.wo[j][k] = -0.01

        # last change in weights for momentum
        self.ci = np.zeros((self.nos_entrada, self.nos_ocultos))
        self.co = np.zeros((self.nos_ocultos, self.nos_saida))

    def fase_forward(self, entradas):

        if(self.teste):
            print "Entradas:"
            print entradas

        if(self.teste):
            print "Nos entrada=" + str(self.nos_entrada)

        for i in range(self.nos_entrada - 1):
            self.ativacao_entrada[i] = entradas[i]
            if(self.teste):
                print "Valor Nos Entrada:" + str(self.ativacao_entrada[i])

        if(self.teste):
            print "Nos ocultos=" + str(self.nos_ocultos)

        # calcula as ativacoes dos neuronios da camada escondida
        for j in range(self.nos_ocultos):

            soma = 0.0
            for i in range(self.nos_entrada):
                soma = soma + self.ativacao_entrada[i] * self.wi[i][j]

            if(self.teste):
                print "Soma Nos ocultos=" + str(soma)

            self.ativacao_ocultos[j] = funcao_ativacao_log(soma)
            if(self.teste):
                print "Valor Nos Ocultos:" + str(self.ativacao_ocultos[j])

        # calcula as ativacoes dos neuronios da camada de saida
        # Note que as saidas dos neuronios da camada oculta fazem o papel de entrada
        # para os neuronios da camada de saida.
        if(self.teste):
            print "Nos saida=" + str(self.nos_saida)

        for j in range(self.nos_saida):
            soma = 0.0
            for i in range(self.nos_ocultos):
                soma = soma + self.ativacao_ocultos[i] * self.wo[i][j]

            if(self.teste):
                print "Soma_saida:" + str(soma)

            self.ativacao_saida[j] = funcao_ativacao_log(soma)

        if(self.teste):
            print "Saida ativacao:"
            print self.ativacao_saida

        return self.ativacao_saida

    def fase_backward(self, saidas_desejadas):

        # calcular os gradientes locais dos neuronios da camada de saida
        output_deltas = np.zeros(self.nos_saida)
        erro = 0.0
        for i in range(self.nos_saida):

            print "Saida Desejada:" + str(saidas_desejadas[i])
            print "Ativacao saida:" + str(self.ativacao_saida[i])

            print str(saidas_desejadas[i]) + " - " +  str(self.ativacao_saida[i])
            erro = np.float64(saidas_desejadas[i]) - np.float64(self.ativacao_saida[i])
            print "Erro: " + str(erro)

            output_deltas[i] = derivada_funcao_ativacao_log(self.ativacao_saida[i]) * erro

        # calcular os gradientes locais dos neuronios da camada escondida
        hidden_deltas = np.zeros(self.nos_ocultos)
        for i in range(self.nos_ocultos):
            erro = 0.0
            for j in range(self.nos_saida):
                erro = erro + output_deltas[j] * self.wo[i][j]
            hidden_deltas[i] = derivada_funcao_ativacao_log(self.ativacao_ocultos[i]) * erro

        # a partir da ultima camada ate a camada de entrada
        # os nos da camada atual ajustam seus pesos de forma a reduzir seus erros
        for i in range(self.nos_ocultos):
            for j in range(self.nos_saida):
                change = output_deltas[j] * self.ativacao_ocultos[i]
                self.wo[i][j] = self.wo[i][j] + (self.taxa_aprendizado * change)
                self.co[i][j] = change

        # atualizar os pesos da primeira camada
        for i in range(self.nos_entrada):
            for j in range(self.nos_ocultos):
                change = hidden_deltas[j] * self.ativacao_entrada[i]
                self.wi[i][j] = self.wi[i][j] + (self.taxa_aprendizado * change)
                self.ci[i][j] = change

        # calcula erro
        erro = 0.0
        for i in range(len(saidas_desejadas)):
            erro = erro + 0.5 * (saidas_desejadas[i] - self.ativacao_saida[i]) ** 2
        return erro

    def test(self, entradas_saidas):

        self.teste = 0
        for p in entradas_saidas:
            array = self.fase_forward(p[0])
            print('Saida encontrada/fase forward: ' + str(array[0]))

    def fit(self, entradas_saidas):

        for i in range(self.max_interacoes):
            erro = 0.0
            l = 0
            for p in entradas_saidas:
                entradas = p[0]
                saidas_desejadas = p[1]

                self.fase_forward(entradas)
                erro = erro + self.fase_backward(saidas_desejadas)

            if i % 100 == 0:
                print "Erro = %2.3f"%erro
