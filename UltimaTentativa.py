import cv2
import numpy as np
import json
import os
from sklearn import svm,metrics
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

import math
import random



# corrigir o erro = TERM environment variable not set.
# os.environ["TERM"] = 'xterm'

# gera numeros aleatorios obedecendo a regra:  a <= rand < b
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
    return log(x) * (1 - log(x))

# Logistic function with output in [-1, 1].
def funcao_ativacao_log2(x):
    return 1 - 2 * log(x)

# derivada da funcao
def derivada_funcao_ativacao_log2(x):
    return -2 * log(x) * (1 - log(x))

class RedeNeural:
    def __init__(self, nos_entrada, nos_ocultos, nos_saida):
        # camada de entrada
        self.nos_entrada = nos_entrada + 1 # +1 por causa do no do bias
        # camada oculta
        self.nos_ocultos = nos_ocultos
        # camada de saida
        self.nos_saida = nos_saida
        # quantidade maxima de max_interacoes
        self.max_interacoes = 100
        # taxa de aprendizado
        self.taxa_aprendizado = 0.5
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
                self.wi[i][j] = rand(-0.2, 0.2)

        # vetor de pesos da camada intermediaria - saida
        for j in range(self.nos_ocultos):
            for k in range(self.nos_saida):
                self.wo[j][k] = rand(-2.0, 2.0)

        # last change in weights for momentum
        self.ci = np.zeros((self.nos_entrada, self.nos_ocultos))
        self.co = np.zeros((self.nos_ocultos, self.nos_saida))

    def fase_forward(self, entradas):
        # input activations: -1 por causa do bias
        #if(self.teste):
        #    print "fase_forward_valor: " +str(self.nos_entrada - 1);
        #    print "fase_forward_quantidade: " +str(self.nos_entrada);
        #    print "fase_forward_entrada: " +str(len(entradas));

        for i in range(self.nos_entrada - 1):
        #    if(self.teste):
        #        print "fase_forward_entradas:" +str(i)+"-"+str(entradas[i]);            
            self.ativacao_entrada[i] = entradas[i]

        # calcula as ativacoes dos neuronios da camada escondida
        for j in range(self.nos_ocultos):
            soma = 0
            for i in range(self.nos_entrada):
                soma = soma + self.ativacao_entrada[i] * self.wi[i][j]
            self.ativacao_ocultos[j] = funcao_ativacao_tang_hip(soma)

        # calcula as ativacoes dos neuronios da camada de saida
        # Note que as saidas dos neuronios da camada oculta fazem o papel de entrada
        # para os neuronios da camada de saida.
        for j in range(self.nos_saida):
            soma = 0
            for i in range(self.nos_ocultos):
                soma = soma + self.ativacao_ocultos[i] * self.wo[i][j]
            self.ativacao_saida[j] = funcao_ativacao_tang_hip(soma)

        return self.ativacao_saida


    def fase_backward(self, saidas_desejadas):
        # calcular os gradientes locais dos neuronios da camada de saida
        output_deltas = np.zeros(self.nos_saida)
        erro = 0
        for i in range(self.nos_saida):
            erro = saidas_desejadas[i] - self.ativacao_saida[i]
            output_deltas[i] = derivada_funcao_ativacao(self.ativacao_saida[i]) * erro

        # calcular os gradientes locais dos neuronios da camada escondida
        hidden_deltas = np.zeros(self.nos_ocultos)
        for i in range(self.nos_ocultos):
            erro = 0
            for j in range(self.nos_saida):
                erro = erro + output_deltas[j] * self.wo[i][j]
            hidden_deltas[i] = derivada_funcao_ativacao(self.ativacao_ocultos[i]) * erro

        # a partir da ultima camada ate a camada de entrada
        # os nos da camada atual ajustam seus pesos de forma a reduzir seus erros
        for i in range(self.nos_ocultos):
            for j in range(self.nos_saida):
                change = output_deltas[j] * self.ativacao_ocultos[i]
                self.wo[i][j] = self.wo[i][j] + (self.taxa_aprendizado * change) + (self.momentum * self.co[i][j])
                self.co[i][j] = change

        # atualizar os pesos da primeira camada
        for i in range(self.nos_entrada):
            for j in range(self.nos_ocultos):
                change = hidden_deltas[j] * self.ativacao_entrada[i]
                self.wi[i][j] = self.wi[i][j] + (self.taxa_aprendizado * change) + (self.momentum * self.ci[i][j])
                self.ci[i][j] = change

        # calcula erro
        erro = 0
        for i in range(len(saidas_desejadas)):
            erro = erro + 0.5 * (saidas_desejadas[i] - self.ativacao_saida[i]) ** 2
        return erro

    def test(self, entradas_saidas):
        self.teste = 1
        # print entradas_saidas
        for p in entradas_saidas:
            print p[0]
            array = self.fase_forward(p[0])
            #print("Entradas: " + str(p[0]) + ' - Saida encontrada/fase forward: ' + str(array[0]))
            print('Saida encontrada/fase forward: ' + str(array))

    def treinar(self, entradas_saidas):
        for i in range(self.max_interacoes):
            erro = 0
            for p in entradas_saidas:
                entradas = p[0]
                saidas_desejadas = p[1]
                self.fase_forward(entradas)
                erro = erro + self.fase_backward(saidas_desejadas)
            if i % 100 == 0:
                print "Erro = %2.3f"%erro



class Directions:
    NORTH = 'North'
    SOUTH = 'South'
    EAST = 'East'
    WEST = 'West'
    STOP = 'Stop'

class Region():
    def __init__(self):
        self.max = [-1,-1,-1]
        self.min = [256,256,256]
        
    def set(self,m):
        # min
        for i in range(3):
            if m[i] < self.min[i]:
                self.min[i] = m[i]
        # max
        for i in range(3):
            if m[i] > self.max[i]:
                self.max[i] = m[i]

class GetColor():
    def __init__(self):
        self.frame = None
        self.x = 0
        self.y = 0
        self.dist = 0
        self.px = 0
        self.py = 0
        self.ltmin = None
        self.ltmax = None
        self.size = 200
        self.start = (380,140)
        self.region = Region()
        self.thsv = None
        self.myframe = None
        self.main_dir = 'dataset'
        cv2.namedWindow('frame')
        cv2.namedWindow('patch')
        cv2.setMouseCallback('frame',self.mouse_callback)
        self.pressed = False
 
    def reset(self):
        self.x = 0
        self.y = 0
        self.ltmin = None
        self.ltmax = None
        self.region = Region()
        self.show()


    def show(self):
        text_msg = '%d %d %d'%(self.x,self.y,self.dist)
        cv2.putText(self.frame,text_msg,(10,self.frame.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
        cv2.imshow('frame',self.frame)
        if (self.thsv is not None):  cv2.imshow('patch',self.thsv)
        else:  cv2.imshow('patch',self.cropped)

        if (self.myframe is not None): cv2.imshow('result',self.myframe)

    def process_myframe(self):
        if self.thsv is not None:
            im2, contours, hierarchy = cv2.findContours(self.thsv.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            self.myframe = self.cropped.copy()
            tcontours = []
            for cnt in contours:
                perimeter = cv2.arcLength(cnt,True)
                if perimeter > 200:
                    tcontours.append(cnt)

            cv2.drawContours(self.myframe, tcontours, -1, (0,255,0), 3)

    def save_patch(self,patch_class):
        if self.thsv is not None:
            if not os.path.exists(self.main_dir):
                os.makedirs(self.main_dir)
            counter = 0
            for file_name in os.listdir(self.main_dir):
                if file_name.find('.png') > 0:
                    counter += 1
            lfile_name = self.main_dir+os.sep+"file%05d_%s.png"%(counter,patch_class)
            cv2.imwrite(lfile_name,self.thsv)
            print "%s saved!"%(lfile_name)

    def update_threshold(self):    
        if self.ltmin is not None:
            thsv = cv2.inRange(self.hsv, self.ltmin, self.ltmax)
            self.thsv = cv2.GaussianBlur(thsv,(5,5), 0)

    def set_frame(self,frame):

        crop_start = self.start
        crop_end   = (self.start[0] + self.size,self.start[1]+self.size  )
        cv2.rectangle(frame,crop_start,crop_end,(0,255,0),0)
        self.cropped = frame[crop_start[1]:crop_end[1],crop_start[0]:crop_end[0]]
        self.frame = frame
        self.hsv = cv2.cvtColor(self.cropped, cv2.COLOR_BGR2HSV)


    def in_cropped_region(self,x,y):
        if x > self.start[0] and x < self.start[0] + self.size:
            if y > self.start[1] and y < self.start[0] + self.size:
                return True
        return False

    def save(self):
        f = open('config.json','w')
        tcolor = (self.region.min,self.region.max)
        data = {'color': tcolor, 'start': self.start}
        json.dump(data,f)
        f.close()
    def load(self):
        f = open('config.json')    
        data = json.load(f)
        f.close()
        self.region.set(data['color'][0])
        self.region.set(data['color'][1])
        self.start = (data['start'][0],data['start'][1])
        self.ltmin = np.array(self.region.min)
        self.ltmax = np.array(self.region.max)

        

    def mouse_callback(self, event, x, y, flags, param):
        self.x = x
        self.y = y
        self.dist = dist = abs(self.x-self.px) +abs(self.y - self.py)
        if event == cv2.EVENT_MOUSEMOVE:
            if self.pressed and dist > 20:
                self.start = (x,y)
        if event == cv2.EVENT_LBUTTONDOWN:
            self.px = x
            self.py = y
            self.pressed = True
        elif event == cv2.EVENT_LBUTTONUP:
            self.pressed = False
            if dist < 1 and self.in_cropped_region(x,y):
                box = 1
                lx = x - self.start[0]
                ly = y - self.start[1]
                x1 = lx - box 
                x2 = lx + box
                y1 = ly - box
                y2 = ly + box
                cframe = self.hsv[y1:y2,x1:x2]
                for line in cframe:
                    for cols in line:
                        if cols[0] !=0:
                            self.region.set(cols.tolist())
                tmin = np.array(self.region.min)
                tmax = np.array(self.region.max)
                self.ltmin = tmin
                self.ltmax = tmax
class AnyJoystick:
    def __init__(self,main_dir = 'dataset'):
        self.main_dir = main_dir
        self.X = []
        self.y = []
        self.hclass = dict()
        self.vclass = []
        self.clf = None
        self.cont =0

    def img2instance(self,frame):
        dsize = (20,20)
        smaller = cv2.resize(frame,dsize)
        if len(smaller.shape) == 3:
            instance = smaller[:,:,0].reshape((1,-1)).tolist()[0]
        else:
            instance = smaller.reshape((1,-1)).tolist()[0]
        return instance

    def num_class(self,tclass):
        if tclass not in self.hclass.keys():
            next_i = len(self.vclass)
            self.vclass.append(tclass)
            self.hclass[tclass] = next_i
        return self.hclass[tclass]
            
    def load(self):
        cont = 0;
        for file_name in sorted(os.listdir(self.main_dir)):
            #print cont
            cont = cont+1
            fname = self.main_dir+os.sep+file_name
            frame = cv2.imread(fname)
            tclass = file_name.split('.')[0].split('_')[1]
            self.X.append(self.img2instance(frame))
            self.y.append(self.num_class(tclass))

    def train(self):

        #self.clf = MLPClassifier(solver='lbfgs', alpha=10, hidden_layer_sizes=(15,), random_state=1)
        #self.clf = svm.SVC(gamma=0.1)
        #self.evaluate()
        #print len(self.y)
        #self.cont = 0
        
        #[
        #    [[0,0], [0]],
        #    [[0,1], [1]],
        #    [[1,0], [1]],
        #    [[1,1], [0]]
        #]
        
        lista_modelo = 562*[2*[0]]
        print "cont x: " + str(len(self.X))

        #print "*******************************"
        for listax in self.X:
            #print "----------------------------------------"+str(self.cont)
            #print listax
            lista_y = [self.y[self.cont]]
            lista_modelo[self.cont][0] = listax
            lista_modelo[self.cont][1] = lista_y
            
            #if(lista_y == [3]):
            #    print listax

            #print lista_modelo[0]
            #print "classe: " + str(lista_modelo[self.cont][1])
            #print "----------------------------------------"
            self.cont = self.cont+1
        #print "*******************************"
        #print lista_modelo
        

        n = RedeNeural(400, 16, 1)
        
        # treinar com os padroes
        n.treinar(lista_modelo)

        print "***************************************************INICIANDO TESTE******************************************"
        print self.hclass
        entradas_saidas = 1*[2*[0]]
        entradas_saidas[0][0] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 210, 0, 0, 0, 0, 172, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 151, 2, 0, 20, 56, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16, 252, 69, 0, 255, 255, 255, 113, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 46, 182, 0, 210, 255, 253, 169, 207, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 23, 5, 241, 255, 255, 215, 4, 255, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 241, 106, 255, 255, 197, 254, 250, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 79, 255, 255, 253, 255, 255, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 254, 249, 255, 255, 230, 255, 231, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 255, 255, 255, 240, 255, 189, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 251, 255, 255, 255, 255, 254, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 110, 255, 255, 255, 252, 132, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 208, 55, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 189, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 237, 255, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 99, 235, 255, 228, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 12, 210, 255, 255, 252, 63, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 246, 255, 255, 255, 255, 201, 0, 0, 0];
        
        #print entradas_saidas

        n.test(entradas_saidas)

        #self.clf.fit(self.X,self.y)

    def evaluate(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.33, random_state=42)
        self.clf.fit(X_train,y_train)
        pred = self.clf.predict(X_test)
        
        print("Confusion matrix:\n%s"%(metrics.confusion_matrix(y_test,pred)))
        print self.vclass
    def predict(self,frame):
        if self.clf is not None:
            inst = np.matrix(self.img2instance(frame)).reshape((1,-1))
            print self.vclass[self.clf.predict(inst)]
            return self.vclass[self.clf.predict(inst)]


        
class AnyCamera:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.obj = GetColor()
        self.obj.load()
        self.joystick = AnyJoystick()
        self.joystick.load()
        self.joystick.train()

    def getMove(self):
        ret, frame = self.cap.read()
        self.obj.set_frame(frame)
        self.obj.update_threshold()
        return self.joystick.predict(self.obj.thsv)
        

