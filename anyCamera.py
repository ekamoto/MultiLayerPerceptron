# Alunos: Leandro Shindi Ekamoto, Diego Takaki
# Leia readme
#

import cv2
import numpy as np
import json
import os
from sklearn import svm,metrics
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from backpropagation import RedeNeural

########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################
########################################################################################################################

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
        self.n = RedeNeural(400, 12, 1)

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

        self.clf = MLPClassifier(solver='lbfgs', alpha=10, hidden_layer_sizes=(15,), random_state=1)
        self.clf = svm.SVC(gamma=0.1)
        self.evaluate()

        self.clf.fit(self.X,self.y)

    def evaluate(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.30, random_state=42)
        #self.clf.fit(X_train,y_train)
        #pred = self.clf.predict(X_test)

        matrix = []

        # hisamoto
        for listax in X_train:
            lista_y = [y_train[self.cont]]

            if(lista_y == [4]):
                #print listax
                contador = 0
                for lista in listax:
                    listax[contador] = listax[contador]/100.0
                    contador = contador + 1

                contador = 0
                for lista in lista_y:
                    lista_y[contador] = lista_y[contador]/10.0
                    lista_y[contador] = 0.44
                    contador = contador + 1

                matrix.append([listax, lista_y])

            self.cont = self.cont+1

        self.cont = 0
        for listax in X_train:
            lista_y = [y_train[self.cont]]

            if(lista_y == [3]):
                #print listax
                contador = 0
                for lista in listax:
                    listax[contador] = listax[contador]/100.0
                    contador = contador + 1

                contador = 0
                for lista in lista_y:
                    lista_y[contador] = lista_y[contador]/10.0
                    lista_y[contador] = 0.33
                    contador = contador + 1

                matrix.append([listax, lista_y])

            self.cont = self.cont+1

        self.cont = 0
        for listax in X_train:
            lista_y = [y_train[self.cont]]

            if(lista_y == [2]):
                #print listax
                contador = 0
                for lista in listax:
                    listax[contador] = listax[contador]/100.0
                    contador = contador + 1

                contador = 0
                for lista in lista_y:
                    lista_y[contador] = lista_y[contador]/10.0
                    lista_y[contador] = 0.22
                    contador = contador + 1

                matrix.append([listax, lista_y])

            self.cont = self.cont+1

        self.cont = 0
        for listax in X_train:
            lista_y = [y_train[self.cont]]

            if(lista_y == [1]):
                #print listax
                contador = 0
                for lista in listax:
                    listax[contador] = listax[contador]/100.0
                    contador = contador + 1

                contador = 0
                for lista in lista_y:
                    lista_y[contador] = lista_y[contador]/10.0
                    lista_y[contador] = 0.11
                    contador = contador + 1

                matrix.append([listax, lista_y])
            self.cont = self.cont+1

        self.cont = 0
        for listax in X_train:
            lista_y = [y_train[self.cont]]

            if(lista_y == [0]):
                #print listax
                contador = 0
                for lista in listax:
                    listax[contador] = listax[contador]/100.0
                    contador = contador + 1

                contador = 0
                for lista in lista_y:
                    lista_y[contador] = lista_y[contador]/10.0
                    lista_y[contador] = 0.5
                    contador = contador + 1

                matrix.append([listax, lista_y])
            self.cont = self.cont+1

        self.n = RedeNeural(400, 12, 1)

        # treinar com os padroes
        self.n.fit(matrix)




        print "-----------------------------------------------hisamototeste--------------------------------------------------------"
        indice = 0

        pred = []
        valor_estimado = []
        for p in X_test:

            lista_teste = []
            for lista in p:
                lista_teste.append(lista/100.0)

            array = self.n.fase_forward(lista_teste)

            print(str(y_test[indice]/10.0) + '-->' + str(array[0]) + "--->" + str(self.analisaClasse(array[0])))
            pred.append(self.analisaClasse(array[0]))
            valor_estimado.append(self.analisaClasse(y_test[indice]/10.0))
            indice = indice + 1

        print pred
        print valor_estimado

        print("Confusion matrix:\n%s"%(metrics.confusion_matrix(valor_estimado, pred)))

    def predict(self,frame):

        if self.clf is not None:

            inst = np.matrix(self.img2instance(frame)).reshape((1,-1))

            lista_teste = []
            for lista in self.img2instance(frame):
                lista_teste.append(lista/100.0)

            array = self.n.fase_forward(lista_teste)
            print "Resposta da rede:"+str(self.analisaClasse(array[0]))
            return "Resposta da rede:"+str(self.analisaClasse(array[0]))

    def analisaClasse(self, valor):
        #odem das direcoes
        #parado, direita, esquerda, cima, baixo

        if(valor >= 0.47):
            return 0
        if(valor >= 0.4 and valor < 0.47):
            return 4
        if(valor >= 0.27 and valor < 0.4):
            return 3
        if(valor >= 0.17 and valor < 0.3):
            return 2
        if(valor < 0.2):
            return 1
        return 0

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
