# Alunos: Leandro Shindi Ekamoto, Diego Takaki
# Leia readme
#

import numpy as np
import cv2
from anyCamera import GetColor,Directions,AnyJoystick

cap = cv2.VideoCapture(0)
obj = GetColor()
obj.load()

#cima = 82
#baixo = 84
#esquerda = 81
#direita = 83

keymap ={82:Directions.NORTH,84:Directions.SOUTH,83:Directions.EAST,81:Directions.WEST,32:Directions.STOP}

joystick = AnyJoystick()
cont = 0
while(True):

    cont = cont+1
    ret, frame = cap.read()

    obj.set_frame(frame)
    obj.update_threshold()
    obj.show()
    key = cv2.waitKey(1) & 0xFF

    if key ==   ord('q'):
        break
    elif key == ord('r'):
        obj.reset()
        print "Color reset!"
    elif key == ord('s'): # salva posicao do retangulo e cor da segmentacao
        obj.save()
        print "Saved"
    elif key in keymap.keys(): # salva seta para cima,baixo,direta,esquerda e espao
            patch_class = keymap[key]
            obj.save_patch(patch_class)
            print "%s saved"%(patch_class)
    elif key == ord('t'):
        print "Training algorithm"
        joystick.load()
        joystick.train()

    joystick.predict(obj.thsv)

cap.release()
cv2.destroyAllWindows()
