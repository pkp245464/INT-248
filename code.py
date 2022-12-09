##########################################################################
######################: Human Face Detection: ############################
##########################################################################

#####################: Detect face from Image :########################### 
import cv2
img = cv2.imread('SwamiVivekananda.jpg')

gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

haar_cascade = cv2.CascadeClassifier('Haarcascade_frontalface_default.xml')

faces_rect = haar_cascade.detectMultiScale(gray_img, 1.1, 9)

for (x, y, w, h) in faces_rect:
	cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow('Detected faces', img)

cv2.waitKey(0)
###############################---:End:---################################



#####################: Detect face from Camera :############################

import cv2
face_cascade = cv2.CascadeClassifier('Haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
while True:
    _, img = cap.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.1,4)
    
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x + w,y + h),(255,0,0),2)
    cv2.imshow('img',img)
    
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
cap.release()        
###############################---:End:---################################



##########################################################################
########################: Color Dectection: ##############################
##########################################################################

from collections import Counter
from sklearn.cluster import KMeans
from matplotlib import colors
import matplotlib.pyplot as plt
import numpy as np
import cv2 

def rgb_to_hex(rgb_color):
    hex_color = "#"
    for i in rgb_color:
        i = int(i)
        hex_color += ("{:02x}".format(i))
    return hex_color


rgb_to_hex((255,0,0))


img_name = 'SwamiVivekananda.jpg'
raw_img = cv2.imread(img_name)
raw_img = cv2.cvtColor(raw_img,cv2.COLOR_BGR2RGB)


img = cv2.resize(raw_img,(900,600),interpolation = cv2.INTER_AREA)
img.shape


img = img.reshape(img.shape[0]*img.shape[1],3)
img.shape


img


clf = KMeans(n_clusters=5)
color_labels = clf.fit_predict(img)
center_colors = clf.cluster_centers_


color_labels


center_colors


counts = Counter(color_labels)
counts


ordered_colors = [center_colors[i] for i in counts.keys()]
hex_colors = [rgb_to_hex(ordered_colors[i]) for i in counts.keys()]
hex_colors


plt.figure(figsize = (12,8))
plt.pie(counts.values(),labels = hex_colors,colors = hex_colors)
plt.savefig(f'{img_name[:-4]}-analysis.png')


def hex_to_rgb(hex):
  rgb = []
  for i in (0, 2, 4):
    decimal = int(hex[i:i+2], 16)
    rgb.append(decimal)
  
  return tuple(rgb)

print(hex_to_rgb('a96139'))
###############################---:End:---################################



