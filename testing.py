import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets,layers,models

class_name = ["Plane","Car","Bird","Cat","Deer","Dog","Frog","Horse","Sheep","Truck"]

model = models.load_model('image_classifier.model')

img = cv.imread("plane.jpg")
img = cv.cvtColor(img,cv.COLOR_BGR2RGB)
plt.imshow(img,cmap=plt.cm.binary)

prediction = model.predict(np.array([img])/255)
index = np.argmax(prediction)                   #argmax gives index of the maximum value

print(f"Prediction : {class_name[index]}")
plt.show()
