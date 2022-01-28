import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets,layers,models

class_name = ["Plane","Car","Bird","Cat","Deer","Dog","Frog","Horse","Sheep","Truck"]

#loading the model
model = models.load_model('image_classifier.model')

#loading the test image
img = cv.imread("plane.jpg")
#images have BGR format but we worked(matplotlib) on RGB format so we convert it.
img = cv.cvtColor(img,cv.COLOR_BGR2RGB)

#in order to show image
plt.imshow(img,cmap=plt.cm.binary)

#lets predict the classes
prediction = model.predict(np.array([img])/255)

index = np.argmax(prediction)                   #argmax gives index of the maximum value

#and result is here
print(f"Prediction : {class_name[index]}")
plt.show()
