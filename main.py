import matplotlib.pyplot as plt             #to view images on a grid plot.
from tensorflow.keras import datasets,layers,models     #to create CNN

(training_images,training_labels),(testing_images,testing_labels)=datasets.cifar10.load_data() #dataset comes from the tensorflow
training_images,testing_images = training_images/255,testing_images/255     # we divide because every pixel should get value between 0 and 255

class_name = ["Plane","Car","Bird","Cat","Deer","Dog","Frog","Horse","Sheep","Truck"]  # The first ten classes


#This loops for arrange the first 16 images in a plot
for i in range(16):
    plt.subplot(4,4,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(training_images[i],cmap=plt.cm.binary)
    plt.xlabel(class_name[training_labels[i][0]])

plt.show()

#subsampling the example space in order to speed up the proccess
training_images = training_images[:20000]
training_labels = training_labels[:20000]
testing_images = testing_images[:4000]
testing_labels = testing_labels[:4000]

model = models.Sequential() # our model will be sequential
#input layer
model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(32,32,3)))
#Hiddeb Layers
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Conv2D(64,(3,3),activation='relu'))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Conv2D(64,(3,3),activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64,activation='relu'))

# output layer
model.add(layers.Dense(10,activation='softmax'))

# compiling the model
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

#fitting the model
model.fit(training_images,training_labels,epochs=10,validation_data=(testing_images,testing_labels))

#Evaluating the model
loss,accuracy = model.evaluate(testing_images,testing_labels)
print(f"Loss: {loss}")
print(f"Accuracy: {accuracy}")

#save model to test
model.save('image_classifier.model')
