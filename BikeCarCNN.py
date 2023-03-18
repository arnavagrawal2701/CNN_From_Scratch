#Importing Libraries
import tensorflow as tf
from tensorflow.keras import layers,models
import matplotlib.pyplot as plt
from collections import Counter
from keras.preprocessing.image import ImageDataGenerator

#importing Dataset
file_path='Car-Bike-Dataset'

batch_size=32
img_height=256
img_width=256

#Generating Data
data_generator=ImageDataGenerator(rescale=1/255.,
                                  rotation_range=90,
                                  width_shift_range=0.1,
                                  horizontal_flip=True,
                                  vertical_flip=True,
                                  validation_split=0.2)

train_set=data_generator.flow_from_directory(file_path,
                                             class_mode='binary',
                                             target_size=(img_height,img_width),
                                             shuffle=True,
                                             batch_size=batch_size,
                                             subset='training')

test_set=data_generator.flow_from_directory(file_path,
                                            class_mode='binary',
                                            target_size=(img_height,img_width),
                                            shuffle=False,
                                            batch_size=batch_size,
                                            subset='validation')

train_counter = Counter(train_set.classes)
test_counter = Counter(test_set.classes)
print(train_counter.items())
print(test_counter.items())

#Visualising Data
tr_keys = list(train_counter.keys())
te_keys = list(test_counter.keys())
vals = [train_counter[k] for k in tr_keys]
vals_te = [test_counter[j] for j in te_keys]

class_names_t = ["Bike", "Car"]
n_images = 10
plt.figure(figsize=(10, 10))
# DictionaryIterator
images, labels = train_set.next()
for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i])
    plt.title(class_names_t[labels[i].astype("uint8")])
    plt.axis("off")
    i+=1
    if i>=(n_images+1):
        break
plt.tight_layout()
plt.show()

#Building, Compiling and Fitting CNN model
model = models.Sequential([
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),
    layers.Conv2D(128, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])
model.build(input_shape = (None,256,256,3))

model.summary()

model.compile(optimizer='adam',loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),metrics=['accuracy'])
model.fit(train_set, validation_data=test_set, epochs=15, shuffle = True, batch_size = 32)

#Evaluating model
eva1 = model.evaluate(test_set)

print(f'Test Accuracy of model: {eva1[1]}')

