import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array

DATA_PATH = '../PlantVillage'

for cat in os.listdir(DATA_PATH):
    path = os.path.join(DATA_PATH, cat)
    for img in os.listdir(path):
        image = cv2.imread(os.path.join(path, img), cv2.IMREAD_UNCHANGED)
        plt.imshow(image)
        plt.title(f'{cat}')
        plt.show()
        break
    
IMG_SHAPE = (224, 224)
INPUT_SHAPE = [224, 224, 3]
EPOCHS = 50
BS = 32
img_data_gen = ImageDataGenerator(rescale=1./255, rotation_range=0.2, horizontal_flip=True, vertical_flip=True,
                                 shear_range=0.2, validation_split=0.25)

train_data_gen = img_data_gen.flow_from_directory(DATA_PATH, batch_size=BS, subset='training', 
                                                  class_mode='categorical', shuffle=True) 

val_data_gen = img_data_gen.flow_from_directory(DATA_PATH, batch_size=BS, subset='validation', 
                                                  class_mode='categorical', shuffle=True)

label = train_data_gen.class_indices
label

img = train_data_gen.__getitem__(11)[0]
plt.imshow(img[0])
#plt.title(label[11])

plt.figure(figsize=(16,10))
for i in range(15):
    plt.subplot(5, 3, i+1)
    img = train_data_gen.__getitem__(i)[0]
    plt.imshow(img[0])
    plt.xticks()
    plt.show()
    
def model_building(model_name, INPUT_SHAPE=INPUT_SHAPE):
    print('Model Initialization started')
    base_model = model_name(include_top=False, weights='imagenet', input_shape=INPUT_SHAPE)
    
    for layers in base_model.layers:
        layers.trainable = False
    print('Model Initialization finished')
    
    #model creation
    print('Model creation started')
    inp_model = base_model.output
    
    x = GlobalAveragePooling2D()(inp_model)
    x = Dense(128, activation = 'relu')(x)
    x = Dense(15, activation = 'sigmoid')(x)
    
    model = Model(inputs = base_model.input, outputs = x)
    
    #model summary
    print('Model summary')
    #model.summary()
    
    #model compilation
    model.compile(optimizer = 'adam', metrics=['accuracy'], loss = 'categorical_crossentropy')
    
    history = model.fit(train_data_gen, validation_data=val_data_gen, 
                       validation_steps=len(val_data_gen)//BS,
                       steps_per_epoch=len(train_data_gen)//BS,
                       batch_size=BS, 
                       epochs=EPOCHS)
    
    print('Model Building Finished')
    
    model.save(f'../output_models/{model_name}_1.h5')
    print('Model was saved')
    
    return history

def evaluation_plot(model):
    sns.set_style('whitegrid')
    
    plt.figure(figsize=(10, 8))
    plt.plot(model['loss'], label = 'loss')
    plt.plot(model['accuracy'], label = 'accuracy')
    plt.plot(model['val_loss'], label = 'val_loss')
    plt.plot(model['val_accuracy'], label = 'val_accuracy')
    plt.legend()
    plt.title('Model Evaluation')
    plt.show()
    
#VGG16 model
from tensorflow.keras.applications.vgg16 import VGG16

vgg16_hist = model_building(VGG16)

evaluation_plot(vgg16_hist.history)

#InceptionV3 model

from tensorflow.keras.applications.inception_v3 import InceptionV3

inc_history = model_building(InceptionV3)

evaluation_plot(inc_history.history)

#Custom CNN model

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set the path to your dataset
DATA_PATH = '../PlantVillage'

# Define image dimensions and other parameters
IMG_SHAPE = (224, 224)
INPUT_SHAPE = (224, 224, 3)
EPOCHS = 50
BS = 32

# Create data generators
img_data_gen = ImageDataGenerator(rescale=1./255, rotation_range=0.2, horizontal_flip=True,
                                  vertical_flip=True, shear_range=0.2, validation_split=0.25)

train_data_gen = img_data_gen.flow_from_directory(DATA_PATH, batch_size=BS, subset='training',
                                                  class_mode='categorical', shuffle=True,
                                                  target_size=IMG_SHAPE)

val_data_gen = img_data_gen.flow_from_directory(DATA_PATH, batch_size=BS, subset='validation',
                                                class_mode='categorical', shuffle=True,
                                                target_size=IMG_SHAPE)

# Define and compile the custom CNN model
def custom_cnn_model(input_shape=INPUT_SHAPE):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(15, activation='sigmoid'))

    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Train the custom CNN model
custom_model = custom_cnn_model()
custom_history = custom_model.fit(train_data_gen, validation_data=val_data_gen,
                                  validation_steps=len(val_data_gen)//BS,
                                  steps_per_epoch=len(train_data_gen)//BS,
                                  batch_size=BS, epochs=EPOCHS)

# Save the trained model
model_save_path = '../output_models/custom_cnn_model.h5'
custom_model.save(model_save_path)
print(f"Custom CNN model saved at: {model_save_path}")

# Plot the evaluation metrics
sns.set_style('whitegrid')
plt.figure(figsize=(10, 8))
plt.plot(custom_history.history['loss'], label='loss')
plt.plot(custom_history.history['accuracy'], label='accuracy')
plt.plot(custom_history.history['val_loss'], label='val_loss')
plt.plot(custom_history.history['val_accuracy'], label='val_accuracy')
plt.legend()
plt.title('Custom CNN Model Evaluation')
plt.show()

#evaluation

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import seaborn as sns

# Load VGG16 model
vgg16_model = load_model('../output_models/<function VGG16 at 0x7e0346c37f40>_1.h5')  # Update with the correct path

# Load InceptionV3 model
inception_model = load_model('../output_models/<function InceptionV3 at 0x7e0346c34c10>_1.h5')  # Update with the correct path

# Load Custom CNN model
custom_model = load_model('../output_models/custom_cnn_model.h5')  # Update with the correct path

def evaluate_model(model, data_gen):
    # Generate predictions
    predictions = model.predict(data_gen)
    y_pred = np.argmax(predictions, axis=1)
    y_true = data_gen.classes

    # Accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f'Accuracy: {accuracy:.4f}')

    # F1 Score, Precision, Recall
    f1 = f1_score(y_true, y_pred, average='weighted')
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')

    print(f'F1 Score: {f1:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=data_gen.class_indices.keys(),
                yticklabels=data_gen.class_indices.keys())
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    
# Evaluate VGG16
vgg16_eval = vgg16_model.evaluate(val_data_gen)
print("Evaluation on VGG16 model:")
print(f"Loss: {vgg16_eval[0]:.4f}")
print(f"Accuracy: {vgg16_eval[1]:.4f}")

evaluate_model(vgg16_model, val_data_gen)

# Evaluate InceptionV3
inc_eval = inception_model.evaluate(val_data_gen)
print("\nEvaluation on InceptionV3 model:")
print(f"Loss: {inc_eval[0]:.4f}")
print(f"Accuracy: {inc_eval[1]:.4f}")

evaluate_model(inception_model, val_data_gen)

# Evaluate Custom CNN
# custom_eval = custom_model.evaluate(val_data_gen)
print("\nEvaluation on Custom CNN model:")
print(f"Loss: {custom_history.history['loss']:.4f}")
print(f"Accuracy: {custom_history.history['accuracy']:.4f}")

# evaluate_model(custom_model, val_data_gen)   