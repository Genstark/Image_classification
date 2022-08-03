from tensorflow.keras.models import load_model
import cv2
import numpy as np
import sys

def mapper(val):
    
    CLASSES_MAP = {
        0: "bottle",
        1: "car",
        2: "cat",
        3: "dog",
    }
    return CLASSES_MAP[val]


model = load_model("premodel/keras_model.h5")

model.compile(loss="categorical_crossentropy",
              optimizer='adam',
              metrics=['accuracy'])

filepath = "dataset/Cat/cats (12500).jpg"

# prepare the image
img = cv2.imread(filepath)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (224, 224))

# predict the move made
pred = model.predict(np.array([img]))
move_code = np.argmax(pred[0])
move_name = mapper(move_code)

print("Predicted:",move_name)
print(pred)