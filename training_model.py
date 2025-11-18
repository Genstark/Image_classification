from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Activation, Dense, Dropout, BatchNormalization
from tensorflow.keras.layers import MaxPool2D, Conv2D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

class_object = ["bottle","car", "cat", "dog"]

train_path = "dataset/train/train"
valid_path = "dataset/train/valid"
test_path = "dataset/train/test"

# -------------------------------------------------------------------------
# ★ 1. Stronger augmentation
# -------------------------------------------------------------------------
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    width_shift_range=0.15,
    height_shift_range=0.15,
    zoom_range=0.25,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest"
)

valid_gen = ImageDataGenerator(rescale=1./255)
test_gen  = ImageDataGenerator(rescale=1./255)

train_batch = train_gen.flow_from_directory(
    directory=train_path,
    target_size=(224,224),
    classes=class_object,
    batch_size=16,
    class_mode="categorical"
)

valid_batch = valid_gen.flow_from_directory(
    directory=valid_path,
    target_size=(224,224),
    classes=class_object,
    batch_size=16,
    class_mode="categorical"
)

test_batch = test_gen.flow_from_directory(
    directory=test_path,
    target_size=(224,224),
    classes=class_object,
    batch_size=16,
    class_mode="categorical",
    shuffle=False
)

# -------------------------------------------------------------------------
# ★ 2. Much improved CNN architecture
# -------------------------------------------------------------------------
model = Sequential()

# Block 1
model.add(Conv2D(32, (3,3), padding="same", activation="relu", input_shape=(224,224,3)))
model.add(BatchNormalization())
model.add(Conv2D(32, (3,3), padding="same", activation="relu"))
model.add(BatchNormalization())
model.add(MaxPool2D(2,2))
model.add(Dropout(0.25))

# Block 2
model.add(Conv2D(64, (3,3), padding="same", activation="relu"))
model.add(BatchNormalization())
model.add(Conv2D(64, (3,3), padding="same", activation="relu"))
model.add(BatchNormalization())
model.add(MaxPool2D(2,2))
model.add(Dropout(0.25))

# Block 3
model.add(Conv2D(128, (3,3), padding="same", activation="relu"))
model.add(BatchNormalization())
model.add(Conv2D(128, (3,3), padding="same", activation="relu"))
model.add(BatchNormalization())
model.add(MaxPool2D(2,2))
model.add(Dropout(0.3))

# Block 4
model.add(Conv2D(256, (3,3), padding="same", activation="relu"))
model.add(BatchNormalization())
model.add(Conv2D(256, (3,3), padding="same", activation="relu"))
model.add(BatchNormalization())
model.add(MaxPool2D(2,2))
model.add(Dropout(0.4))

# Dense Head
model.add(Flatten())
model.add(Dense(512, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(len(class_object), activation="softmax"))

# -------------------------------------------------------------------------
# ★ 3. Compile
# -------------------------------------------------------------------------
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# -------------------------------------------------------------------------
# ★ 4. Callbacks to improve accuracy
# -------------------------------------------------------------------------
callbacks = [
    EarlyStopping(patience=6, restore_best_weights=True),
    ReduceLROnPlateau(factor=0.3, patience=3, min_lr=1e-6)
]

# -------------------------------------------------------------------------
# ★ 5. Train
# -------------------------------------------------------------------------
model.fit(
    x=train_batch, 
    validation_data=valid_batch,
    epochs=30,
    callbacks=callbacks,
    verbose=1
)

# -------------------------------------------------------------------------
# ★ 6. Save model
# -------------------------------------------------------------------------
model.save("model.h5")
