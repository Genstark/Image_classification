from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import vgg16

class_object = ["bottle","car", "cat", "dog"]

train_path = "dataset/train/train"
valid_path = "dataset/train/valid"
test_path = "dataset/train/test"

imgdata = ImageDataGenerator(
                featurewise_center=False,
                samplewise_center=False,
                featurewise_std_normalization=False,
                samplewise_std_normalization=False,
                zca_whitening=False,
                zca_epsilon=1e-6,
                rotation_range=20,
                width_shift_range=0.1,
                height_shift_range=0.1,
                brightness_range=None,
                shear_range=0.1,
                zoom_range=0.2,
                channel_shift_range=0.,
                fill_mode="nearest",
                cval=0.,
                horizontal_flip=True,
                vertical_flip=False,
                rescale=None,
                preprocessing_function=None,
                data_format=None,
                validation_split=0.0,
                dtype=None
)

imgerescale = ImageDataGenerator(rescale=1./255)

train_batch = imgdata.flow_from_directory(
                            directory=train_path,
                            target_size=(224,224),
                            color_mode="rgb",
                            classes=class_object,
                            class_mode="categorical",
                            batch_size=16,
                            shuffle=True,
                            seed=None,
                            save_to_dir=None,
                            save_prefix=None,
                            save_format=None,
                            follow_links=None,
                            subset=None,
                            interpolation="nearest",
)

valid_batch = imgerescale.flow_from_directory(
                            directory=valid_path,
                            target_size=(224,224),
                            color_mode="rgb",
                            classes=class_object,
                            class_mode="categorical",
                            batch_size=16,
                            shuffle=True,
                            seed=None,
                            save_to_dir=None,
                            save_prefix=None,
                            save_format=None,
                            follow_links=False,
                            subset=None,
                            interpolation="nearest"
)

test_batch = imgerescale.flow_from_directory(
                            directory=test_path,
                            target_size=(224,224),
                            color_mode="rgb",
                            classes=class_object,
                            class_mode="categorical",
                            batch_size=16,
                            shuffle=False,
                            seed=100,
                            save_to_dir=None,
                            save_prefix=None,
                            save_format=None,
                            follow_links=False,
                            subset=None,
                            interpolation="nearest"
)


from tensorflow.keras.layers import Activation, Dense, Dropout 
from tensorflow.keras.layers import BatchNormalization, MaxPool2D, Conv2D, Flatten
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential


model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(2,2), padding="same", input_shape=(224,224,3)))
model.add(Activation("relu")) 

model.add(Conv2D(filters=64, kernel_size=(2,2), padding="same"))
model.add(Activation("relu"))
model.add(MaxPool2D(pool_size=(2,2), strides=2))

model.add(Conv2D(filters=128, kernel_size=(2,2), padding="same"))
model.add(Activation("relu"))
model.add(MaxPool2D(pool_size=(2,2), strides=2))

model.add(Conv2D(filters=128, kernel_size=(2,2), padding="same"))
model.add(Activation("relu"))
model.add(MaxPool2D(pool_size=(2,2), strides=2))

model.add(Conv2D(filters=256, kernel_size=(2,2), padding="same"))
model.add(Activation("relu"))
model.add(MaxPool2D(pool_size=(2,2), strides=2))

model.add(Conv2D(filters=256, kernel_size=(2,2), padding="same"))
model.add(Activation("relu"))
model.add(MaxPool2D(pool_size=(2,2), strides=2))

model.add(Conv2D(filters=512, kernel_size=(2,2), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=1))
model.add(Dropout(0.2))

model.add(Conv2D(filters=512, kernel_size=(2,2), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=1))
model.add(Dropout(0.1))

model.add(Conv2D(filters=1024, kernel_size=(2,2), padding="same"))
model.add(Activation("relu"))
model.add(BatchNormalization(axis=1))

model.add(Flatten())

model.add(Dense(500))
model.add(Activation("relu"))

model.add(Dense(100))
model.add(Activation("relu"))

model.add(Dense(units=len(class_object)))
model.add(Activation("softmax"))

model.compile(
    optimizer=Adam(learning_rate=0.0001), 
    loss="categorical_crossentropy", 
    metrics=["accuracy"]
)

model.fit(
    x=train_batch, 
    validation_data=valid_batch,
    epochs=4,
    verbose=1,
    shuffle=False
)

# model.save("premodel/model_4.h5")