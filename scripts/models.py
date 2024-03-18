from keras.models import Sequential
from keras.layers import Input, Conv2D, Dense, Flatten, Dropout, MaxPooling2D, \
    BatchNormalization
from keras.models import Model
from keras.layers import Input, Conv2D, Dense, Flatten, Dropout, MaxPooling2D, \
    BatchNormalization
from keras.models import Model
from keras.models import Sequential

from tensorflow.keras.layers import Input, Conv2D
from tensorflow.keras.layers import MaxPool2D, Flatten, Dense
from tensorflow.keras import Model


def create_cnn(input_shape, output_shape, dropout=None, regularizer=None):
    input_data = Input(shape=input_shape)
    x = Conv2D(20, (5, 5), activation="relu")(input_data)
    x = MaxPooling2D()(x)
    # if dropout is not None:
    #     x = Dropout(dropout)(x)
    x = Conv2D(50, (5, 5), activation="relu")(x)
    x = MaxPooling2D()(x)
    # if dropout is not None:
    #     x = Dropout(dropout)(x)
    x = Flatten()(x)
    x = Dense(500, activation="relu", kernel_regularizer=regularizer)(x)
    if dropout is not None:
        x = Dropout(dropout)(x)
    output = Dense(output_shape)(x)
    model = Model(input_data, output)
    return model


def create_model(input_shape, output_shape):
    input_data = Input(shape=input_shape)
    x = Conv2D(20, (5, 5), activation="relu")(input_data)
    x = MaxPooling2D()(x)
    x = Conv2D(50, (5, 5), activation="relu")(x)
    x = MaxPooling2D()(x)
    x = Flatten()(x)
    x = Dense(500, activation="relu")(x)
    output = Dense(output_shape)(x)
    model = Model(input_data, output)
    return model


def create_model_softmax(input_shape=(32, 32, 3), output_shape=10):
    input_data = Input(shape=input_shape)
    x = Conv2D(20, (5, 5), activation="relu")(input_data)
    x = MaxPooling2D()(x)
    x = Conv2D(50, (5, 5), activation="relu")(x)
    x = MaxPooling2D()(x)
    x = Flatten()(x)
    x = Dense(500, activation="relu")(x)
    # x = Dropout(0.25)(x)
    output = Dense(output_shape)(x)
    model = Model(input_data, output)
    return model


def custom_model_ch_minst(dropout=None, regularizer=None):
    model = Sequential()
    model.add(Conv2D(filters=128, kernel_size=(5, 5), padding='same', activation='relu', input_shape=(64, 64, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(256, activation="relu"))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(8))
    return model


def regularized_model_ch_minst(dropout=None, regularizer=None):
    input_data = Input(shape=(64, 64, 1))
    x = Conv2D(filters=128, kernel_size=(5, 5), padding='same', activation='relu')(input_data)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    x = Dense(256, activation="relu", kernel_regularizer=regularizer)(x)
    if dropout is not None:
        x = Dropout(dropout)(x)
    x = Dense(64, activation="relu", kernel_regularizer=regularizer)(x)
    if dropout is not None:
        x = Dropout(dropout)(x)
    x = Dense(32, activation="relu", kernel_regularizer=regularizer)(x)
    if dropout is not None:
        x = Dropout(dropout)(x)
    output = Dense(8)(x)
    model = Model(input_data, output)

    # model = Sequential()
    # model.add(Conv2D(filters=128, kernel_size=(5, 5), padding='same', activation='relu', input_shape=(64, 64, 1)))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))

    # model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))

    # model.add(Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))

    # model.add(Flatten())

    # model.add(Dense(256, activation="relu"))
    # model.add(Dense(64, activation="relu"))
    # model.add(Dense(32, activation="relu"))

    # model.add(Dense(8, activation="softmax"))

    return model


def create_multi_task_model(input_shape, output_shape):
    input_data = Input(shape=input_shape, name='input')
    x = Conv2D(20, (5, 5), activation="relu")(input_data)
    x = MaxPooling2D()(x)
    x = Conv2D(50, (5, 5), activation="relu")(x)
    x = MaxPooling2D()(x)
    x = Flatten()(x)
    # task1 = Dense(1024, activation="relu")(x)
    task1 = Dense(500, activation="relu")(x)
    output_task1 = Dense(output_shape, activation='softmax', name='output_1')(task1)
    # task2 = Dense(1024, activation="relu")(x)
    task2 = Dense(500, activation="relu")(x)
    output_task2 = Dense(output_shape, activation='softmax', name='output_2')(task2)
    model = tf.keras.Model(inputs=input_data, outputs=[output_task1, output_task2])
    return model


import tensorflow as tf


def customized_restnet(input_shape=(32, 32, 3), output_shape=10):
    # Load pre-trained ResNet18 model (excluding top fully connected layers)
    base_model = tf.keras.applications.ResNet50(input_shape=input_shape, weights='imagenet', include_top=False)
    # tf.compat.v1.disable_eager_execution()
    resnet14 = Model(inputs=base_model.input, outputs=base_model.get_layer('conv2_block1_0_conv').output)

    # pretrained_resnet = tf.keras.applications.ResNet50(
    #     include_top=False, weights='imagenet', input_shape=input_shape
    # )

    # Freeze the layers in the pre-trained model
    # for layer in resnet14.layers:
    #     layer.trainable = False
    # Disable eager execution
    # tf.compat.v1.disable_eager_execution()
    # Customize the pre-trained model by adding new layers
    custom_model = tf.keras.Sequential()
    custom_model.add(resnet14)
    custom_model.add(tf.keras.layers.GlobalAveragePooling2D())
    custom_model.add(tf.keras.layers.Dense(128, activation='relu'))
    custom_model.add(
        tf.keras.layers.Dense(output_shape, activation='softmax'))  # Customize the number of output classes
    # tf.compat.v1.enable_eager_execution()
    return custom_model

    # Compile the model
    # custom_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# Optionally, you can fine-tune some layers if needed
# for layer in custom_model.layers[:100]:
#     layer.trainable = True

# Train your model using custom data
# model.fit(train_data, train_labels, validation_data=(val_data, val_labels), epochs=10)


def custom_vgg19():
    # Create the ResNet50 model and set the layers to be non-trainable
    resnet_model = Sequential()
    pretrained_model = tf.keras.applications.VGG19(include_top=False,
                                                   input_shape=(32, 32, 3),
                                                   pooling='avg',
                                                   weights='imagenet')

    # for layer in pretrained_model.layers:
    #     layer.trainable = False
    resnet_model.add(pretrained_model)

    # Add fully connected layers for classification
    resnet_model.add(Flatten())
    resnet_model.add(Dense(512, activation='relu'))
    resnet_model.add(Dense(10))
    return resnet_model


def create_model_with_batch_norm(input_shape, output_shape):
    input_data = Input(shape=input_shape)

    x = Conv2D(20, (5, 5), activation="relu")(input_data)
    x = BatchNormalization()(x)  # Add Batch Normalization here
    x = MaxPooling2D()(x)

    x = Conv2D(50, (5, 5), activation="relu")(x)
    x = BatchNormalization()(x)  # Add Batch Normalization here
    x = MaxPooling2D()(x)

    x = Flatten()(x)
    x = Dense(500, activation="relu")(x)
    x = BatchNormalization()(x)  # Add Batch Normalization here

    output = Dense(output_shape, activation='softmax')(x)

    model = Model(input_data, output)
    return model


class PurchaseClassifier(tf.keras.Model):
    def __init__(self, num_classes=100, droprate=0):
        super(PurchaseClassifier, self).__init__()

        self.features = tf.keras.Sequential([
            tf.keras.layers.Dense(1024, activation='tanh', input_shape=(600,)),
            tf.keras.layers.Dense(512, activation='tanh'),
            tf.keras.layers.Dense(256, activation='tanh'),
            tf.keras.layers.Dense(128, activation='tanh'),
        ])

        if droprate > 0:
            self.classifier = tf.keras.Sequential([
                tf.keras.layers.Dropout(droprate),
                tf.keras.layers.Dense(num_classes)
            ])
        else:
            self.classifier = tf.keras.layers.Dense(num_classes)

    def call(self, x):
        hidden_out = self.features(x)
        return self.classifier(hidden_out)


def create_purchase_classifier(num_classes=100, droprate=0):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(1024, activation='tanh', input_shape=(600,)),
        tf.keras.layers.Dense(512, activation='tanh'),
        tf.keras.layers.Dense(256, activation='tanh'),
        tf.keras.layers.Dense(128, activation='tanh'),
    ])

    if droprate > 0:
        model.add(tf.keras.layers.Dropout(droprate))

    model.add(tf.keras.layers.Dense(num_classes))

    return model


def regularized_purchase_classifier(num_classes=100, dropout=None, regularizer=None):
    input_data = Input(shape=(600,))
    x = Dense(1024, activation='tanh')(input_data)
    x = Dense(512, activation='tanh')(x)
    x = Dense(256, activation='tanh')(x)
    if dropout is not None:
        x = Dropout(dropout)(x)
    x = Dense(128, activation='tanh', kernel_regularizer=regularizer)(x)
    if dropout is not None:
        x = Dropout(dropout)(x)
    output = Dense(num_classes)(x)
    model = Model(input_data, output)
    return model


def vgg19_scratch(output, dropout=None, regularizer=None):
    # input
    input = Input(shape=(32, 32, 3))
    # 1st Conv Block

    x = Conv2D(filters=64, kernel_size=3, padding='same', activation='relu')(input)
    x = Conv2D(filters=64, kernel_size=3, padding='same', activation='relu')(x)
    x = MaxPool2D(pool_size=2, strides=2, padding='same')(x)
    # 2nd Conv Block

    x = Conv2D(filters=128, kernel_size=3, padding='same', activation='relu')(x)
    x = Conv2D(filters=128, kernel_size=3, padding='same', activation='relu')(x)
    x = MaxPool2D(pool_size=2, strides=2, padding='same')(x)
    # 3rd Conv block

    x = Conv2D(filters=256, kernel_size=3, padding='same', activation='relu')(x)
    x = Conv2D(filters=256, kernel_size=3, padding='same', activation='relu')(x)
    x = Conv2D(filters=256, kernel_size=3, padding='same', activation='relu')(x)
    x = MaxPool2D(pool_size=2, strides=2, padding='same')(x)
    # 4th Conv block

    x = Conv2D(filters=512, kernel_size=3, padding='same', activation='relu')(x)
    x = Conv2D(filters=512, kernel_size=3, padding='same', activation='relu')(x)
    x = Conv2D(filters=512, kernel_size=3, padding='same', activation='relu')(x)
    x = MaxPool2D(pool_size=2, strides=2, padding='same')(x)

    # 5th Conv block

    x = Conv2D(filters=512, kernel_size=3, padding='same', activation='relu')(x)
    x = Conv2D(filters=512, kernel_size=3, padding='same', activation='relu')(x)
    x = Conv2D(filters=512, kernel_size=3, padding='same', activation='relu')(x)
    x = MaxPool2D(pool_size=2, strides=2, padding='same')(x)
    # Fully connected layers

    x = Flatten()(x)
    x = Dense(units=4096, activation='relu', kernel_regularizer=regularizer)(x)
    if dropout is not None:
        x = Dropout(dropout)(x)
    x = Dense(units=4096, activation='relu', kernel_regularizer=regularizer)(x)
    if dropout is not None:
        x = Dropout(dropout)(x)
    output = Dense(units=output)(x)
    # creating the model

    model = Model(inputs=input, outputs=output)
    model.summary()
    return model


class Attack(tf.keras.Model):
    def __init__(self, input_dim, num_classes=1, hiddens=[100, 1024, 512, 64]):
        super(Attack, self).__init__()
        self.layers_list = []
        for i in range(len(hiddens)):
            if i == 0:
                layer = tf.keras.layers.Dense(hiddens[i], input_shape=(input_dim,))
            else:
                layer = tf.keras.layers.Dense(hiddens[i])
            self.layers_list.append(layer)
        self.last_layer = tf.keras.layers.Dense(num_classes)
        self.relu = tf.keras.activations.relu
        self.sigmoid = tf.keras.activations.sigmoid

    def call(self, x):
        output = x
        for layer in self.layers_list:
            output = self.relu(layer(output))
        output = self.last_layer(output)
        output = self.sigmoid(output)
        return output


def inference_model(input1_shape, input2_shape, output_shape=1):
    input1 = Input(shape=input1_shape)
    # Define the input layer

    # Define the first head of the network
    x1 = tf.keras.layers.Dense(100, activation='relu')(input1)
    x1 = tf.keras.layers.Dense(1024, activation='relu')(x1)
    x1 = tf.keras.layers.Dense(512, activation='relu')(x1)
    x1 = tf.keras.layers.Dense(64, activation='relu')(x1)

    input2 = Input(shape=input2_shape)

    # Define the second head of the network
    x2 = tf.keras.layers.Dense(100, activation='relu')(input2)
    x2 = tf.keras.layers.Dense(512, activation='relu')(x2)
    x2 = tf.keras.layers.Dense(64, activation='relu')(x2)

    # Concatenate the output of the two heads
    concat = tf.keras.layers.Concatenate()([x1, x2])
    x = tf.keras.layers.Dense(256, activation='relu')(concat)
    x = tf.keras.layers.Dense(64, activation='relu')(x)

    # Pass the concatenated output to another Dense layer for binary classification
    output = tf.keras.layers.Dense(output_shape, activation='sigmoid')(x)

    # Define the model with the input layer and the final output layer
    model = tf.keras.Model(inputs=[input1, input2], outputs=output)
    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Print the model summary
    model.summary()
    return model
