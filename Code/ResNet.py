import tensorflow as tf

from tensorflow.keras import layers

class ResBlock(tf.keras.Model):
    """Defines the Residual Block used in the ResNet architecture.

    Composed of 2 convlutional layers pair with batch normalization and relu activation functions,
    Dropout added as regularization and a residual connection.

    Keyword arguments:
    --init--
    channels -- the number of output channels.
    stride -- specification of the stride used on the convolutions, used for output size manipulation.

    --call--
    input -- Data input to the residual block.

    Returns:
    out -- Data output of the residual block.
    """
    def __init__(self, channels, stride=1):
        super().__init__(name='ResBlock')
        self.flag = (stride != 1)
        self.conv_1 = layers.Conv2D(channels, 3, stride, padding='same')
        self.batch_norm_1 = layers.BatchNormalization()
        self.conv_2 = layers.Conv2D(channels, 3, padding='same')
        self.batch_norm_2 = layers.BatchNormalization()
        self.dropout = layers.Dropout(0.2)
        self.relu = layers.ReLU()
        if self.flag:
            self.batch_norm_3 = layers.BatchNormalization()
            self.conv_3 = layers.Conv2D(channels, 1, stride)

    def call(self, x_in):
        x = self.conv_1(x_in)
        x = self.batch_norm_1(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.conv_2(x)
        x = self.batch_norm_2(x)
        x = self.dropout(x)
        if self.flag:
            x_in = self.conv_3(x_in)
            x_in = self.batch_norm_3(x_in)
        x = layers.add([x, x_in])
        out = self.relu(x)
        return out


class ResNet(tf.keras.Model):
    """Definition of the ResNet34 architecture.

    composed of [3, 4, 6, 3] sequence of residual blocks

    Keyword arguments:
    --init--
    input_shape -- self explanatory. Needed for future model.summary().
    output_size -- number of classes in the classification problem.

    --call--
    input -- input image of initialized input_shape.

    Returns:
    output -- Probabilities vector, presenting each of the classes probabilities of appearing on the input image.
    """
    def __init__(self, input_shape = (512, 512, 3), output_size = 14, **kwargs):
        super().__init__(name='ResNet34', **kwargs)
        self.input_layer = layers.Input(input_shape)
        self.conv1 = layers.Conv2D(64, 7, 2, padding='same')
        self.batch_norm = layers.BatchNormalization()
        self.relu = layers.ReLU()
        self.max_pool1 = layers.MaxPooling2D(3, 2)
        self.dropout = layers.Dropout(0.5)

        self.conv2_1 = ResBlock(64)
        self.conv2_2 = ResBlock(64)
        self.conv2_3 = ResBlock(64)

        self.conv3_1 = ResBlock(128, 2)
        self.conv3_2 = ResBlock(128)
        self.conv3_3 = ResBlock(128)
        self.conv3_4 = ResBlock(128)

        self.conv4_1 = ResBlock(256, 2)
        self.conv4_2 = ResBlock(256)
        self.conv4_3 = ResBlock(256)
        self.conv4_4 = ResBlock(256)
        self.conv4_5 = ResBlock(256)
        self.conv4_6 = ResBlock(256)

        self.conv5_1 = ResBlock(512, 2)
        self.conv5_2 = ResBlock(512)
        self.conv5_3 = ResBlock(512)

        self.pool = layers.GlobalAveragePooling2D()
        self.fc1 = layers.Dense(128, activation='relu')
        self.fc2 = layers.Dense(output_size, activation='sigmoid')

    def call(self, x):
        x = self.conv1(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.max_pool1(x)

        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.conv2_3(x)

        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        x = self.conv3_4(x)

        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv4_3(x)
        x = self.conv4_4(x)
        x = self.conv4_5(x)
        x = self.conv4_6(x)

        x = self.conv5_1(x)
        x = self.conv5_2(x)
        x = self.conv5_3(x)

        x = self.pool(x)
        x = self.fc1(x)
        x = self.dropout(x)
        output = self.fc2(x)

        return output



def resnet_def(input_shape = (512, 512, 3), output_size = 14, verbose = True):
    """Definition and building of the model.

    Args:
    input_shape(tuple) -- Shape of the input image to the resnet model
    output_shape(int) -- number of classes in the classification problem.
    verbose(bool) -- print the summary of the model
    """

    model = ResNet(input_shape=input_shape, output_size=output_size)
    model.build(input_shape = (None, 512, 512, 3))
    if verbose: print(model.summary())

    return model
