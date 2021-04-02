import tensorflow as tf


class Inputs(object):
    def __init__(self, img1: tf.Tensor, img2: tf.Tensor, label: tf.Tensor):
        self.img1 = img1
        self.img2 = img2
        self.label = label

class Truss(Inputs):
    def __init__(self,E,A,L):
        pass

class ElemSet(object):
    img1_resized = 'img1_resized'
    img2_resized = 'img2_resized'
    label = 'same_person'

    def __init__(self, generator=PairGenerator()):
        self.next_element = self.build_iterator(generator)

    def build_iterator(self, pair_gen: PairGenerator):
        batch_size = 10
        prefetch_batch_buffer = 5

        dataset = tf.data.Dataset.from_generator(pair_gen.get_next_pair,
                                                 output_types={PairGenerator.person1: tf.string,
                                                               PairGenerator.person2: tf.string,
                                                               PairGenerator.label: tf.bool})
        dataset = dataset.map(self._read_image_and_resize)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(prefetch_batch_buffer)
        iter = dataset.make_one_shot_iterator()
        element = iter.get_next()

        return Inputs(element[self.img1_resized],
                      element[self.img2_resized],
                      element[PairGenerator.label])

    def _read_image_and_resize(self, pair_element):
        target_size = [128, 128]
        # read images from disk
        img1_file = tf.read_file(pair_element[PairGenerator.person1])
        img2_file = tf.read_file(pair_element[PairGenerator.person2])
        img1 = tf.image.decode_image(img1_file)
        img2 = tf.image.decode_image(img2_file)

        # let tensorflow know that the loaded images have unknown dimensions, and 3 color channels (rgb)
        img1.set_shape([None, None, 3])
        img2.set_shape([None, None, 3])

        # resize to model input size
        img1_resized = tf.image.resize_images(img1, target_size)
        img2_resized = tf.image.resize_images(img2, target_size)

        pair_element[self.img1_resized] = img1_resized
        pair_element[self.img2_resized] = img2_resized
        pair_element[self.label] = tf.cast(pair_element[PairGenerator.label], tf.float32)

        return pair_element


class Model(object):
    def __init__(self, inputs: Inputs):
        self.inputs = inputs
        self.predictions = self.predict(inputs)
        self.loss = self.calculate_loss(inputs, self.predictions)
        self.opt_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.loss)

    def predict(self, inputs: Inputs):
        with tf.name_scope("image_substraction"):
            img_diff = (inputs.img1 - inputs.img2)
            x = img_diff
        with tf.name_scope('conv_relu_maxpool'):
            for conv_layer_i in range(5):
                x = tf.layers.conv2d(x,
                                     filters=20 * (conv_layer_i + 1),
                                     kernel_size=3,
                                     activation=tf.nn.relu)
                x = tf.layers.max_pooling2d(x,
                                            pool_size=3,
                                            strides=2)
        with tf.name_scope('fully_connected'):
            for conv_layer_i in range(1):
                x = tf.layers.dense(x,
                                    units=200,
                                    activation=tf.nn.relu)
        with tf.name_scope('linear_predict'):
            predicted_logits = tf.layers.dense(x, 1, activation=None)

        return tf.squeeze(predicted_logits)

    def calculate_loss(self, inputs: Inputs, prediction_logits: Tensor):
        with tf.name_scope('calculate_loss'):
            return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=inputs.label,
                                                                          logits=prediction_logits))

