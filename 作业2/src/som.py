import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm as cm
import pandas as pd
from optparse import OptionParser
import os
import time


class SOM(object):
    """
    2-D Self-Organizing Map with Gaussian Neighbourhood function and linearly decreasing learning rate.
    """
    # To check if the SOM has been trained
    _trained = False

    def __init__(self, output_width, output_length, input_dimension, n_iterations=100, learning_rate=0.3,
                 influence_radius=None):
        """
        Initializes all necessary components of the TensorFlow Graph.
        :param output_width: the SOM's output length
        :param output_length: the SOM's output width
        :param input_dimension: dimension of training inputs
        :param n_iterations: learning iteration count
        :param learning_rate: initial iteration-no-based learning rate
        :param influence_radius: initial neighbourhood value, denoting the influence radius of the best matching unit,
         max(output_width, output_length) by default
        """
        self._output_width = output_width
        self._output_length = output_length
        influence_radius = max(output_width, output_length) / 2.0 if influence_radius is None else float(
            influence_radius)
        self._n_iterations = abs(int(n_iterations))
        output_size = output_width * output_length

        with tf.Graph().as_default():
            self._weight_vectors = tf.Variable(tf.random_normal([output_size, input_dimension]))

            self._location_vectors = tf.constant(np.array(list(self._neuron_locations(output_width, output_length))))

            self._input_vector = tf.placeholder('float32', [input_dimension])
            self._input_iteration_number = tf.placeholder('float32')

            self._heat = tf.Variable(tf.to_float(np.ones([output_size])))

            # To compute the Best Matching Unit given a vector, simply calculates the Euclidean distance between
            # every neuron's weight vector and the input, and returns the index of the neuron which gives the least
            # value
            bmu_index = tf.argmin(tf.sqrt(tf.reduce_sum(
                tf.pow(tf.subtract(self._weight_vectors, tf.stack(
                    [self._input_vector for i in range(output_size)])), 2), 1)), 0)

            # This will extract the location of the BMU based on the BMU's index
            slice_input = tf.pad(tf.reshape(bmu_index, [1]),
                                 np.array([[0, 1]]))
            bmu_loc = tf.reshape(tf.slice(self._location_vectors, slice_input,
                                          tf.constant(np.array([1, 2], dtype=np.int64))), [2])

            # To compute the alpha and sigma values based on iteration number
            learning_rate_op = tf.subtract(1.0, tf.div(self._input_iteration_number, self._n_iterations))
            _alpha_op = tf.multiply(learning_rate, learning_rate_op)
            _sigma_op = tf.multiply(influence_radius, learning_rate_op)

            # Construct the op that will generate a vector with learning rates for all neurons, based on iteration
            # number and location wrt BMU.
            bmu_distance_squares = tf.reduce_sum(tf.pow(tf.subtract(
                self._location_vectors, tf.stack([bmu_loc for i in range(output_size)])), 2), 1)
            neighbourhood_func = tf.exp(tf.negative(tf.sqrt(tf.div(tf.cast(
                bmu_distance_squares, 'float32'), tf.pow(_sigma_op, 2)))))
            learning_rate_op = tf.multiply(_alpha_op, neighbourhood_func)

            # Finally, the op that will use learning_rate_op to update the weight vectors of all neurons based on a
            # particular input
            learning_rate_multiplier = tf.stack(
                [tf.tile(tf.slice(learning_rate_op, np.array([i]), np.array([1])), [input_dimension]) for i in
                 range(output_size)])
            weight_delta = tf.multiply(learning_rate_multiplier, tf.subtract(tf.stack(
                [self._input_vector for i in range(output_size)]), self._weight_vectors))
            new_weight_op = tf.add(self._weight_vectors, weight_delta)
            self._training_op = tf.assign(self._weight_vectors, new_weight_op)

            heat_rate = tf.to_float(tf.div(1.0, self._n_iterations))
            heat_alpha = tf.multiply(learning_rate, heat_rate)
            heat_sigma = tf.multiply(influence_radius, heat_rate)
            heat_neighbourhood = tf.exp(tf.negative(tf.sqrt(tf.div(tf.cast(
                bmu_distance_squares, 'float32'), tf.pow(heat_sigma, 2)))))
            heat_rate = tf.multiply(heat_alpha, heat_neighbourhood)
            new_heat = tf.multiply(tf.add(heat_rate, np.ones([output_size])), self._heat)
            self._update_heat_op = tf.assign(self._heat, new_heat)

            self._sess = tf.Session()
            self._sess.run(tf.global_variables_initializer())

    @staticmethod
    def _neuron_locations(length, width):
        """
        Yields the 2-D locations of the neurons in the SOM.
        """
        # Nested iterations over both dimensions
        # to generate all 2-D locations in the map
        for i in range(length):
            for j in range(width):
                yield np.array([i, j])

    def train_step(self, input_vectors, iter_no):
        """
        Trains the SOM one step.
        :param input_vectors: randomly picked input vectors
        :param iter_no:
        :return:
        """
        for input_vector in input_vectors:
            self._sess.run(self._training_op,
                           feed_dict={self._input_vector: input_vector,
                                      self._input_iteration_number: iter_no})

    def train(self, input_vectors):
        """
        Trains the SOM. Current weight vectors for all neurons (initially random) are taken as starting conditions
        for training.
        :param input_vectors: an iterable of 1-D NumPy arrays with dimensionality as provided during initialization
        of this SOM.
        """
        print('Training begins...')
        # Training iterations
        for iter_no in range(self._n_iterations):
            start = time.time()
            self.train_step(input_vectors, iter_no)
            print('Iter #{}/{}, train time:{}s'.format(iter_no + 1, self._n_iterations, time.time() - start))
        self._trained = True

    def get_heat_vec(self, input_vectors):
        """
        Returns a list of 'length' lists, with each inner list containing the 'width' corresponding centroid
        locations as 1-D NumPy arrays.
        """
        if not self._trained:
            raise ValueError('SOM not trained yet')
        for vector in input_vectors:
            self._sess.run(self._update_heat_op, feed_dict={self._input_vector: vector})
        heat = list(self._sess.run(self._heat))
        # normalize the heat list
        max_val = np.nanmax(heat)
        heat = [1000 * x / max_val for x in heat]
        v_min_val = int(np.nanmin(heat))
        v_max_val = int(np.nanmax(heat) + 1)
        heat = np.reshape(heat, [self._output_width, self._output_length])
        return heat, v_min_val, v_max_val

    def save_to_file(self, file_name):
        weight_vector = self._sess.run(self._weight_vectors)
        np.save(file_name, weight_vector)

    def load_from_file(self, file_name):
        if not os.path.exists(file_name):
            raise FileNotFoundError('No weight_vectors file!')
        self._weight_vectors = tf.Variable(np.load(file_name))
        self._trained = True


def construct_user_behavior_array(user_behavior_file, size):
    if not os.path.exists(user_behavior_file):
        raise FileNotFoundError('No user behavior file found!')
    print('Constructing user behavior array...')
    df = pd.read_csv(user_behavior_file)
    df = df.filter(items=['user_id', 'item_id', 'action_type'])
    df['user_id'] = df['user_id'].astype(int)
    df['item_id'] = df['item_id'].astype(int)
    # 0 - no action, 1 - click, 2 - add to cart, 3 - purchase, 4 - add to favourite
    df['action_type'] += 1
    # normalize
    df['action_type'] /= 4.0
    # shuffle
    df.sample(frac=1)
    # construct a very sparse matrix (user_count * item_count)
    groups = df.groupby(['user_id'])
    user_map = {}
    item_map = {}
    output = np.zeros([min(len(set(df['user_id'])), size), len(set(df['item_id']))])
    for _, items in groups:
        if len(user_map) >= size:
            break
        user_id = items['user_id'].values[0]
        if user_id not in user_map.keys():
            user_map[user_id] = len(user_map)
        item_list = items['item_id'].values
        action_list = items['action_type'].values
        for i in range(len(item_list)):
            if item_list[i] not in item_map.keys():
                item_map[item_list[i]] = len(item_map)
            output[user_map[user_id]][item_map[item_list[i]]] = action_list[i]
        print('User behaviour array #{}/{}'.format(len(user_map), size))
    return output[:, :len(item_map)]


def save_result(plot, destination='result/som_result.png'):
    if not os.path.exists('result'):
        os.makedirs('result')
    plot.savefig(destination)


if __name__ == '__main__':
    opt_parser = OptionParser()
    opt_parser.add_option('-f', '--input_file',
                          dest='input',
                          help='user behavior record csv',
                          default='data_set/user_log_format1.csv')
    opt_parser.add_option('-w', '--som_width',
                          dest='som_width',
                          help='width of som network',
                          default=50,
                          type='int')
    opt_parser.add_option('-l', '--som_length',
                          dest='som_length',
                          help='length of som network',
                          default=50,
                          type='int')
    opt_parser.add_option('-i', '--iteration_count',
                          dest='iteration_count',
                          help='iteration count',
                          default=10,
                          type='int')
    opt_parser.add_option('-b', '--batch_size',
                          dest='batch_size',
                          help='batch size',
                          default=128,
                          type='int')
    (options, args) = opt_parser.parse_args()

    if os.path.exists('data_set/user_behavior.npy'):
        user_behavior = np.load('data_set/user_behavior.npy')
        print('User behavior data loaded!')
    else:
        user_behavior = construct_user_behavior_array(options.input, options.batch_size)
        np.save('data_set/user_behavior.npy', user_behavior)
        print('User behavior data saved!')

    som = SOM(options.som_width, options.som_length, user_behavior[0].size, options.iteration_count)
    if os.path.exists('data_set/weight_vectors.npy'):
        som.load_from_file('data_set/weight_vectors.npy')
        print('SOM weight vectors loaded!')
    else:
        som.train(user_behavior)
        som.save_to_file('data_set/weight_vectors.npy')
        print('SOM weight vectors saved!')

    # Get output grid
    heat_vec, v_min, v_max = som.get_heat_vec(user_behavior)

    # Plot
    fig = plt.figure(facecolor='w')
    ax1 = fig.add_subplot(1, 1, 1)
    cmap = cm.get_cmap('nipy_spectral', 1000)
    m = ax1.imshow(heat_vec, interpolation="nearest", cmap=cmap, aspect='auto', vmin=v_min, vmax=v_max)
    cb = plt.colorbar(mappable=m, cax=None, ax=None)
    plt.title('User behavior SOM')
    save_result(plt)
    plt.show()
