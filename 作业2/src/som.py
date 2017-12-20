import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from optparse import OptionParser
import os


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

        with tf.Graph().as_default():
            self._weight_vectors = tf.Variable(tf.random_normal([output_width * output_length, input_dimension]))

            self._location_vectors = tf.constant(np.array(list(self._neuron_locations(output_width, output_length))))

            self._input_vector = tf.placeholder("float", [input_dimension])
            self._input_iteration_number = tf.placeholder("float")

            # To compute the Best Matching Unit given a vector, simply calculates the Euclidean distance between
            # every neuron's weight vector and the input, and returns the index of the neuron which gives the least
            # value
            bmu_index = tf.argmin(tf.sqrt(tf.reduce_sum(
                tf.pow(tf.subtract(self._weight_vectors, tf.stack(
                    [self._input_vector for i in range(output_width * output_length)])), 2), 1)), 0)

            # This will extract the location of the BMU based on the BMU's index
            slice_input = tf.pad(tf.reshape(bmu_index, [1]),
                                 np.array([[0, 1]]))
            bmu_loc = tf.reshape(tf.slice(self._location_vectors, slice_input,
                                          tf.constant(np.array([1, 2]))), [2])

            # To compute the alpha and sigma values based on iteration number
            learning_rate_op = tf.subtract(1.0, tf.div(self._input_iteration_number, self._n_iterations))
            _alpha_op = tf.multiply(learning_rate, learning_rate_op)
            _sigma_op = tf.multiply(influence_radius, learning_rate_op)

            # Construct the op that will generate a vector with learning rates for all neurons, based on iteration
            # number and location wrt BMU.
            bmu_distance_squares = tf.reduce_sum(tf.pow(tf.subtract(
                self._location_vectors, tf.stack([bmu_loc for i in range(output_width * output_length)])), 2), 1)
            neighbourhood_func = tf.exp(tf.negative(tf.div(tf.cast(
                bmu_distance_squares, "float32"), tf.pow(_sigma_op, 2))))
            learning_rate_op = tf.multiply(_alpha_op, neighbourhood_func)

            # Finally, the op that will use learning_rate_op to update the weight vectors of all neurons based on a
            # particular input
            learning_rate_multiplier = tf.stack(
                [tf.tile(tf.slice(learning_rate_op, np.array([i]), np.array([1])), [input_dimension]) for i in
                 range(output_width * output_length)])
            weight_delta = tf.multiply(
                learning_rate_multiplier,
                tf.subtract(tf.stack([self._input_vector for i in range(output_width * output_length)]),
                            self._weight_vectors))
            new_weight_op = tf.add(self._weight_vectors, weight_delta)
            self._training_op = tf.assign(self._weight_vectors, new_weight_op)

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

    def train(self, input_vectors):
        """
        Trains the SOM. Current weight vectors for all neurons (initially random) are taken as starting conditions
        for training.
        :param input_vectors: an iterable of 1-D NumPy arrays with dimensionality as provided during initialization
        of this SOM.
        """
        # Training iterations
        for iter_no in range(self._n_iterations):
            # Train with each vector one by one
            for input_vector in input_vectors:
                self._sess.run(self._training_op,
                               feed_dict={self._input_vector: input_vector,
                                          self._input_iteration_number: iter_no})
        self._trained = True

    def get_centroids(self):
        """
        Returns a list of 'length' lists, with each inner list containing the 'width' corresponding centroid
        locations as 1-D NumPy arrays.
        """
        if not self._trained:
            raise ValueError("SOM not trained yet")
        centroid_grid = [[] for i in range(self._output_width)]
        weight = list(self._sess.run(self._weight_vectors))
        locations = list(self._sess.run(self._location_vectors))
        for i, loc in enumerate(locations):
            centroid_grid[loc[0]].append(weight[i])
        return centroid_grid

    def map_vectors(self, input_vectors):
        """
        Maps each input vector to the relevant neuron in the SOM grid.
        :param input_vectors: an iterable of 1-D NumPy arrays with dimensionality as provided during initialization
        of this SOM.
        :return: a list of 1-D NumPy arrays containing (row, column) info for each input vector(in the same order),
        corresponding to mapped neuron.
        """
        if not self._trained:
            raise ValueError("SOM not trained yet")
        weight = list(self._sess.run(self._weight_vectors))
        locations = list(self._sess.run(self._location_vectors))
        to_return = []
        for vector in input_vectors:
            min_index = min([i for i in range(len(weight))], key=lambda x: np.linalg.norm(vector - weight[x]))
            to_return.append(locations[min_index])
        return to_return


def construct_user_behavior_array(user_behavior_file):
    if not os.path.exists(user_behavior_file):
        raise FileNotFoundError('No user behavior file found!')
    df = pd.read_csv(user_behavior_file)
    df = df.filter(items=['user_id', 'item_id', 'action_type'])
    df['user_id'] = df['user_id'].astype(int)
    df['item_id'] = df['item_id'].astype(int)
    # 0 - no action, 1 - click, 2 - add to cart, 3 - purchase, 4 - add to favourite
    df['action_type'] += 1
    # normalize
    df['action_type'] /= 4.0
    # construct a very sparse matrix (user_count * item_count)
    users = set(df['user_id'])
    items = set(df['item_id'])
    user_map = {}
    item_map = {}
    output = np.zeros([len(users), len(items)])
    for index, row in df.iterrows():
        if row['user_id'] not in user_map.keys():
            user_map[row['user_id']] = len(user_map)
        if row['item_id'] not in item_map.keys():
            item_map[row['item_id']] = len(item_map)
        output[user_map[row['user_id']]][item_map[row['item_id']]] = row['action_type']
    return output


def save_result(plot, destination='result/som_result.jpg'):
    if not os.path.exists('result'):
        os.makedirs('result')
    plot.savefig(destination)
    pass


if __name__ == '__main__':
    opt_parser = OptionParser()
    opt_parser.add_option('-f', '--input_file',
                          dest='input',
                          help='user behavior record csv',
                          default='data_set/user_log_format_temp.csv')
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
                          default=400,
                          type='int')
    (options, args) = opt_parser.parse_args()

    user_behavior = construct_user_behavior_array(options.input)

    som = SOM(options.som_width, options.som_length, user_behavior[0].size, options.iteration_count)
    som.train(user_behavior)

    # Get output grid
    image_grid = som.get_centroids()

    # Plot
    plt.imshow(image_grid)
    plt.title('User behavior SOM')
    save_result(plt)
    plt.show()
