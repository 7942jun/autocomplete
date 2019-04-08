import tensorflow as tf

class ACModel:
    def __init__(self, config):
        self.config = config
        self.build_model()

    def build_model(self):
        self.X = tf.placeholder(tf.float32, [None, self.config.n_step, self.config.n_input])
        self.Y = tf.placeholder(tf.int32, [None])
        W = tf.Variable(tf.random_normal([self.config.n_hidden, self.config.n_class]))
        b = tf.Variable(tf.random_normal([self.config.n_class]))

        # TODO: 이해하기
        cell1 = tf.nn.rnn_cell.BasicLSTMCell(self.config.n_hidden)
        cell1 = tf.nn.rnn_cell.DropoutWrapper(cell1, output_keep_prob=0.5)
        cell2 = tf.nn.rnn_cell.BasicLSTMCell(self.config.n_hidden)

        multi_cell = tf.nn.rnn_cell.MultiRNNCell([cell1, cell2])

        outputs, states = tf.nn.dynamic_rnn(multi_cell, self.X, dtype=tf.float32)

        outputs = tf.transpose(outputs, [1, 0, 2])
        outputs = outputs[-1]
        self.output = tf.matmul(outputs, W) + b

        entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.output, labels=self.Y)
        self.cost = tf.reduce_mean(entropy)

        self.optimizer = tf.train.AdamOptimizer(self.config.learning_rate).minimize(self.cost)
