import tensorflow as tf
import numpy as np


class ACTrainer:
    def __init__(self, sess, model, data, config):
        self.sess = sess
        self.model = model
        self.data = data
        self.config = config
        self.init = tf.global_variables_initializer()
        self.sess.run(self.init)

    def train(self):
        input_batch, target_batch = self.data.batch()

        for epoch in range(self.config.total_epoch):
            _, loss = self.sess.run([self.model.optimizer, self.model.cost], feed_dict={
                                    self.model.X: input_batch, self.model.Y: target_batch})

            print('Epoch:', '%04d' % (epoch + 1),
                  'cost =', '{:.6f}'.format(loss))

        print('최적화 완료!')

    def test(self):
        prediction = tf.cast(tf.argmax(self.model.output, 1), tf.int32)
        num_dic = {n: i for i, n in enumerate(self.data.char_arr)}

        while True:
            user_input = input('Enter part of word(enter q to quit): ')
            if user_input == 'q':
                break
            user_input_decode = [num_dic[n] for n in user_input]
            input_batch = [np.eye(self.config.dic_len)[user_input_decode]]

            predict = self.sess.run(prediction, feed_dict={
                                    self.model.X: input_batch})

            last_char = self.data.char_arr[predict[0]]
            predict_word = user_input + last_char

            print('\n===예측 결과===')
            print('입력값: ', user_input)
            print('예측값: ', predict_word)
            print('')
