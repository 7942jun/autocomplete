import tensorflow as tf

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
        prediction_check = tf.equal(prediction, self.model.Y)
        accuracy = tf.reduce_mean(tf.cast(prediction_check, tf.float32))

        input_batch, target_batch = self.data.batch()

        predict, accuracy_val = self.sess.run([prediction, accuracy], feed_dict={self.model.X: input_batch, self.model.Y: target_batch})

        predict_words = []
        for idx, val in enumerate(self.data.word_data):
        	last_char = self.data.char_arr[predict[idx]]
        	predict_words.append(val[:3] + last_char)

        print('\n===예측 결과===')
        print('입력값: ', [w[:3] + ' ' for w in self.data.word_data])
        print('예측값: ', predict_words)
        print('정확도: ', accuracy_val)
