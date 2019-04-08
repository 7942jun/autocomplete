import numpy as np


class Data:
    def __init__(self):
        self.word_data = ['word', 'wood', 'best', 'hide', 'tool', 'told',
                          'deep', 'dive', 'cold', 'cool', 'load', 'love', 'kiss', 'kind']
        self.char_arr = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k',
                         'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

    def batch(self):
        input_batch = []
        target_batch = []
        num_dic = {n: i for i, n in enumerate(self.char_arr)}

        for seq in self.word_data:
            input = [num_dic[n] for n in seq[:-1]]
            target = num_dic[seq[-1]]

            dic_len = len(self.char_arr)
            input_batch.append(np.eye(dic_len)[input])

            target_batch.append(target)

        return input_batch, target_batch
