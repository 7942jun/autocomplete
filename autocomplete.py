#-*-coding:utf-8-*-
import tensorflow as tf
import numpy as np

from data_loader.data import Data
from config.config import Config
from models.autocomplete_model import ACModel
from trainers.autocomplete_trainer import ACTrainer

data = Data()
config = Config()
model = ACModel(config)
sess = tf.Session()
trainer = ACTrainer(sess, model, data, config)

trainer.train()
trainer.test()
