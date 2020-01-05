# -*- coding: utf-8 -*-
# @Time    : 2020/1/5 
# @Author  : zhaowq
# @desc: youtube 召回模型

import tensorflow as tf
import tools.utils as ut
from layers.embedding_layer import *

class YoutubeRecallModel:
    def __init__(self, params):
        self.video_total_num = params.get("video_total_num", 0)#
        self.search_total_num = params.get("search_total_num",0)#
        self.hidden_units = params.get("hidden_units", [512,256,128])
        self.learning_rate = params.get("learning_rate", 0.01)
        self.is_training = params.get("is_training", True)
        self.batch_size = params.get("batch_size", 128)
        self.epoch = params.get("epoch", 10)
        self.class_num = params.get("class_num", 2)
        self.embedding_size = params.get("embedding_size",8)

        self.global_step = tf.Variable(0, trainable = False)

    def build_model(self):
        #初始化数据入口，placeholder名字要唯一，不能更改ph任何信息
        self.video_ids_ph = tf.placeholder(tf.int32, shape=[None, None], name='video_ids')
        self.search_id_ph = tf.placeholder(tf.int32, shape=[None], name='search_id')
        self.age_ph = tf.placeholder(tf.float32, shape=[None], name='age')
        self.gender_ph = tf.placeholder(tf.float32, shape=[None], name='gender')
        self.label_ph = tf.placeholder(tf.float32, shape=[None], name='label_ph')

        #初始化视频embedding、搜索条件的embedding, concat两个embedding和age、gender
        video_embedding = embedding_layer(self.video_ids_ph, "video_embedding", self.video_total_num, self.embedding_size)
        search_embedding = embedding_layer(self.search_id_ph, "search_embedding", self.search_total_num, self.embedding_size)

        input = tf.concat([tf.reshape(tf.reduce_mean(video_vecs, axis=1), shape=[-1, 1]),
                           tf.reshape(search_vec, shape=[-1, 1]), 
                           tf.reshape(self.age_ph, shape=[-1, 1]),
                           tf.reshape(self.gender_ph, shape=[-1, 1])],
                        axis=1)
        
        #未完待续