# -*- coding: utf-8 -*-
# @Time    : 2020/1/5 
# @Author  : zhaowq
# @desc: 输入是string的时候要进行embedding

def embedding_layer(feature_column, embedding_name, hash_bucket_size, embedding_size, value_scope="embedding_input"):
    with tf.variable_scope(value_scope, reuse = tf.AUTO_REUSE):
        embedding_wight = tf.get_variable(embedding_name, shape=[hash_bucket_size, embedding_size], dtype=tf.float32,
                                            initializer = tf.variance_scaling_initializer())
        x_embedding = tf.nn.embedding_lookup(embedding_wight, feature_column)
    return x_embedding

#hash_bucket_size 桶的个数
def spase_embedding_layer(string_feature,embedding_name, hash_bucket_size, embedding_size, value_scope="embedding_input"):
    with tf.variable_scope(value_scope, reuse = tf.AUTO_REUSE):
        feature_column = tf.string_to_hash_bucket_fast(string_feature,hash_bucket_size)
        embedding_wight = tf.get_variable(embedding_name, shape=[hash_bucket_size, embedding_size], dtype=tf.float32,
                                            initializer = tf.variance_scaling_initializer())
        x_embedding = tf.nn.embedding_lookup(embedding_wight, feature_column)
    return x_embedding