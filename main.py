# -*- coding: utf-8 -*-
# @Time    : 2020/1/5 
# @Author  : zhaowq
# @desc: youtube dnn main


from __future__ import divisioin, print_function
import tensorflow as tf
import tools.utils as ut

def main():
    #调用模型等完成训练，验证和预测
    model_dir='sss'
    


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    flags = ut.get_tfapp_flags(flags=tf.app.flags)
    tf.app.run()