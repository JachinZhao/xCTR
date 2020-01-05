# -*- coding: utf-8 -*-
# @Time    : 2020/1/5 
# @Author  : zhaowq
# @desc: 常用工具函数集合


#所有的算法都会用到的参数，要在训练模型的命令行中传入 #遇到再继续添加
#如checkpointdir、learning rate、max_step
def get_tfapp_flags(flags):
    flags.DEFINE_string("checkpoint_dir", None, "save model dir")
    flags.DEFINE_string("task", "train", "trian、eval、pred")
    flags.DEFINE_float("learing_rate", 0.01, "learing rate")
    #flags.DEFINE_boolean()
    flags.DEFINE_integer("epoch",None, "")
    return flags.FLAGS