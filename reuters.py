# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 23:01:39 2018

@author: yuxi
"""

#加载reuters数据集
from keras.datasets import reuters
(train_data,train_labels),(test_data,test_labels)=reuters.load_data(num_words=10000)

import numpy as np
def vectorize_sequences(sequences,dimension=10000):
    results=np.zeros((len(sequences),dimension))
    for i,sequence in enumerate(sequences):
        results[i,sequence]=1.
    return results

#将训练数据和测试数据向量化
x_train =vectorize_sequences(train_data)
x_test=vectorize_sequences(test_data)


def to_one_hot (labels,dimension=46):
    results=np.zeros((len(labels),dimension))
    for i,label in enumerate(labels):
        results[i,label]=1.
    return results

one_hot_train_labels =to_one_hot(train_labels)
one_hot_test_labels =to_one_hot(test_labels)



# =============================================================================
# #Keras 内置方法实现分类编码
#  from keras.utils.np_utils import to_categorical
#  one_hot_train_labels=to_categorical(train_labels)
#  one_hot_test_labels=to_categorical(test_labels)
# =============================================================================

 


   #编译模型
from keras import models
from keras import layers
model=models.Sequential()
model.add(layers.Dense(64,activation='relu',input_shape=(10000,)))
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(46,activation='softmax'))
model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])

   


# ============================================================================
#留出验证集
x_val=x_train[:1000]
partial_x_train=x_train[1000:]

y_val=one_hot_train_labels[:1000]
partial_y_train=one_hot_train_labels[1000:]

#训练模型
#history =model.fit(partial_x_train,partial_y_train,epochs=20,batch_size=512,validation_data=(x_val,y_val))

# =============================================================================
# #绘制训练损失和验证损失
# import matplotlib.pyplot as plt
# loss =history.history['loss']
# val_loss=history.history['val_loss']
# 
# epochs=range(1,len(loss)+1)
# plt.plot(epochs,loss,'bo',label='Training loss')
# plt.plot(epochs,val_loss,'b',label='Validation loss')
# plt.title('Trainning and validation loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()
# =============================================================================

# =============================================================================
# #绘制训练精度和验证精度
# plt.clf()
# acc=history.history['acc']
# val_acc=history.history['val_acc']
# plt.plot(epochs,acc,'bo',label='Trainning acc')
# plt.plot(epochs,val_acc,'b',label='Validation acc')
# plt.title('Training and validation accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.show()
# =============================================================================

# =============================================================================
# #根据上图，可以看出第九轮后开始过拟合，从头开始训练一个网络，共九个轮次，注意运行此段代码时，注释前面部分代码
# model=models.Sequential()
# model.add(layers.Dense(64,activation='relu',input_shape=(10000,)))
# model.add(layers.Dense(64,activation='relu'))
# model.add(layers.Dense(46,activation='softmax'))
# 
# model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])
# model.fit(partial_x_train,partial_y_train,epochs=9,batch_size=512,validation_data=(x_val,y_val))
# =============================================================================

results=model.evaluate(x_test,one_hot_test_labels)

#在新数据上生成预测结果，每个元素的最大概率值即为类别,启动
predictions =model.predict(x_test)

print(np.argmax(predictions[0]))



