from __future__ import absolute_import, division, print_function
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
from model import *
from batch import *
from utils.plot import *

import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix

def getAccuracy(res1, res2, test_y):
    dist = np.sqrt(np.sum((res1 - res2) ** 2, axis=1) + epsilon)
    pred = (dist > 0.001).astype(float)
    return  pred, 100 * np.sum((pred == test_y).astype(float), axis=0) / len(test_y)

#Hyperparameter
n_epoch = 50000
learning_rate = 0.001

with tf.Session() as sess:
    siamese = Siamese()
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(siamese.loss)
    saver = tf.train.Saver()
    tf.global_variables_initializer().run()

    for step in range(n_epoch + 1):
        batch_input_1, batch_y1 = getRandomData()
        batch_input_2, batch_y2 = getRandomData()
        batch_y = (batch_y1 != batch_y2).astype(float)

        _, loss_v = sess.run([train_step, siamese.loss], feed_dict={
                            siamese.input_1: batch_input_1,
                            siamese.input_2: batch_input_2,
                            siamese.y_: batch_y})
    
        if np.isnan(loss_v):
            print('Model diverged with loss = NaN')
            quit()

        if step % 1000 == 0 and step > 0:
            saver.save(sess, 'saved/tf/snn/model.ckpt')
            tf.train.write_graph(tf.get_default_graph(), 'SNN',
                        'saved_model.pb', as_text=False)

            test_input_1, test_input_2, test_y = getTestData()
            print ('Step %d: loss %.3f' % (step, loss_v))
            # res1 = siamese.o1.eval(feed_dict={siamese.input_1: test_input_1})
            # res2 = siamese.o2.eval(feed_dict={siamese.input_2: test_input_2})
            pred, accu = getAccuracy(siamese.o1.eval(feed_dict={siamese.input_1: test_input_1}), 
                siamese.o2.eval(feed_dict={siamese.input_2: test_input_2}),
                test_y)
            print('Test accuracy: {}%'.format(accu))

            if step == n_epoch:
                conf_mat = tf.confusion_matrix(labels=test_y, predictions=pred, num_classes=2).eval()
                printConfusionMatrix(conf_mat)


    

