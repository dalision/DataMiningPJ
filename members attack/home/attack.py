from sklearn.metrics import classification_report, accuracy_score
import numpy as np
# import tensorflow as tf
import tensorflow.compat.v1 as tf
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier,AdaBoostClassifier,RandomForestClassifier
from sklearn.metrics import classification_report

def add_layer(inputs,in_size,out_size,activation_function=None):
    dropout=0.5
    inputs = tf.nn.dropout(inputs, 1-dropout)
    w = tf.Variable(tf.random_normal([in_size,out_size]))
    b = tf.Variable(tf.zeros([1,out_size])+0.1)
    f = tf.matmul(inputs,w)+b
    if activation_function is None:
        outputs = f
    else:
        outputs = activation_function(f)
    return outputs

def get_tensor_net(placeholder1,in_size,out_size,hidden_size):
    # Create the model
    l1 = add_layer(placeholder1,in_size,hidden_size,activation_function=tf.nn.relu)
    prediction = add_layer(l1,hidden_size,out_size,activation_function=tf.nn.relu)
    return prediction


# 全连接神经 0.69
def train(dataset,n_hidden=50,epochs=100, learning_rate=0.01):
    train_x, train_y, test_x, test_y = dataset
    n_in = train_x.shape[1]
    n_out = len(np.unique(train_y))
    # n_out=1

    # if batch_size > len(train_y):
    #     batch_size = len(train_y)

    print ('Building model with {} training data, {} classes...'.format(len(train_x), n_out))
    xs = tf.placeholder(tf.float32,[None,n_in])
    ys = tf.placeholder(tf.float32,[None,n_out])
    prediction=tf.nn.softmax(get_tensor_net(xs,n_in,n_out,10))
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),reduction_indices=[1]))
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
    correct_predictions = tf.equal(tf.argmax(prediction, 1), tf.argmax(ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32)) 
    # 初始化
    init = tf.global_variables_initializer()
    sess = tf.Session()

    graph = tf.get_default_graph()
    with graph.as_default():
    # sess.run()
        sess.run(init)
    # 开始迭代训练
    for i in range(epochs):
        sess.run(train_step,feed_dict={xs:train_x,ys:train_y})
        if(i%5==0):
            print("train epoch "+str(i)+":")
            print(sess.run([cross_entropy,accuracy],feed_dict={xs:train_x,ys:train_y}))
    if test_x is not None:
        print ('Testing...')
        # if batch_size > len(test_y):
        #     batch_size = len(test_y)
        # print(sess.run([cross_entropy,accuracy],feed_dict={xs:test_x,ys:test_y}))
        print("accuracy:",sess.run(accuracy,feed_dict={xs:test_x,ys:test_y}))
        accuracy=sess.run(accuracy,feed_dict={xs:test_x,ys:test_y})
        return accuracy