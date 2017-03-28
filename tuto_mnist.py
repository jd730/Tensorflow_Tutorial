##THe MNIST Data
import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
###Implementing the Regression
import tensorflow as tf

x = tf.placeholder(tf.float32,[None,784]) #784-dim vector
W = tf.Variable(tf.zeros([784,10])) #Weight
b = tf.Variable(tf.zeros([10])) #Bias
y = tf.nn.softmax(tf.matmul(x,W) +b)

###Training
y_ = tf.placeholder(tf.float32,[None,10]) #true distribution (the one-hot vector we will input)
cross_entropy = -tf.reduce_sum(y_*tf.log(y))

#Optimization Alogirthms : what you want to use such as https://www.tensorflow.org/versions/master/api_docs/python/train.html#optimizers
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy) #0.01 is a learning rate

#set up to train
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init) #initializes the variables.


#train.
for i in range(1000) : #Number of steps.
	#if we use all data for every step. it is too expensive, so we choose 100 random data points
	batch_xs, batch_ys = mnist.train.next_batch(100) 
	sess.run(train_step,feed_dict={x:batch_xs, y_:batch_ys})

## Evaluating Our Model

#compare our model's label(y) to correct label(y_), correct_prediction is list of Boolean.
#argmax(vector,rank) Returns the index with the largest value across rank of a tensor.
correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float")) #reduce_mean -> mean from list
print (sess.run(accuracy,feed_dict={x:mnist.test.images,y_ : mnist.test.labels}))
