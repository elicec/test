import tensorflow as tf
tf.enable_eager_execution()

class Model(object):
    def __init__(self):
        self.W = tf.Variable(2.0)
        self.b = tf.Variable(0.0)

    def __call__(self,x):
        return self.W*x +self.b

model = Model()

#define loss func
def loss(predicted_y, desired_y):
    return tf.reduce_mean(tf.square(predicted_y-desired_y))

#obtain training data
TRUE_W = 3.0
TRUE_B = 2.0
NUM_EXAMPLES = 2000
inputs = tf.random_normal(shape=[NUM_EXAMPLES])
noise = tf.random_normal(shape=[NUM_EXAMPLES])
outputs = inputs*TRUE_W+TRUE_B+noise

#print the training data
import matplotlib.pyplot as plt
# plt.scatter(inputs,outputs,c='b')
# plt.scatter(inputs,model(inputs),c='r')
# plt.show()
print("current loss."),
print(loss(model(inputs),outputs).numpy())

#train data
def train(model,inputs,outputs,lr):
    with tf.GradientTape() as t:
        current_loss = loss(model(inputs),outputs)
        dW,db = t.gradient(current_loss,[model.W,model.b])
        model.W.assign_sub(lr*dW)
        model.b.assign_sub(lr*db)

model = Model()
Ws,bs = [],[]
epochs = range(50)
for epoch in epochs:
    Ws.append(model.W.numpy())
    bs.append(model.b.numpy())
    current_loss = loss(model(inputs),outputs)

    train(model,inputs,outputs,lr=0.5)
    print("epoch %d: W=%1.2f b=%1.2f,loss=%2.5f" %
          (epoch,Ws[-1],bs[-1],current_loss))

plt.plot(epochs,Ws,'r',
         epochs,bs,'b')
plt.plot([TRUE_W]*len(epochs),'r--',
         [TRUE_B]*len(epochs),'b--')
plt.legend(['W','b','TRUE_W','TRUE_B'])
plt.show()
