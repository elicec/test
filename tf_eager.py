import tensorflow as tf
tf.enable_eager_execution()

print(tf.add(1,2))
print(tf.add([1,2],[3,4]))
print(tf.square(5))
print(tf.reduce_sum([1,2,3]))
print(tf.encode_base64("hello world"))
print(tf.square(2)+tf.square(3))

x = tf.matmul([[1]],[[2,3]])
print(x)

import numpy as np
ndarray = np.ones([3,3])
print(ndarray)
print("tensorflow convert np")
tensor = tf.multiply(ndarray,42)
print(tensor)

print(np.add(tensor,1))

print(tensor.numpy())

print("test gpu")
print("Is there a GPU :"),
print(tf.test.is_gpu_available())

import time
def time_matmul(x):
    start = time.time()
    for loop in range(10):
        tf.matmul(x,x)
    result = time.time() - start
    print("10 loops:{:0.2f}ms".format(1000*result))

print("On CPU:")
with tf.device("CPU:0"):
    x = tf.random_uniform([1000,1000])
    assert x.device.endswith("CPU:0")
    time_matmul(x)

x = tf.random_uniform([1000,1000])
print(x)

if tf.test.is_gpu_available():
  with tf.device("GPU:0"): # Or GPU:1 for the 2nd GPU, GPU:2 for the 3rd etc.
    x = tf.random_uniform([2000, 2000])
    assert x.device.endswith("GPU:0")
    time_matmul(x)
