import tensorflow as tf
import stackeddag.tf as sd

def mydataflow():
  a = tf.constant(1,name="a")
  b = tf.constant(2,name="b")
  c = tf.add(a,b,name="c")
  return tf.get_default_graph()

print(sd.fromGraph(mydataflow()),end="")
