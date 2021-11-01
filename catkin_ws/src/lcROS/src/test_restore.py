import sys, os
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import time 
import numpy as np
import tensorflow as tf
from tf_restore_graph import restore_graph
from tensorflow.python.summary.event_accumulator import EventAccumulator
sess = tf.InteractiveSession()

# saver = tf.train.import_meta_graph('/home/user/Desktop/LoopClosure/CGF/checkpoints/embed_d10_p11_r17_a12_real_norelu/embed_model_9500000.ckpt.meta')  # load meta
# saver.restore(sess, '/home/user/Desktop/LoopClosure/CGF/checkpoints/embed_d10_p11_r17_a12_real_norelu/embed_model_9500000.ckpt')  # load ckpt
# writer = tf.summary.FileWriter(logdir='/home/user/Desktop/LoopClosure/SRC/catkin_ws/src/lcROS/tfeventdir', graph=tf.get_default_graph())  # write to event
# writer.flush()

events = EventAccumulator('/home/user/Desktop/LoopClosure/SRC/catkin_ws/src/lcROS/tfeventdir/events.out.tfevents.1588661039.cu-cs-vm')
events.Reload()
(x,y), saver = restore_graph(
    events.Graph(),
    tf.train.get_checkpoint_state('/home/user/Desktop/LoopClosure/CGF/checkpoints/embed_d10_p11_r17_a12_real_norelu/embed_model_9500000.ckpt').model_checkpoint_path,
    return_elements=['x', 'y']
)

npzfile = np.load('/home/user/Desktop/LoopClosure/SRC/catkin_ws/src/lcROS/output.npz')
data = npzfile['data']

print(sess.run(y, feed_dict={x:data}))
