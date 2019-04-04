import tensorflow as tf

tf.reset_default_graph()

# # Create some variables.
# v1 = tf.get_variable("up_latents", shape=[2])
#
# # Add ops to save and restore all the variables.
# saver = tf.train.Saver()
#
# # Later, launch the model, use the saver to restore variables from disk, and
# # do some work with the model.
#
# with tf.Session() as sess:
#   # Restore variables from disk.
#   saver.restore(sess, "/home/eric/Desktop/LyreBird/Data_Dump/Saved_Models/LyreBird_TN/model.ckpt-49")
#   print("Model restored.")
#   # Check the values of the variables
#   print("v1 : %s" % v1.eval())

#
# import tensorflow as tf
# from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
#
#
# latest_ckp = tf.train.latest_checkpoint('/home/eric/Desktop/LyreBird/Data_Dump/Saved_Models/LyreBird_TN/model.ckpt-49')
# print_tensors_in_checkpoint_file(latest_ckp, all_tensors=True, tensor_name='')


from tensorflow.python import pywrap_tensorflow
import os


reader = pywrap_tensorflow.NewCheckpointReader("/home/eric/Desktop/LyreBird/Data_Dump/Saved_Models/LyreBird_TN/model.ckpt-24999")
var_to_shape_map = reader.get_variable_to_shape_map()


please_god = False
count = 0
for key in var_to_shape_map:

    if "Adam" in key:
        please_god = True
        count += 1


    print("tensor_name: ", key)
    print(reader.get_tensor(key))

if please_god:
    print("YESSSSS :)")
    print(count)

else:
    print("Dam :(")
