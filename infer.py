# Use a trained Image Transform Net to generate
# a style transferred image with a specific style

import tensorflow as tf

from style_transfer_net import StyleTransferNet
from utils import get_images, save_images
from datetime import datetime
# import pdb

def stylize(contents_path, styles_path, output_dir, encoder_path, model_path, 
            resize_height=None, resize_width=None, suffix=None, n_iter=1):

    if isinstance(contents_path, str):
        contents_path = [contents_path]
    if isinstance(styles_path, str):
        styles_path = [styles_path]

    with tf.Graph().as_default(), tf.Session() as sess:
        # build the dataflow graph
        content = tf.placeholder(
            tf.float32, shape=(1, None, None, 3), name='content')
        style   = tf.placeholder(
            tf.float32, shape=(1, None, None, 3), name='style')

        stn = StyleTransferNet(encoder_path)

        output_image = stn.transform(content, style)

        sess.run(tf.global_variables_initializer())

        # restore the trained model and run the style transferring
        saver = tf.train.Saver()
        saver.restore(sess, model_path)

        # start_time = datetime.now()
        outputs = []
        for content_path in contents_path:

            content_img = get_images(content_path, 
                height=resize_height, width=resize_width)

            for style_path in styles_path:

                style_img   = get_images(style_path)

                for n_it in range(n_iter):
                    
                    if n_it == 0:
                        result = sess.run(output_image, 
                            feed_dict={content: content_img, style: style_img})
                        tmp_cont = result
                    else:
                        result = sess.run(output_image, 
                            feed_dict={content: tmp_cont, style: style_img})
                        tmp_cont = result
                outputs.append(result[0])
                # elapsed_time = datetime.now() - start_time
                # print(elapsed_time)
                # pdb.set_trace()
        
                            
    save_images(outputs, contents_path, styles_path, output_dir, suffix=suffix, n_iter=1)

    return outputs

