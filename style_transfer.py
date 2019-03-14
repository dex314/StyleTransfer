'''
Simple Style Transfer
W.Schill 3.11.19

This is a real basic style transfer model using the VGG19 framework through
Tensorflow. Its started as a notebook with help from:
https://github.com/markojerkic/Tensorflow_style_transfer

It uses cv2 rather than scipy for image reading and writing.

INPUTS =====================================================
    base_path - default is current working directory
        Path to get base image.
    style_path - default is current working directory
        Path to get stylization image.
    output_path - default is current working directory
        Path to output the style trasnferred image.
    shape - default is (400,600,3)
        Desired or set shape of an image
    content_layer - default is block4_conv1 specific to VGG19
        Deisred lyaer for content loss
    style_layers - specified style loss layers
    iters - default is 1
        Not yet implemented
    max_opt_iters - default is 200
        Optimization steps for the TF optimizer.
    content_wt - default is 0.1
        Base image weight
    style_wt - default is 0.9
        Sytle image weight
    variance_wt - default is 1.0
        Variance weight
    save_each_pass - default is False
        Not yet implemented
        To save an image over each iteration of the style transfer
    base - base image
    style - style image
    model - Default is vgg19
        Neural net model
    sess - Tensorflow session
    step - Step counter for update function

METHODS ====================================================
    load_img(path, shape, content=True) - Preprocesses the image from path.
    deprocess_img(img) - De-Processes the image.
    initialize() - Initialize the loss and model.
    evaluate_and_create() - Evaluates and Creates new image. The Main call.
    update(loss) - Takes a loss and is implemented as a callback for the optimizer.
    calc_content_loss(sess, model, content_img) - Calculates content loss
    gram_matrix(x) - calculates the inner gram matrix of a feature map
    calc_style_loss(sess, model, style_img) - Calculates style loss.

TO DO ======================================================
    - Add channels_first / channels_last selection options
    - Add iters option or collapse with the max_opt_iters
    - Add save each pass or collapse with max_opt_iters

EXAMPLE ====================================================
    CONTENT_PATH = '../examples/bases/junk.jpg'
    STYLE_PATH = '../examples/styles/starrynight.jpg'
    CONTENT_LAYER = 'block4_conv1'
    STYLE_LAYERS = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
    obase = os.path.basename(CONTENT_PATH)
    import style_transfer as stx
    st = stx.StyleTransfer(base_path=CONTENT_PATH, style_path=STYLE_PATH,
                  output_path='../output/new_'+obase,
                  content_wt = 0.1,
                  style_wt = 0.1,
                  variance_wt = 0.1,
                  shape=(400,600,3))
    st.evaluate_and_create()

    import numpy as np
    import matplotlib.pyplot as plt
    from cv2 import imread

    stx.plot_comparison(CONTENT_PATH, STYLE_PATH, '../output/new_'+obase)
'''
import numpy as np
import os
import matplotlib.pyplot as plt
import datetime
import numpy as np
import tensorflow as tf
from cv2 import imread, imwrite, resize

import vgg19

## This is here for a particular issue with Tensorflow and Nvidia Incompatability Errors
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

class StyleTransfer(object):
    def __init__(self, base_path = None, style_path = None, output_path = None,
                       content_layer = 'block4_conv1',
                       style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1'],
                       shape = (400, 600, 3),
                       iters = 1, max_opt_iters = 200,
                       content_wt = 0.1, style_wt = 0.9, variance_wt = 1.0,
                       save_each_pass=False):

        curr_wd = os.getcwd()
        if base_path is None:
            self.base_path = curr_wd
        else:
            self.base_path = base_path
        if style_path is None:
            self.style_path = curr_wd
        else:
            self.style_path = style_path
        if output_path is None:
            self.output_path = curr_wd
        else:
            self.output_path = output_path

        self.shape = shape
        self.content_layer = content_layer
        self.style_layers = style_layers

        self.iters = iters
        self.max_opt_iters = max_opt_iters

        self.content_wt = content_wt
        self.style_wt = style_wt
        self.variance_wt = variance_wt
        self.save_each_pass = save_each_pass

        ## INITIALIZATIONS ====================================================
        self.base = self.load_img(self.base_path, self.shape)
        self.style = self.load_img(self.style_path, self.base.shape, content=False)

        self.model = vgg19.VGG()
        self.sess = sess
        self.step = 0

    def load_img(self, path, shape, content=True):
        ''' path = path of Image
            shape = shape of Image
            content = True if using loaded image shape
                      False if resizing
        '''
        img = imread(path)
        if content:
            # If the image is the content image, calculate the shape
            h, w, d = img.shape
            # width = int((w * shape / h))
            img_new = np.zeros((h, w, d))
            for j in range(d):
                img_new[:,:,j] = resize(img[:,:,j], (w, h))
            print('content {}'.format(img_new.shape))
        else:
            # The style image is set to be the same shape as the content image
            img_new = np.zeros((shape[1], shape[2], shape[3]))
            for j in range(shape[3]):
                img_new[:,:,j] = resize(img[:,:,j], (shape[2], shape[1]))
            print('style {}'.format(img.shape))
        img_new = img_new.astype('float32')
        # Subtract the mean values
        img_new -= np.array([123.68, 116.779, 103.939], dtype=np.float32)
        # Add a batch dimension
        img_new = np.expand_dims(img_new, axis=0)
        return img_new

    def deprocess_img(self, img):
        # Remove the fourth dimension
        img = img[0]
        # Add the mean values
        img += np.array([123.68, 116.779, 103.939], dtype=np.float32)
        return img

    def initialize(self):
        tf_base = tf.constant(self.base, dtype=tf.float32, name='base_img')
        tf_style = tf.constant(self.style, dtype=tf.float32, name='style_img')
        tf_new = tf.random_normal(tf_base.shape)

        model = self.model.create_graph(tf_base)

        closs = self.content_wt * self.calc_content_loss(self.sess, model, tf_base)
        sloss = self.style_wt * self.calc_content_loss(self.sess, model, tf_style)
        vloss = self.variance_wt * tf.image.total_variation(model['input'])
        loss = closs + sloss + vloss

        self.sess.run(tf.global_variables_initializer())
        self.sess.run(model['input'].assign(tf_new))

        optimizer = tf.contrib.opt.ScipyOptimizerInterface(loss,
                                                           method='L-BFGS-B',
                                                           options={'maxiter':self.max_opt_iters})
        return loss, optimizer, model

    def update(self,loss):
        '''Single callback for TF Optimizer and prints loss.'''
        if self.step%100 == 0:
            print('Step {}; loss{}'.format(self.step, loss))
        self.step += 1
        return loss

    def evaluate_and_create(self):
        '''Evaluates the loss and creates the new image.'''
        loss, optimizer, model = self.initialize()

        optimizer.minimize(self.sess, fetches=[loss], loss_callback=self.update)
        imwrite(self.output_path, self.deprocess_img(self.sess.run(model['input'])))
        print("New image written...")


    def calc_content_loss(self, sess, model, content_img):
        sess.run(tf.global_variables_initializer())
        # Set the input of the graph to the content image
        sess.run(model['input'].assign(content_img))
        # Get the feature maps
        p = sess.run(model[self.content_layer])
        x = model[self.content_layer]
        # Euclidean distance
        return tf.reduce_sum(tf.square(x - p)) * 0.5

    def gram_matrix(self, x):
        # Flatten the feature map
        x = tf.reshape(x, (-1, x.shape[3]))
        return tf.matmul(x, x, transpose_a=True)

    def calc_style_loss(self, sess, model, style_img):
        sess.run(tf.global_variables_initializer())
        # Set the input of the graph to the style image
        sess.run(model['input'].assign(style_img))
        loss = 0
        # We need to calculate the loss for each style layer
        for layer_name in self.style_layers:
            a = sess.run(model[layer_name])
            a = tf.convert_to_tensor(a)
            x = model[layer_name]
            size = a.shape[1].value * a.shape[2].value
            depth = a.shape[3].value
            gram_a = self.gram_matrix(a)
            gram_x = self.gram_matrix(x)
            loss += (1. / (4. * ((size ** 2) * (depth ** 2)))) * tf.reduce_sum(tf.square(gram_x - gram_a))
        return loss / len(STYLE_LAYERS)

## End class
## Utilities ==========================================================
def plot_comparison(base_path, style_path, output_path, figsize=(12,12)):
    result = imread(output_path)
    result = result[:, :, ::-1] ##switch to RGB
    orig = imread(base_path)
    orig = orig[:,:,::-1]
    sn = imread(style_path)
    sn = sn[:,:,::-1]

    fig, ax=plt.subplots(3,1,figsize=figsize)
    ax[0].imshow(orig); ax[0].set_title("Original");
    ax[1].imshow(sn); ax[1].set_title("Style");
    ax[2].imshow(result); ax[2].set_title("New");
    ax[0].get_xaxis().set_visible(False);
    ax[0].get_yaxis().set_visible(False);
    ax[1].get_xaxis().set_visible(False);
    ax[1].get_yaxis().set_visible(False);
    ax[2].get_xaxis().set_visible(False);
    ax[2].get_yaxis().set_visible(False);
    plt.tight_layout();

## END ================================================================
