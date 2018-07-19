import tensorflow as tf
import os
import numpy as np
"""
Dual Path Networks for super resolution
Uses modifed Dual Path Networks model to recover the image from low resolution to high resolution

"""
conv_conter=0
class SRDPNs(object):
    def __init__(self,
                 sess,
                 image_size=33,
                 label_size=21,
                 batch_size=128,
                 c_dim=3,
                 ):

        self.image_size = image_size
        self.label_size = label_size
        self.batch_size = batch_size
        self.c_dim =c_dim
        self.channel_axis=-1
        self.build_model()
        self.sess=sess


    def build_model(self):
        self.images = tf.placeholder(tf.float32, [None, self.image_size,self.image_size,self.c_dim],name='image')
        self.labels = tf.placeholder(tf.float32, [None, self.label_size, self.label_size, self.c_dim], name='labels')
        self.param={'initial_conv_filters': 64, 'depth': [3,3,3,3], 'filter_increment': [16, 32,32, 64],'cardinality': 1, 'width':3}

    def save(self,sess, checkpoint_dir, step):
        model_name = "SRDPN.model"
        model_dir = "%s_%s" % ("srdpn", self.label_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
             os.makedirs(checkpoint_dir)
        self.saver = tf.train.Saver()
        self.saver.save(sess,os.path.join(checkpoint_dir, model_name),global_step=1000)

    def load(self, sess, checkpoint_dir):
         print(" [*] Reading checkpoints...")
         model_dir = "%s_%s" % ("srdpn", self.label_size)
         checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

         ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
         if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver = tf.train.Saver()
            self.saver.restore(sess, os.path.join(checkpoint_dir,ckpt_name ))
            return True
         else:
            return False


    def _create_dpn(self):

        ''' Creates a DPN model with specified parameters
        Args:
            initial_conv_filters: number of features for the initial convolution
            filter_increment: number of filters incremented per block, defined as a list.
            cardinality: the cardinality of the ResNext path
            depth: the number of layers from corresponding block
        '''

        initial_conv_filters = self.param['initial_conv_filters']
        filter_increment= self.param['filter_increment']
        depth = self.param['depth']
        cardinality  = self.param['cardinality']
        width  = self.param['width']
        channel_axis = 1

        with tf.name_scope('deblurDPNs') as vs:
            img_input = self.images


            #  block 1 (initial conv block)
            input_1 = _initial_conv_block_inception(img_input, self.c_dim, 256)
            input_2 =_initial_conv_block_inception(input_1, 256, initial_conv_filters,1)
            x=input_2


            # block 2 (DPN block)
            base_filters = 128
            filter_inc = filter_increment[0]
            filters = int(cardinality * width)
            x = _dual_path_block(x, in_c=initial_conv_filters,
                                 a=filters,
                                 b=filters,
                                 c=base_filters,
                                 filter_increment=filter_inc,
                                 cardinality=cardinality,
                                 block_type='projection')
            in_chs = base_filters + 3 * filter_inc

            for i in range(2, depth[0] + 1):
                x = _dual_path_block(x, in_c=in_chs,
                                     a=filters,
                                     b=filters,
                                     c=base_filters,
                                     filter_increment=filter_inc,
                                     cardinality=cardinality,
                                     block_type='normal')
                in_chs += filter_inc


            # block 3 (DPN block)
            filter_inc = filter_increment[1]
            filters *= 2
            base_filters *= 2
            x = _dual_path_block(x, in_c=in_chs,
                                 a=filters,
                                 b=filters,
                                 c=base_filters,
                                 filter_increment=filter_inc,
                                 cardinality=cardinality,
                                 block_type='projection')
            in_chs = base_filters + 3 * filter_inc

            for i in range(2, depth[1] + 1):
                x = _dual_path_block(x, in_c=in_chs,
                                     a=filters,
                                     b=filters,
                                     c=base_filters,
                                     filter_increment=filter_inc,
                                     cardinality=cardinality,
                                     block_type='normal')
                in_chs += filter_inc

            # block 4 (DPN block)
            filter_inc = filter_increment[2]
            filters *= 2
            base_filters *= 2
            x = _dual_path_block(x, in_c=in_chs,
                                 a=filters,
                                 b=filters,
                                 c=base_filters,
                                 filter_increment=filter_inc,
                                 cardinality=cardinality,
                                 block_type='projection')
            in_chs = base_filters + 3 * filter_inc

            for i in range(2, depth[2] + 1):
                x = _dual_path_block(x, in_c=in_chs,
                                     a=filters,
                                     b=filters,
                                     c=base_filters,
                                     filter_increment=filter_inc,
                                     cardinality=cardinality,
                                     block_type='normal')
                in_chs += filter_inc


            # block 5 (DPN block)
            filter_inc = filter_increment[3]
            filters *= 2
            base_filters *= 2
            x = _dual_path_block(x, in_c=in_chs,
                                 a=filters,
                                 b=filters,
                                 c=base_filters,
                                 filter_increment=filter_inc,
                                 cardinality=cardinality,
                                 block_type='projection')
            in_chs = base_filters + 3 * filter_inc

            for i in range(2, depth[3] + 1):
                x = _dual_path_block(x, in_c=in_chs,
                                     a=filters,
                                     b=filters,
                                     c=base_filters,
                                     filter_increment=filter_inc,
                                     cardinality=cardinality,
                                     block_type='normal')
                in_chs += filter_inc

            result = []
            result.append(x[0])
            result.append(x[1])
            x=tf.concat(result,axis=-1)
            dim = x.get_shape()

            # layer Bottleneck used for dimensionality reduction
            with tf.variable_scope("Bottleneck") as scope:
                    W = [1, 1, dim[-1], 256]
                    S = [1, 1, 1, 1]
                    biases = tf.get_variable('biases', [256], initializer=tf.random_normal_initializer())
                    kernel = tf.get_variable('WBottleneck', shape=W, dtype=tf.float32,
                                             initializer=tf.contrib.layers.xavier_initializer_conv2d())
                    bottleneck = tf.nn.relu(tf.nn.conv2d(x, kernel, S, padding='VALID')+biases,name=scope.name)

            # layer Deconvolution used for restore the detail information
            reconstruction = parametric_relu(deconv(bottleneck, "deconv2", [3, 3, 3], [1, 1], padding='VALID'))

            return self.labels , reconstruction


def get_deconv2d_output_dims(input_dims, filter_dims, stride_dims, padding):
    # Returns the height and width of the output of a deconvolution layer.
        batch_size, input_h, input_w, num_channels_in = input_dims
        filter_h, filter_w, num_channels_out = filter_dims
        stride_h, stride_w = stride_dims
        out_h = 0
        out_w = 0

        if padding == 'SAME':
            out_h = input_h * stride_h
        elif padding == 'VALID':
            out_h = (input_h - 1) * stride_h + filter_h

        if padding == 'SAME':
            out_w = input_w * stride_w
        elif padding == 'VALID':
            out_w = (input_w - 1) * stride_w + filter_w

        return [batch_size, out_h, out_w, num_channels_out]


def deconv(input, name, filter_dims, stride_dims, padding='SAME',non_linear_fn=tf.nn.relu):
    input_dims = input.get_shape().as_list()
    assert len(input_dims) == 4  # batch_size, height, width, num_channels_in
    assert len(filter_dims) == 3  # height, width and num_channels out
    assert len(stride_dims) == 2  # stride height and width

    num_channels_in = input_dims[-1]
    filter_h, filter_w, num_channels_out = filter_dims
    stride_h, stride_w = stride_dims

    with tf.variable_scope(name) as scope:
            kernel = tf.get_variable('kernels',
                [filter_h, filter_w, num_channels_in, num_channels_out],
                initializer=tf.truncated_normal_initializer(mean=0.01, stddev=0.1))
            biases = tf.get_variable('biases', [num_channels_out],
                                     initializer=tf.zeros_initializer())
            output = tf.nn.conv2d(input, kernel, strides=[1, stride_h, stride_w, 1], padding=padding)
            output = tf.nn.bias_add(output, biases)
            return output


def bias_variable(shape):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial)


def parametric_relu(_x):
  alphas = tf.get_variable('alpha', _x.get_shape()[-1],
                       initializer=tf.constant_initializer(0.0),
                        dtype=tf.float32)
  pos = tf.nn.relu(_x)
  neg = alphas * (_x - abs(_x)) * 0.5
  return pos + neg

def conv2d(input, in_c, out_c, kernel_size, strides, with_bias=True,padding='SAME'):
    with tf.variable_scope("conv") as scope:
        W = [kernel_size, kernel_size, in_c, out_c]
        S = [1, strides, strides, 1]
        global conv_conter
        kernel = tf.get_variable(name="W"+str(conv_conter), shape=W, dtype=tf.float32, initializer=tf.truncated_normal_initializer())
        conv_conter+=1
        conv = tf.nn.conv2d(input, kernel, S, padding=padding)
        if with_bias:
            return conv + bias_variable([out_c])
        return conv

def _initial_conv_block_inception(input, in_c, out_c, kernel_size=5):

    #Adds an initial conv block, with batch norm and relu for the DPN

    x = tf.nn.relu(conv2d(input=input, in_c=in_c, out_c=out_c, kernel_size=kernel_size,strides=1))
    x = tf.contrib.layers.batch_norm(x, scale=True, updates_collections=None)
    return x


def _bn_relu_conv_block(input, in_c, out_c, kernel=3, stride=1):

    #Adds a Batchnorm-Relu-Conv block for DPN

    x = conv2d(input, in_c=in_c, out_c=out_c, kernel_size=kernel,
               strides=stride,padding='VALID')
    x = tf.contrib.layers.batch_norm(x, scale=True, updates_collections=None)
    x = tf.nn.relu(x)
    return x


def _grouped_convolution_block(input, in_c, grouped_channels, cardinality,kernel=3, stride=1):

    '''
        Adds a grouped convolution block
        cardinality is the variable used to decide number of groups
    '''

    init = input
    group_list = []

    if cardinality == 1:
        # with cardinality 1, it is a standard convolution
        x = conv2d(init, in_c=in_c, out_c=grouped_channels, kernel_size=kernel, strides=stride)
        x = tf.contrib.layers.batch_norm(x, scale=True, updates_collections=None)
        x = tf.nn.relu(x)
        return x

    for c in range(cardinality):
        x = input[:,:,:,c * grouped_channels:(c + 1) * grouped_channels]
        x = conv2d(x, in_c=grouped_channels, out_c=grouped_channels, kernel_size=3, strides=stride)
        group_list.append(x)

    group_merge = tf.concat(group_list, axis=-1)
    group_merge = tf.contrib.layers.batch_norm(group_merge, scale=True, updates_collections=None)
    group_merge = tf.nn.relu(group_merge)
    return group_merge


def _dual_path_block(input, in_c, a, b, c,filter_increment, cardinality, block_type='normal'):

    # Construct a Dual Path unit of one block:

    init = tf.concat(input,axis=-1)
    grouped_channels = int(b / cardinality)

    if block_type == 'projection':
        projection = True
    elif block_type == 'normal':
        projection = False
    else:
        raise ValueError('`block_type` must be one of ["projection", "normal"]. Given %s' % block_type)

    if projection:
        projection_path = _bn_relu_conv_block(init, in_c=in_c,
                                              out_c=c + 2 * filter_increment,
                                              kernel=1, stride=1)
        input_residual_path = projection_path[:, :, :, :c]
        input_dense_path = projection_path[:, :, :, c:]

    else:
        input_residual_path = input[0]
        input_dense_path = input[1]

    x = _bn_relu_conv_block(init, in_c=in_c, out_c=a,
                                kernel=1, stride=1)
    x = _grouped_convolution_block(x, in_c=a, grouped_channels=grouped_channels,
                                       cardinality=cardinality, kernel=3,stride=1)
    x = _bn_relu_conv_block(x, in_c=b, out_c=c + filter_increment,
                                kernel=1, stride=1)

    output_residual_path = x[:, :, :, :c]
    output_dense_path = x[:, :, :, c:]
    dense_list=[]
    dense_list.append(input_dense_path)
    dense_list.append(output_dense_path)
    dense_path = tf.concat(dense_list, axis=-1)
    residual_path = tf.add(input_residual_path, output_residual_path)

    return [residual_path, dense_path]





