from VGG import *
from DPNs_model import *
from utils import *
import os
import time
import utils as ul
try:
    xrange
except:
    xrange = range
CONTENT_LAYERS = ['relu1_2','relu2_2','relu3_4', 'relu4_2']


def perceptual_loss(truth_features,pred_features):

    # Caculate the perceptual loss at the VGG19

    _,height,width,channel = map(lambda i:i.value,pred_features.get_shape())
    content_size = height * width * channel
    return tf.nn.l2_loss(truth_features - pred_features) / content_size

def VGG_loss_function(truth_image,pred_image):

    # Read the feature maps form VGG16 and obtain the preceptual loss and gram loss respectively

    truth_features = vgg19([truth_image])
    pred_features = vgg19([pred_image])
    loss = 0.0
    for layer in CONTENT_LAYERS:
        loss += CONTENT_WEIGHT * perceptual_loss(truth_features[layer],pred_features[layer])
    return loss

def MSE_loss(truth_image,pred_image):
    return tf.reduce_mean(tf.square(truth_image-pred_image))

def get_variables_with_name(name, train_only=True, printable=False):

    print("  [*] Geting variables with %s" % name)
    if train_only:
        t_vars = tf.trainable_variables()
    else:
        try:
            t_vars = tf.global_variables()
        except:
            t_vars = tf.all_variables()

    d_vars = [var for var in t_vars if name in var.name]
    if printable:
        for idx, v in enumerate(d_vars):
            print("  got {:3}: {:15}   {}".format(idx, v.name, str(v.get_shape())))
    return d_vars


def train(config):
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    with tf.Session(config = tf_config) as sess:

        # Load Data
        if config.is_train:
            train_data, train_label= input_setup(config)
        else:
            nx, ny, train_data, train_label = input_setup(config)
            print(" [*] Loading data has been sucessful...")

        # Define the model
        DPNs = SRDPNs(sess,image_size=config.image_size,label_size=config.label_size,batch_size=config.batch_size)
        truth_image, pred_image = DPNs._create_dpn()
        print(" [*] Sucessfully initialize SRDPNs networks...")

        # Define the loss
        vgg_loss=VGG_loss_function(truth_image, pred_image)
        mse_loss = MSE_loss(truth_image, pred_image)
        total_loss = mse_loss+0.00005*vgg_loss
        print(" [*] Sucessfully initialize loss function...")
        Learning_rate = tf.train.exponential_decay(config.learning_rate,global_step=1000,decay_steps=config.decay_step,decay_rate=config.decay_rate)
        optim = tf.train.AdamOptimizer(learning_rate = Learning_rate).minimize(total_loss)

        #Initialize the variables
        init = tf.initialize_all_variables()
        sess.run(init)
        counter = 0

        #Load the existing model
        if  DPNs.load(sess=sess,checkpoint_dir=config.checkpoint_dir):
                print(" [*] Load model sucessed...")
        else:
                print(" [!] Load model  failed...")

        if config.is_train:
            print("training...")
            start_time = time.time()
            for ep in xrange(config.epoch):

                 # Run by batch images
                 batch_idxs = len(train_data) // config.batch_size
                 for idx in xrange(0, batch_idxs):
                     batch_images = train_data[idx * config.batch_size: (idx + 1) * config.batch_size]
                     batch_labels = train_label[idx * config.batch_size: (idx + 1) * config.batch_size]
                     counter += 1
                     _, err1,err2 = sess.run([optim, mse_loss,total_loss],
                                               feed_dict={DPNs.images: batch_images, DPNs.labels: batch_labels})

                     if counter % 10 == 0:
                            print("Epoch: [%2d], step: [%2d], time: [%4.4f], loss mse: [%.8f],total loss: [%.8f]" \
                                  % ((ep + 1), counter, time.time() - start_time, err1,err2))

                     if counter % 100 == 0:
                            DPNs.save(sess=sess,checkpoint_dir=config.checkpoint_dir, step=counter)
            print("finished training")

        else:

             print("test...")
             truth,pred = sess.run([truth_image,pred_image],feed_dict={DPNs.images: train_data, DPNs.labels: train_label})
             result = merge(pred, [nx, ny])

             if(ul.w_l!=ul.h_l):
                result = crop(result.squeeze())

             image_path = os.path.join(os.getcwd(), config.sample_dir)
             image_path = os.path.join(image_path, "test_image.png")
             imsave(result, image_path)


