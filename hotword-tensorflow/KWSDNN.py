#!/usr/bin/python2

import tensorflow as tf
import argparse
from nnet_dnn import BasicDNN
from nnet import Nnet
import sys
tf.logging.set_verbosity(tf.logging.INFO)
NUM_CLASSES = 5

def redirect_to_file():
    stdo = sys.stdout
    fhandle = open("out.txt", 'w')
    sys.stdout = fhandle
    
    return stdo
    
def redirect_to_stdo(stdo):
    sys.stdout = stdo
    
class KWSDNN(object):
    """
    FC DNN model to train key word spotting
    """
    def __init__(self, args):
        # TODO(wszhao@mobvoi.vom): adding args like 1)prepadding 2)postpadding 3)dim_feature
        # 3)
        self.args = args
        self.prepad = args.prepad
        self.postpad = args.postpad
        self.feature_dim = args.feature_dim
        self.batch_size = args.batch_size
        self.tr_tfrecord = args.tr_tfrecord
        self.cv_tfrecord = args.cv_tfrecord
        self.epoch = args.epoch
        self.save_dir = args.save_dir
        self.tr_model = None
        self.lr_rate = args.lr_rate * self.batch_size
    
    
    def check_args(self, args):
        # TODO(wszhao@mobvoi.com): check tfrecord file exist ans so on
        pass

    def shuffle_batch(self, filenames, num_enqueue_thread=5, num_epochs=1,
                             capacity=6000):
        with tf.name_scope('Reader'):
            filename_queue = tf.train.string_input_producer(
                filenames, num_epochs=num_epochs, shuffle=True, seed=777)
        
            reader = tf.TFRecordReader()
            _, serialized_example = reader.read(filename_queue)
            
            features = tf.parse_single_example(
                serialized_example,
                features={
                    'id': tf.FixedLenFeature([], tf.string),
                    'labels': tf.FixedLenFeature([], tf.string),
                    'feats': tf.FixedLenFeature([], tf.string)
                })
        
        with tf.name_scope('Decoder_reshape'):
            labels = tf.decode_raw(features['labels'], tf.int32)
            labels = tf.cast(labels, tf.float32)
            labels = tf.reshape(labels, [-1, 1])
            # labels = tf.one_hot(indices=labels, depth=NUM_CLASSES)
            
            size_window = self.prepad + self.postpad + 1
            dim_sample = size_window * self.feature_dim
            
            feats = tf.decode_raw(features['feats'], tf.float32)
            feats = tf.reshape(feats, [1, -1, self.feature_dim, 1])
            feats = tf.extract_image_patches(
                images=feats,
                ksizes=[1, size_window, self.feature_dim, 1],
                strides=[1, 1, 1, 1],
                rates=[1, 1, 1, 1],
                padding='VALID')
            feats = tf.reshape(feats, [-1, dim_sample])
            feats_labels = tf.concat([feats, labels], axis=1)
        
        with tf.name_scope('Enuqueue_Dequeue'):
            queue = tf.RandomShuffleQueue(capacity=capacity,
                                          min_after_dequeue=int(0.9 * capacity),
                                          shapes=[dim_sample+1],
                                          dtypes=feats.dtype,
                                          name='feats_random_shuffle_queue',
                                          seed=777)
    
            # Create an op to enqueue one item.
            enqueue = queue.enqueue_many(feats_labels)
    
            # Create a feats_queue runner that, when started, will launch 2(default) threads applying
            # that enqueue op.
            qr = tf.train.QueueRunner(queue, [enqueue] * num_enqueue_thread)
    
            # Register the feats_queue runner so it can be found and started by
            # `tf.train.start_queue_runners` later (the threads are not launched yet).
            tf.train.add_queue_runner(qr)

            # Create an op to dequeue a batch
            # return feats_queue.dequeue_many(batch_size), labels_queue.dequeue_many(batch_size)
            return queue.dequeue_many(self.batch_size)

    def get_input(self, type=0):
        filenames = []
        # 0-train, 1-validation, 2-test
        if type == 0:
            filenames = [self.tr_tfrecord]
        elif type == 1:
            filenames = [self.cv_tfrecord]
        elif type == 2:
            pass
        
        with tf.name_scope('Input'):
            feats_labels_batch = self.shuffle_batch(
                filenames=filenames)
            feats_batch, labels_batch = tf.split(
                feats_labels_batch,
                [(self.prepad + self.postpad + 1) * self.feature_dim, 1],
                axis=1
            )
            labels_batch = tf.cast(labels_batch, tf.int32)
            # return {'x': feats_batch, 'y': labels_batch}
            # return (np.random.rand((self.prepadding + self.postpadding + 1) * self.dim_feature), 2)
            return feats_batch, labels_batch
        
    def input_fn_train(self):
        return self.get_input(type=0)
        pass
    
    def input_fn_eval(self):
        return self.get_input(type=1)
        pass

    # TODO: how to add predict data
    
    
    def fit_DNNModel(self):
        with tf.name_scope('DNNModel'):
            # validation_monitor = tf.contrib.learn.monitors.ValidationMonitor(
            #     input_fn=lambda: self.input_fn_eval(),
            #     every_n_steps=20000,
            #     metrics=self.validation_metrics())
            
            feature_columns = [
                tf.contrib.layers.real_valued_column(
                    "", dimension=(self.prepad+self.postpad+1)*23)]
            
            classifier = tf.contrib.learn.DNNClassifier(
                feature_columns=feature_columns,
                hidden_units=[128, 128, 128],
                n_classes=NUM_CLASSES,
                activation_fn=tf.sigmoid,
                optimizer="Adam",
                model_dir='/tmp/mobvoi')
                # config=tf.contrib.learn.RunConfig(save_checkpoints_steps=20000))
            # print(self.input_fn_train())
            classifier.fit(input_fn=lambda: self.input_fn_train())
                # monitors=[validation_monitor])
            classifier.evaluate(input_fn=lambda: self.input_fn_eval())
            
    def eval_model(self):
        cv_model = BasicDNN(self.args, 'test')
        cv_model.Run(['eval'])
    
    def fit_old_model(self):
        print("training model...")
        
        for ep in range(self.epoch):
            if ep == 0:
                print('Initing 0th model...')
                self.tr_model = BasicDNN(self.args)
                
                print('Saving 0th model...')
                self.tr_model.Save(self.save_dir+'nnet.{}.model'.format(ep))
            else:
                self.tr_model = Nnet(filename=self.save_dir+'nnet.{}.model'.format(ep-1))
                if ep > 5:
                    self.lr_rate = self.lr_rate / 2
                    # self.tr_model.SetLrRate(self.lr_rate)
                
                print('Running {}th epoch...'.format(ep))
                self.tr_model.Run(['output', 'train', 'loss', 'eval'], mode=0, learn_rate=self.lr_rate)
                
                print('Saving {}th model...'.format(ep))
                self.tr_model.Save(self.save_dir + 'nnet.{}.model'.format(ep))
                
                print('Testing {}th model...'.format(ep))
                self.tr_model.Run(['loss', 'eval'], mode=1)
                
                self.tr_model.Stop()
                
            tf.reset_default_graph()
            
        

def parseArgs():
    parser = argparse.ArgumentParser()
    # Setup argument handling

    parser.add_argument(
        "--tr_tfrecord",
        help="tfrecord file of training set",
        action='store',
        default='/home/wszhao/data/KWSDNN/feats.tr.norm.tfrecord',
        dest='tr_tfrecord')
    parser.add_argument(
        "--cv_tfrecord",
        help="tfrecord file of validation set",
        action='store',
        default='/home/wszhao/data/KWSDNN/feats.cv.norm.tfrecord',
        dest='cv_tfrecord')
    parser.add_argument(
        "--te_tfrecord",
        help="tfrecord file of test set",
        action='store',
        default='/home/wszhao/data/KWSDNN/feats.te.norm.tfrecord',
        dest='te_tfrecord')
    parser.add_argument(
        "--nnet_in",
        help="model to load in",
        action='store',
        default='./',
        dest='tr_tfrecord')
    parser.add_argument(
        "--prepad",
        help="number of frame to pre padding",
        action='store',
        default=30,
        dest='prepad',
        type=int)
    parser.add_argument(
        "--postpad",
        help="number of frame to post padding",
        action='store',
        default=10,
        dest='postpad',
        type=int)
    parser.add_argument(
        "--feature_dim",
        help="dimention of feature",
        action='store',
        default=23,
        dest='feature_dim',
        type=int)
    parser.add_argument(
        "--batch_size",
        help="training batch size",
        action='store',
        default=256,
        dest='batch_size',
        type=int
    )
    parser.add_argument(
        "--lr_rate",
        help="learn rate",
        action='store',
        default=0.008,
        dest='lr_rate',
        type=float
    )
    parser.add_argument(
        "--epoch",
        help="training epoch",
        action='store',
        default=10,
        dest='epoch',
        type=int
    )
    parser.add_argument(
        "--save_dir",
        help="dir to save model",
        action='store',
        default='./model/',
        dest='save_dir'
    )
    parser.add_argument(
        "--exe_mode",
        help="train | test | predict",
        action='store',
        default='train',
        dest='exe_mode'
    )
    args = parser.parse_args()
    
    return args
    
def qf_model(args):
    KWSModel = KWSDNN(args)
    KWSModel.fit_old_model()
        
def new_model(args):
    KWSModel = KWSDNN(args)
    KWSModel.fit_DNNModel()
    
def main():
    # stdo = redirect_to_file()
    
    args = parseArgs()
    print(args)
    
    qf_model(args)
    
    # new_model(args)
    
    # redirect_to_stdo(stdo)

if __name__ == '__main__':
    main()
