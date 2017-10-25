# Copyright 2017 Mobvoi Inc. All Rights Reserved.
# Author: cfyeh@mobvoi.com (Ching-Feng Yeh)

# !/usr/bin/python2

import math
import tensorflow as tf
from nnet import Nnet


class BasicDNN(Nnet):
    """Basic deep neural network (DNN) model."""
    
    def __init__(self, args, filename=None):
        # TODO: is super necessary?
        if filename == None:
            super(BasicDNN, self).__init__()
        else:
            super(BasicDNN, self).__init__(filename)
        self.component = []
        self.prepad = args.prepad
        self.postpad = args.postpad
        self.feature_dim = args.feature_dim
        self.batch_size = args.batch_size
        self.tr_tfrecord = args.tr_tfrecord
        self.cv_tfrecord = args.cv_tfrecord
        
        self.layer_dims = [943, 128, 128, 128, 5]
        self.seed = 777
        self.queue = None
        
        if filename != None:
            self.Load(filename)
        else:
            self.BuildGraph(self.layer_dims, self.seed)
            
        # self.session.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))
    
            
    def shuffle_batch(self, filenames, num_enqueue_thread=5, num_epochs=1, capacity=6000):
        sip = tf.train.string_input_producer(
            filenames, num_epochs=num_epochs, shuffle=False)
    
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(sip)
    
        features = tf.parse_single_example(
            serialized_example,
            features={
                'id': tf.FixedLenFeature([], tf.string),
                'labels': tf.FixedLenFeature([], tf.string),
                'feats': tf.FixedLenFeature([], tf.string)
            })
    
        labels = tf.decode_raw(features['labels'], tf.int32)
        labels = tf.cast(labels, tf.float32)
        labels = tf.reshape(labels, [-1, 1])
        # labels = tf.one_hot(indices=labels, depth=NUM_CLASSES)
    
        # size_window = self.prepad + self.postpad + 1
        size_window = self.prepad + self.postpad + 1
        dim_sample = size_window * self.feature_dim
    
        feats = tf.decode_raw(features['feats'], tf.float32)
        # reshape feats to call extract_image_patches to splice(30, 10)
        feats = tf.reshape(feats, [1, -1, self.feature_dim, 1])
        feats = tf.extract_image_patches(
            images=feats,
            ksizes=[1, size_window, self.feature_dim, 1],
            strides=[1, 1, 1, 1],
            rates=[1, 1, 1, 1],
            padding='VALID')
        
        feats = tf.reshape(feats, [-1, dim_sample])
        feats_labels = tf.concat([feats, labels], axis=1)
        
        queue = tf.RandomShuffleQueue(capacity=capacity,
                                           min_after_dequeue=int(
                                               0.9 * capacity),
                                           shapes=[dim_sample + 1],
                                           dtypes=tf.float32,
                                           name='queue_{}'.format(filenames[0]),
                                           seed=777)
    
        # Create an op to enqueue one item.
        enqueue = queue.enqueue_many(feats_labels)
    
        # Create a feats_queue runner that, when started, will launch 2(default) threads applying
        # that enqueue op.
        qr = tf.train.QueueRunner(
            queue, [enqueue] * num_enqueue_thread)
    
        # Register the feats_queue runner so it can be found and started by
        # `tf.train.start_queue_runners` later (the threads are not launched yet).
        tf.train.add_queue_runner(qr, collection='queue_runners')
    
        # Create an op to dequeue a batch
        # return feats_queue.dequeue_many(batch_size), labels_queue.dequeue_many(batch_size)
        # return self.queue.dequeue_many(self.batch_size)
        return queue
        
        
    def get_input(self):
        tr_queue = self.shuffle_batch([self.tr_tfrecord])
        cv_queue = self.shuffle_batch([self.cv_tfrecord])
        
        # test data can not use queue. build two queue at the same time, 'mode'
        # is used to control which queue to dequeue within a step
        q = tf.QueueBase.from_list(self.placeholder['mode'], [tr_queue, cv_queue])
        
        feats_labels_batch = q.dequeue_many(self.batch_size)
        feats_batch, labels_batch = tf.split(
            feats_labels_batch,
            [(self.prepad + self.postpad + 1) * self.feature_dim, 1],
            axis=1
        )
        labels_batch = tf.reshape(labels_batch, [-1])
        labels_batch = tf.cast(labels_batch, tf.int32)

        return feats_batch, labels_batch
    
    
    def BuildGraph(self, layer_dims, seed=777):
        """Build a new graph for the model.

        Args:
            layer_dims: The dimensions of each layers in the form of list. For
                        example, [483, 128, 64, 64, 5] gives a model containing
                        the following layers:
                        input layer: 483
                        hidden layer[1]: 483 -> 128
                        hidden layer[2]: 128 -> 64
                        hidden layer[3]: 64 -> 64
                        output layer: 64 -> 5
            seed: The seed for randomization.

        Return:
            None
        """
        # Add constants.
        self.constant['input_dim'] = tf.constant(layer_dims[0], tf.int32,
                                                 shape=(), name='input_dim')
        self.constant['output_dim'] = tf.constant(layer_dims[-1], tf.int32,
                                                  shape=(), name='output_dim')
        self.constant['input_rank'] = tf.constant(2, tf.int32,
                                                  shape=(), name='input_rank')
        
        # Add placeholders.
        # self.placeholder['input'] = tf.placeholder(tf.float32,
        #                                            shape=(None, layer_dims[0]),
        #                                            name = 'input')
        
        
        # self.placeholder['label'] = tf.placeholder(tf.int32, shape = (None),
        #                                            name = 'label')
        
        # TODO:label to one-hot
        self.placeholder['mode'] = tf.placeholder(
            dtype=tf.int32, shape=(), name='mode')
        
        feats_batch, labels_batch = self.get_input()

        # use queue data when it is not fed. If it is fed, use the data in feed_dict
        self.placeholder['input'] = tf.placeholder_with_default(
            feats_batch, shape=(None, layer_dims[0]), name='input')
        
        self.placeholder['label'] = tf.placeholder_with_default(
            labels_batch, shape=(None), name='label')

        self.placeholder['learn_rate'] = tf.placeholder(
            tf.float32, shape=(), name='learn_rate')
        
        self.placeholder['frame_weights'] = tf.placeholder(
            tf.float32, shape=(None), name='frame_weights')
         
        
        # Build inference graph.
        self.component.append(self.placeholder['input'])
        for i in xrange(len(layer_dims) - 1):
            with tf.name_scope('layer%d' % i):
                layer_input_dim = layer_dims[i]
                layer_output_dim = layer_dims[i + 1]
                stddev = 1.0 / math.sqrt(float(layer_input_dim))
                weights = tf.Variable(
                    tf.truncated_normal([layer_input_dim, layer_output_dim],
                                        stddev=stddev, seed=(seed + i)),
                    name='W')
                biases = tf.Variable(
                    tf.zeros([layer_output_dim]), name='b')
                x = self.component[-1]
                y = tf.nn.xw_plus_b(x, weights, biases)
                # Add sigmoid() activation for layers except for the last one.
                if i + 2 < len(layer_dims):
                    y = tf.sigmoid(y)
                self.component.append(y)
        self.operation['logit'] = self.component[-1]
        
        # Define loss.
        cross_entropy_with_logits = \
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self.placeholder['label'],
                logits=self.operation['logit'])
        frame_weights = self.placeholder['frame_weights']
        weighted_cross_entropy_with_logits = \
            tf.multiply(frame_weights, cross_entropy_with_logits)
        frame_weights_sum = tf.reduce_sum(frame_weights)
        cross_entropy_sum = tf.reduce_sum(weighted_cross_entropy_with_logits)
        self.operation['loss'] = tf.divide(cross_entropy_sum, frame_weights_sum)
        
        # Define training.
        criterion = tf.reduce_mean(weighted_cross_entropy_with_logits)
        learn_rate = self.placeholder['learn_rate']
        self.operation['train'] = \
            tf.train.GradientDescentOptimizer(learn_rate).minimize(criterion)
            # tf.train.AdamOptimizer(1e-4).minimize(cross_entropy_sum)
        
        # Define evaluation.
        correct = \
            tf.nn.in_top_k(self.operation['logit'], self.placeholder['label'],
                           1)
        binary_frame_weight = \
            tf.not_equal(frame_weights, tf.zeros_like(frame_weights))
        weighted_correct = \
            tf.multiply(tf.cast(correct, tf.int8),
                        tf.cast(binary_frame_weight, tf.int8))
        self.operation['eval'] = \
            tf.reduce_sum(tf.cast(weighted_correct, tf.int32))
        
        # Define output after softmax
        self.operation['output'] = tf.nn.softmax(self.operation['logit'])
        
        # Initialize all variables.
        self.session.run(tf.global_variables_initializer())
        
        # Add operations/placeholders to collection.
        self.AddToCollection()
        # self.DumpVars()


class BatchNormDNN(Nnet):
    """Deep neural network (DNN) model with batch normalization."""
    
    def __init__(self, layer_dims=None, seed=777):
        super(BatchNormDNN, self).__init__()
        self.component = []
        if layer_dims is not None:
            self.BuildGraph(layer_dims, seed)
    
    def BuildGraph(self, layer_dims, seed=777):
        """Build a new graph for the model.

        Args:
            layer_dims: The dimensions of each layers in the form of list. For
                        example, [483, 128, 64, 64, 5] gives a model containing
                        the following layers:
                        input layer: 483
                        hidden layer[1]: 483 -> 128
                        hidden layer[2]: 128 -> 64
                        hidden layer[3]: 64 -> 64
                        output layer: 64 -> 5
            seed: The seed for randomization.

        Return:
            None
        """
        # Add constants.
        self.constant['input_dim'] = tf.constant(layer_dims[0], tf.int32,
                                                 shape=(), name='input_dim')
        self.constant['output_dim'] = tf.constant(layer_dims[-1], tf.int32,
                                                  shape=(), name='output_dim')
        self.constant['input_rank'] = tf.constant(2, tf.int32,
                                                  shape=(), name='input_rank')
        
        # Add placeholders.
        self.placeholder['input'] = tf.placeholder(tf.float32,
                                                   shape=(None, layer_dims[0]),
                                                   name='input')
        self.placeholder['label'] = tf.placeholder(tf.int32, shape=(None),
                                                   name='label')
        self.placeholder['learn_rate'] = tf.placeholder(tf.float32, shape=(),
                                                        name='learn_rate')
        self.placeholder['frame_weights'] = tf.placeholder(tf.float32,
                                                           shape=(None),
                                                           name='frame_weights')
        self.placeholder['is_training'] = tf.placeholder(tf.bool, shape=(),
                                                         name='is_training')
        
        # Build inference graph.
        self.component.append(self.placeholder['input'])
        for i in xrange(len(layer_dims) - 1):
            with tf.name_scope('layer%d' % i):
                layer_input_dim = layer_dims[i]
                layer_output_dim = layer_dims[i + 1]
                stddev = 1.0 / math.sqrt(float(layer_input_dim))
                weights = tf.Variable(
                    tf.truncated_normal([layer_input_dim, layer_output_dim],
                                        stddev=stddev, seed=(seed + i)),
                    name='W')
                biases = tf.Variable(
                    tf.zeros([layer_output_dim]), name='b')
                x = self.component[-1]
                y = tf.nn.xw_plus_b(x, weights, biases)
                # Add sigmoid() activation for layers except for the last one.
                if i + 2 < len(layer_dims):
                    # TODO(cfyeh): figure out which one is better, before or after activation?
                    y = tf.contrib.layers.batch_norm(y, center=True, scale=True,
                                                     is_training=
                                                     self.placeholder[
                                                         'is_training'],
                                                     scope='layer%d' % i)
                    y = tf.sigmoid(y)
                self.component.append(y)
        self.operation['logit'] = self.component[-1]
        
        # Define loss.
        cross_entropy_with_logits = \
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self.placeholder['label'],
                logits=self.operation['logit'])
        frame_weights = self.placeholder['frame_weights']
        weighted_cross_entropy_with_logits = \
            tf.multiply(frame_weights, cross_entropy_with_logits)
        frame_weights_sum = tf.reduce_sum(frame_weights)
        cross_entropy_sum = tf.reduce_sum(weighted_cross_entropy_with_logits)
        self.operation['loss'] = tf.divide(cross_entropy_sum, frame_weights_sum)
        
        # Define training.
        criterion = tf.reduce_mean(weighted_cross_entropy_with_logits)
        learn_rate = self.placeholder['learn_rate']
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.operation['train'] = \
                tf.train.GradientDescentOptimizer(learn_rate).minimize(
                    criterion)
        
        # Define evaluation.
        correct = \
            tf.nn.in_top_k(self.operation['logit'], self.placeholder['label'],
                           1)
        binary_frame_weight = \
            tf.not_equal(frame_weights, tf.zeros_like(frame_weights))
        weighted_correct = \
            tf.multiply(tf.cast(correct, tf.int8),
                        tf.cast(binary_frame_weight, tf.int8))
        self.operation['eval'] = \
            tf.reduce_sum(tf.cast(weighted_correct, tf.int32))
        
        # Define output after softmax
        self.operation['output'] = tf.nn.softmax(self.operation['logit'])
        
        # Initialize all variables.
        self.session.run(tf.global_variables_initializer())
        
        # Add operations/placeholders to collection.
        self.AddToCollection()
        self.DumpVars()
