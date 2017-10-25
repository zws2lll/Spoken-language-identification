# Copyright 2017 Mobvoi Inc. All Rights Reserved.
# Author: cfyeh@mobvoi.com (Ching-Feng Yeh)

#!/usr/bin/python2

"""Mobvoi neural networks definition and interface for Tensorflow."""

import tensorflow as tf
import numpy as np
# from pyKaldiIO import print
# from pyKaldiIO import print

class Nnet(object):
    """Definition of the interface for neural networks.
    """
    def __init__(self, filename=None):
        self.constant = {}
        self.operation = {}
        self.placeholder = {}
        self.variable = {}
        self.config = tf.ConfigProto()
        self.config.gpu_options.allow_growth = True  # Alway use minimum memory.
        self.session = tf.Session(config = self.config)
        self.coord = tf.train.Coordinator()
        self.threads = []
        self.start = False
        if filename is not None:
            self.Load(filename)
            self.session.run(tf.group(tf.global_variables_initializer(),
                                  tf.local_variables_initializer()))

    def __del__(self):
        self.coord.request_stop()
        self.coord.join(self.threads)
        self.session.close()

    def Load(self, filename):
        tf.train.import_meta_graph(filename + '.meta').restore(self.session, filename)
        with open(filename + '.interface', 'r') as f:
            for line in f.readlines():
                line = line.strip()
                token = line.split()
                interface = token[0]
                key = token[1]
                if interface == 'constant:':
                    vals = tf.get_collection(key)
                    if len(vals) == 0:
                        print('Missing key \"%s\" in the graph.' % key)
                    self.constant[key] = vals[0]
                elif interface == 'operation:':
                    vals = tf.get_collection(key)
                    if len(vals) == 0:
                        print('Missing key \"%s\" in the graph.' % key)
                    self.operation[key] = vals[0]
                elif interface == 'placeholder:':
                    vals = tf.get_collection(key)
                    if len(vals) == 0:
                        print('Missing key \"%s\" in the graph.' % key)
                    self.placeholder[key] = vals[0]
                elif interface == 'variable:':
                    vals = tf.get_collection(key)
                    if len(vals) == 0:
                        print('Missing key \"%s\" in the graph.' % key)
                    self.variable[key] = vals[0]
                else:
                    print('Unrecognized interface \"%s\"' % interface)
        return True

    def Save(self, filename):
        tf.train.Saver().save(self.session, filename)
        with open(filename + '.interface', 'w') as f:
            for key in self.constant:
                f.write('constant: %s\n' % key)
            for key in self.operation:
                f.write('operation: %s\n' % key)
            for key in self.placeholder:
                f.write('placeholder: %s\n' % key)
            for key in self.variable:
                f.write('variable: %s\n' % key)

    def GetSession(self):
        return self.session

    def GetConstant(self, name):
        if name not in self.constant:
            print('constant name \"%s\" is not found in the graph.' % name)
        return self.session.run(self.constant[name], feed_dict = {})

    def GetOperation(self, name):
        if name not in self.operation:
            print('operation name \"%s\" is not found in the graph.' % name)
        return self.operation[name]

    def HasPlaceHolder(self, name):
        if name in self.placeholder:
            return True
        else:
            return False
    
    def GetPlaceHolder(self, name):
        if name not in self.placeholder:
            print('placeholder name \"%s\" is not found in the graph.' % name)
        return self.placeholder[name]

    def Run(self, names, feed_dict=None, mode=0, learn_rate=2.048):
        '''
        
        :param names:
        :param feed_dict:
        :param mode: 0-train, 1-val, 2-test
        :param learn_rate: 2.048 = 0.008 * 256(default batch_size)
        :return:
        '''
        if type(names) is str:
            if names not in self.operation:
                 print('operation name \"%s\" is not found in the graph.' % names)
            else:
                return self.session.run(self.operation[names], feed_dict)
        if type(names) is not list:
            print('Input should be a single string / list of strings')
        for name in names:
            if name not in self.operation:
                print('operation name \"%s\" is not found in the graph.' % name)
        operations = [ self.operation[name] for name in names ]
        
        # Test model. Because queue can not produce test data(test data length
        # is not fixed), so use feed_dict
        if mode == 2:
            return self.session.run(operations, feed_dict=feed_dict)
        
        # Start all enqueue threads for both train queue and val queue when first
        # time to call Run
        if not self.start:
            print('starting all threads...')
            self.threads = tf.train.start_queue_runners(
                sess=self.session,
                coord=self.coord,
                start=True,
                collection='queue_runners')
            
            self.start = True
        
        loss = 0.0
        minibatch_processed = 0
        frame_predicted_correct = 0.0
        frame_processed = 0.0
        try:
            while not self.coord.should_stop():
                # test with first 300000 frames
                if frame_processed > 300000:
                    break
                
                # ensure to make operation[-1] = 'eval', operation[-2] = 'loss'
                results = self.session.run(operations,
                                           feed_dict={self.placeholder['mode']: mode,
                                                      self.placeholder['learn_rate']: learn_rate,
                                                      self.placeholder['frame_weights']: np.ones(256)})
                
                loss_in_minibatch = results[-2]
                frame_predicted_correct_in_minibatch = results[-1]

                frame_predicted_correct += frame_predicted_correct_in_minibatch
                frame_processed += 256
                loss += loss_in_minibatch
                
                if minibatch_processed % 1000 == 0:
                    frame_accuracy = frame_predicted_correct / frame_processed * 100
                    print('Frame accuracy = %.2f %% ( %d/ %d), Loss = %.6f'
                            % (frame_accuracy, frame_predicted_correct,
                               frame_processed, loss / minibatch_processed))
                minibatch_processed += 1
        except Exception as e:
            print(e)
            self.Stop()
        finally:
            frame_accuracy = frame_predicted_correct / frame_processed * 100
            print('Finally frame accuracy = %.2f %% ( %d/ %d), Loss = %.6f'
                  % (frame_accuracy, frame_predicted_correct,
                     frame_processed, loss / minibatch_processed))
            
            
    def Stop(self):
        # Terminate as usual. It is safe to call `coord.request_stop()` twice.
        # stop all threads for both train queue and val queue
        self.coord.request_stop()
        self.coord.join(self.threads)
        
    def AddToCollection(self, name = None, value = None):
        if name is not None and value is not None:
            tf.add_to_collection(key, value)
        else:
            # Add operations/placeholders to collection.
            for key, value in self.constant.iteritems():
                tf.add_to_collection(key, value)
            for key, value in self.operation.iteritems():
                tf.add_to_collection(key, value)
            for key, value in self.placeholder.iteritems():
                tf.add_to_collection(key, value)
            for key, value in self.variable.iteritems():
                tf.add_to_collection(key, value)

    def DumpCollection(self):
        print('Nnet collection(constants):')
        for key, _  in self.constant.iteritems():
            print('  %s' % key)
        print('Nnet collection(operations):')
        for key, _  in self.operation.iteritems():
            print('  %s' % key)
        print('Nnet collection(placeholders):')
        for key, _ in self.placeholder.iteritems():
            print('  %s' % key)

    def DumpVars(self): 
        print('Nnet contain %d global variables' % len(tf.global_variables()))
        trainable_names = [i.name for i in tf.trainable_variables()]
        for i in tf.global_variables(): 
            if i.name in trainable_names:
                print('  trainable variable: %s' % i.name)
            else:
                print('  non-trainable variable: %s' % i.name)
