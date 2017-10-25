# -*- coding: utf-8 -*-
# !/usr/bin/python2
"""
unit test
"""
import os
import sys
import unittest
import numpy as np
import tensorflow as tf
from KWSDNN import KWSDNN, parseArgs
    
class KWSDNNTestCase(tf.test.TestCase):
    DIM_FEATURE = 23
    PRE_PADDING = 10
    POST_PADDING = 10
    LENGTH_FRAME = 40
    NUM_CLASS = 5
    BATCH_SIZE = 10
    TFRECORD_FILE = '/tmp/test.tfrecord'

    def setUp(self):
        print('0')
        args = parseArgs()
        self.KWSDNN = KWSDNN(args)
    #
    # def tearDown(self):
    #     self.deleteTFRecord()
        
    def test_shuffle_batch(self):
        labels_ans, feats_ans = self.genTFRecord()
        with self.test_session() as sess:
            feature_batch = self.KWSDNN.data_reader([self.TFRECORD_FILE])
            self.assertEqual(np.array(feats_ans[0]), feature_batch.eval())
            # sess.close()
            print('1-')
        
    def genTFRecord(self):
        key = 'test'
        labels = np.random.randint(self.NUM_CLASS, size=self.LENGTH_FRAME)
        feats = np.random.rand(self.LENGTH_FRAME, self.DIM_FEATURE)
        extend = np.ones(len(feats), dtype=np.int32)
        extend[0] = self.PRE_PADDING + 1
        extend[-1] = self.POST_PADDING + 1
        feats = np.repeat(feats, extend, axis=0)
        feats_flatten = feats.flatten()
         

        writer = tf.python_io.TFRecordWriter(self.TFRECORD_FILE)
        content = tf.train.Example(
            features=tf.train.Features(feature={
                'id': self.bytes_feature(key),
                'labels': self.bytes_feature(
                    labels.astype(np.int32).tostring()),
                'feats': self.bytes_feature(
                    feats_flatten.astype(np.float32).tostring())
            })
        )
        writer.write(content.SerializeToString())
        writer.close()
        
        return self.genAnswer(labels, feats)
        
    def genAnswer(self, labels, feats):
        labels_ans = np.zeros((len(labels), self.DIM_FEATURE))
        labels_ans[range(len(labels)), labels] = 1
        
        num_frame = self.PRE_PADDING + self.POST_PADDING + 1
        
        feats_ans = np.zeros(shape=(len(labels), num_frame * self.DIM_FEATURE))
        for index in range(0, len(feats) - num_frame + 1):
            feats_ans[index] = feats[index:index+num_frame].flatten()
        feats_ans.reshape((-1,2,num_frame * self.DIM_FEATURE))
        return labels_ans, feats_ans
            
    def deleteTFRecord(self):
        os.remove(self.TFRECORD_FILE)
        
    @staticmethod
    def bytes_feature(value):
        """
        return tensorflow BytesList
        Args:
            value: value to convert

        Returns: tensorflow BytesList

        """
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    @staticmethod
    def int64_feature(value):
        """
        return tensorflow int64
        Args:
            value: value to convert

        Returns: tensorflow int64

        """
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

# def test_silly(self):
#     with self.test_session() as sess:
#         init_op = tf.global_variables_initializer()
#         out = tf.contrib.layers.fully_connected(
#             inputs=tf.zeros([self.batch_size, 10]),
#             num_outputs=20,
#             activation_fn=None,
#             scope="hmmm")
#         sess.run(init_op)
#         print(out.eval())
        
if __name__ == '__main__':
    # tf.test.main()
    sys.argv[:] = sys.argv[1:]
    unittest.main()