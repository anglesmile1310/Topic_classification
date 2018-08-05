#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import datetime
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn
import time
# Parameters
# ================================================================================================================================

# Data loading params         =====================================================
#Flag là cách truyền thông số vào chương trình để chạy model với nhiều cấu hình khác nhau.

# tỉ lệ phần trăm dữ liệu validation dùng để validate,tối ưu parameter
tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
#dữ liệu topic 1 được lưu vào data_file1
tf.flags.DEFINE_string("data_file1", "./TrainData/pre1.txt", "Data source for the data topic 1.")

#dữ liệu tieu cực được lưu vào negative_data_file
tf.flags.DEFINE_string("data_file2", "./TrainData/pre2.txt", "Data source for the data topic 2.")
#dữ liệu tieu cực được lưu vào negative_data_file
tf.flags.DEFINE_string("data_file3", "./TrainData/pre3.txt", "Data source for the data topic 3.")
#dữ liệu tieu cực được lưu vào negative_data_file
tf.flags.DEFINE_string("data_file4", "./TrainData/pre4.txt", "Data source for the data topic 4.")
#dữ liệu tieu cực được lưu vào negative_data_file
tf.flags.DEFINE_string("data_file5", "./TrainData/pre5.txt", "Data source for the data topic 5.")
#dữ liệu tieu cực được lưu vào negative_data_file
tf.flags.DEFINE_string("data_file6", "./TrainData/pre6.txt", "Data source for the data topic 6.")

# Model Hyperparameters       =====================================================
#Flag là cách truyền thông số vào chương trình để chạy model với nhiều cấu hình khác nhau.

# số chiều vector input vao CNN,hay chieu rong cua matrix input,hay chiều của Vector đặc trưng là 128
tf.flags.DEFINE_integer("embedding_dim", 64, "Dimensionality of character embedding (default: 128)")
#Kích thước của bộ lọc filter là 3,4 hoặc 5
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
#Số lượng filter là 128
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
#Khả năng xóa của mỗi node trong 1 layer là như nhau
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")

#
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# Training parameters           ====================================================
#Flag là cách truyền thông số vào chương trình để chạy model với nhiều cấu hình khác nhau.

#để tiết kiệm bộ nhớ tập train sẽ được chia nhỏ thành các batch.Batch size, tức là số lượng ví dụ trong mỗi batch, là 64.
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
#num_epochs: Model đi qua tập train là num_epochs=200 lần
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
# Đánh giá model sau 100 steps
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
# lưu model sau 100 steps
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
#chỉ lưu trữ num_checkpoints=5 checkpoints
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")

# Misc Parameters              ======================================================
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
FLAGS = tf.flags.FLAGS

def preprocess():
    # Data Preparation
    # ==================================================

    # Load data
    print("Loading data...")
    start=time.time()
    x_text, y = data_helpers.load_data_and_labels(FLAGS.data_file1, FLAGS.data_file2,FLAGS.data_file3,FLAGS.data_file4,
                                                  FLAGS.data_file5,FLAGS.data_file6)
    start1=time.time()
    print(start1-start)
    print(len(y))

    # Build vocabulary,xây dựng từ điển
    max_document_length = max([len(x.split(" ")) for x in x_text]) #độ dài của đoạn dài nhất(đoạn có số từ nhiều nhất)
    ## Create the vocabularyprocessor object, setting the max lengh of the documents.
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
    ## Transform the documents using the vocabulary.
    x = np.array(list(vocab_processor.fit_transform(x_text)))
    ## Extract word:id mapping from the object. 1140
#   vocab_dict = vocab_processor.vocabulary_._mapping
    # Randomly shuffle data
    np.random.seed(10)# tạo các random có kết quả giống nhau,thực ra trong TH này cũng đéo cần.
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    #np.random.permutation: tạo ngẫu nhiên lại array trong mảng array,
    #np.arange(len(y): tạo mảng các giá trị đều nhau.
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]

    # Split train/test set,tách dữ liệu Train và Test
    # TODO: This is very crude, should use cross-validation
    dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
    x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]

    del x, y, x_shuffled, y_shuffled

    print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
    print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))

    return x_train, y_train, vocab_processor, x_dev, y_dev

def train(x_train, y_train, vocab_processor, x_dev, y_dev):
    # Training
    # ==================================================

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
          allow_soft_placement=FLAGS.allow_soft_placement,
          log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            cnn = TextCNN(
                sequence_length=x_train.shape[1],
                num_classes=y_train.shape[1],#so luong class,kich thuoc cua chieu thu 2
                vocab_size=len(vocab_processor.vocabulary_),
                embedding_size=FLAGS.embedding_dim,
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),#danh sách các size filters
                num_filters=FLAGS.num_filters,#số lượng filter
                l2_reg_lambda=FLAGS.l2_reg_lambda)

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))#đường dẫn từ cha đến folder hiện tại os.path.curdir
            print("Writing to {}\n".format(out_dir))

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            #max_to_keep:lưu num_checkpoints checkpoints gần nhất để lưu
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

            # Write vocabulary
            vocab_processor.save(os.path.join(out_dir, "vocab.txt"))

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            def train_step(x_batch, y_batch):
                """
                A single training step
                """
                feed_dict = {
                  cnn.input_x: x_batch,
                  cnn.input_y: y_batch,
                  cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
                }
                _, step,  loss, accuracy = sess.run(
                    [train_op, global_step,  cnn.loss, cnn.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                if step == 601:
                     return 0
                else:
                    return 1

            def dev_step(x_batch, y_batch):
                """
                Evaluates model on a dev set
                """
                feed_dict = {
                  cnn.input_x: x_batch,
                  cnn.input_y: y_batch,
                  cnn.dropout_keep_prob: 1.0
                }
                step, loss, accuracy = sess.run(
                    [global_step, cnn.loss, cnn.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
              #  if writer:
               #     writer.add_summary(summaries, step)

            # Generate batches
            batches = data_helpers.batch_iter(
                list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs,shuffle=True)
            # Training loop. For each batch...
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                if(train_step(x_batch, y_batch)==0):
                     break
                current_step = tf.train.global_step(sess, global_step)
                if current_step % FLAGS.evaluate_every == 0:
                    print("\nEvaluation:")
                    dev_step(x_dev, y_dev )
                    print("")
                if current_step % FLAGS.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))

def main(argv=None):

      x_train, y_train, vocab_processor, x_dev, y_dev = preprocess()
      train(x_train, y_train, vocab_processor, x_dev, y_dev)

if __name__ == '__main__':
    tf.app.run()