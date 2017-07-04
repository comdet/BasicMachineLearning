from __future__ import print_function
import pgn
import time
import numpy as np
import random
import os
import fnmatch
import tensorflow as tf
import math 

parsed = 0
move = 0
BOARD_SIZE = 8
FEATURE_PLANES = 6
FILTERS = 128
HIDDEN = 512
LABEL_SIZE = 1428
BATCH_SIZE = 8192 #1024
NUM_STEPS = 150001

def read_labels():
    labels_array = []
    with open("label.data") as f:
        lines = str(f.readlines()[0]).split(" ")
        for label in lines:
            if(label != " " and label != '\n'):
                labels_array.append(label)
    return labels_array

def to_matrix(board):
    mtrix = np.array(board)
    mtrix = mtrix[:-1]
    mtrix = np.reshape(mtrix,[8,8])
    mtrix = mtrix.view(np.uint32).copy() #<<<<<<<<<<<<<<<
    mtrix[mtrix == 49] == 0
    return mtrix

def flip_board(board):
    a = np.fliplr(board)
    a = np.flipud(a) 
    a[a>90] += 500
    a[a<90] += 32
    a[a>500] -= 532
    a[a==32] = 1
    return a

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding="SAME")


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])

def model2(data):
    # network weights
    HIDDEN = 2048

    W_fc1 = weight_variable([BOARD_SIZE*BOARD_SIZE*FEATURE_PLANES, HIDDEN])
    b_fc1 = bias_variable([HIDDEN])

    W_fc2 = weight_variable([HIDDEN, HIDDEN])
    b_fc2 = bias_variable([HIDDEN])

    W_fc3 = weight_variable([HIDDEN, HIDDEN])
    b_fc3 = bias_variable([HIDDEN])

    W_fc4 = weight_variable([HIDDEN, LABEL_SIZE])
    b_fc4 = bias_variable([LABEL_SIZE])
    
    h_fc1 = tf.nn.relu(tf.matmul(data, W_fc1) + b_fc1)
    h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)
    h_fc3 = tf.nn.relu(tf.matmul(h_fc2, W_fc3) + b_fc3)    

    # readout layer
    readout = tf.matmul(h_fc3, W_fc4) + b_fc4
    return readout

def model(data):
    # network weights
    W_conv1 = weight_variable([BOARD_SIZE, BOARD_SIZE, FEATURE_PLANES, FILTERS])
    b_conv1 = bias_variable([FILTERS])

    W_conv2 = weight_variable([5, 5, FILTERS, FILTERS])
    b_conv2 = bias_variable([FILTERS])

    W_conv3 = weight_variable([3, 3, FILTERS, FILTERS])
    b_conv3 = bias_variable([FILTERS])

    W_fc1 = weight_variable([HIDDEN, HIDDEN])
    b_fc1 = bias_variable([HIDDEN])

    W_fc2 = weight_variable([HIDDEN, LABEL_SIZE])
    b_fc2 = bias_variable([LABEL_SIZE])

    # hidden layers
    h_conv1 = tf.nn.relu(conv2d(data, W_conv1, 1) + b_conv1)
    h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2, 3) + b_conv2)
    h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3, 1) + b_conv3)
    h_pool3 = max_pool_2x2(h_conv3)
    h_flat = tf.reshape(h_pool3, [-1, HIDDEN])
    h_fc1 = tf.nn.relu(tf.matmul(h_flat, W_fc1) + b_fc1)

    # readout layer
    readout = tf.matmul(h_fc1, W_fc2) + b_fc2
    return readout


model_sche = 2

if model_sche == 1:
    # Input data.
    tf_train_dataset = tf.placeholder(tf.float32,
                                      shape=(None,
                                      BOARD_SIZE,
                                      BOARD_SIZE,
                                      FEATURE_PLANES))
    tf_train_labels = tf.placeholder(tf.float32,
                                     shape=(None,
                                     LABEL_SIZE))
elif model_sche ==2:
    # Input data.
    tf_train_dataset = tf.placeholder(tf.float32,
                                      shape=(None,BOARD_SIZE*BOARD_SIZE*FEATURE_PLANES))
    tf_train_labels = tf.placeholder(tf.float32,
                                     shape=(None,
                                     LABEL_SIZE))

# Training computation.
logits = model2(tf_train_dataset)
loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(  # pylint: disable=invalid-name
                                      labels=tf_train_labels, logits=logits,
                                      dim=-1, name=None))
# Optimizer.
optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)

# Predictions for the training, validation, and test data.
train_prediction = tf.nn.softmax(logits)

# Initialize session all variables
sess = tf.InteractiveSession()
saver = tf.train.Saver()
sess.run(tf.initialize_all_variables())
checkpoint = tf.train.get_checkpoint_state("logdir_model2")
if checkpoint and checkpoint.model_checkpoint_path:
    saver.restore(sess, checkpoint.model_checkpoint_path)
    print ("Successfully loaded:", checkpoint.model_checkpoint_path)

def prepare_board(mtrix,move):    
    board_blank = mtrix.copy()            
    board_blank[board_blank > 1] = 0 
    # Only white plane
    board_white = mtrix.copy()
    board_white[board_white>90] = 1
    # Only black plane
    board_black = mtrix.copy()            
    board_black[board_black<=90] = 1              
    # One-hot integer plane move number            
    move_number = np.full((BOARD_SIZE, BOARD_SIZE), move, dtype=int)
    # Zeros plane
    zeros = np.full((BOARD_SIZE, BOARD_SIZE), 0, dtype=int)

    planes = np.vstack((np.copy(mtrix),
                    np.copy(board_white),
                    np.copy(board_black),
                    np.copy(board_blank),
                    np.copy(move_number),
                    np.copy(zeros)))
    planes = np.reshape(planes, (BOARD_SIZE, BOARD_SIZE, FEATURE_PLANES))
    ################# Model 2 ####################
    if model_sche == 2:
        planes = np.reshape(planes,BOARD_SIZE*BOARD_SIZE*FEATURE_PLANES)
    ##############################################
    return planes

def main():
    labels = read_labels()    
    game = pgn.PGNGame()
    side = 'w'# input("Side (w/b) : ")
    game.winner = pgn.WHITE if side == 'w' else pgn.BLACK
            
    while True:
        if(game.winner):
            print("==========================")
            print("==========================")
            game.print_board_compat()
            board = game.board
            mtrix = to_matrix(board)
            if game.winner == pgn.BLACK:
                #flip side
                mtrix = flip_board(mtrix)
            planes = prepare_board(mtrix,game.move)
            test_data = []
            test_data.append(planes)            
            prediction = train_prediction.eval(feed_dict={tf_train_dataset: test_data})
            #sort reverse
            outter = prediction[0].copy()        
            maxind = outter.argsort()[-100:][::-1]
            for i in range(10):                
                print("index : %d percentage %.4f : %s" % (i,outter[maxind[i]],labels[maxind[i]]))
            mindex = input("Pickup index to move (1-10) : ")
            do_move = labels[maxind[int(mindex)]]
            print("Pickup move : %s" % do_move)
            game.next_ext(do_move)
            game.print_board_compat()
            ot_move = input("Black Move san ([a-h1-8a-h1-8]) : ")
            game.next_ext(ot_move)
            print("==========================")
            print("==========================")
            print()
            
if __name__ == '__main__':
    main()
