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
HIDDEN = 2048
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

def model(data):
    # network weights
    
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


# Input data.
tf_train_dataset = tf.placeholder(tf.float32,
                                  shape=(BATCH_SIZE,BOARD_SIZE*BOARD_SIZE*FEATURE_PLANES))
tf_train_labels = tf.placeholder(tf.float32,
                                 shape=(BATCH_SIZE,
                                 LABEL_SIZE))
# Training computation.
logits = model(tf_train_dataset)
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



skip_minibatch = 0 # first error at 17720

def main():
    parsed = 0
    gstep = 0
    # for skip minibatch
    cmove = 0
    tmove = 0
    labels = read_labels()
    batch_move = []
    batch_target = []
    print('Training...')
    for step in range(NUM_STEPS):
        for game in pgn.GameIterator("dataset/dataset.pgn"):
            if game != None and len(game.moves) > 10 and game.winner != None:        
                res = True
                #print("Game Winner {}".format(game.winner))
                while(res):
                    try:
                        ######### skip minibatch ########
                        cmove += len(game.moves)
                        if(cmove < (skip_minibatch*BATCH_SIZE)):                            
                            gstep = math.floor(cmove/BATCH_SIZE)
                            print("Skip Minibatch %d" % gstep)
                            continue
                        #cmove = 0
                        #################################

                        res,board,label = game.next_winner()
                        
                        
                        #one hot label
                        onehot = np.zeros(LABEL_SIZE)
                        onehot[labels.index(label)] = 1
                        label = onehot

                        mtrix = to_matrix(board)
                        if game.winner == pgn.BLACK:
                            #flip side
                            mtrix = flip_board(mtrix)
                        
                        board_blank = mtrix.copy()            
                        board_blank[board_blank > 1] = 0 
                        # Only white plane
                        board_white = mtrix.copy()
                        board_white[board_white>90] = 1
                        # Only black plane
                        board_black = mtrix.copy()            
                        board_black[board_black<=90] = 1                        
                        # One-hot integer plane move number            
                        move_number = np.full((BOARD_SIZE, BOARD_SIZE), game.move, dtype=int)
                        # Zeros plane
                        zeros = np.full((BOARD_SIZE, BOARD_SIZE), 0, dtype=int)

                        planes = np.vstack((np.copy(mtrix),
                                        np.copy(board_white),
                                        np.copy(board_black),
                                        np.copy(board_blank),
                                        np.copy(move_number),
                                        np.copy(zeros)))
                        planes = np.reshape(planes, (BOARD_SIZE, BOARD_SIZE, FEATURE_PLANES))
                        planes = np.reshape(planes,BOARD_SIZE*BOARD_SIZE*FEATURE_PLANES)
                        batch_move.append(planes)
                        batch_target.append(label)
                        
                        if len(batch_move) == BATCH_SIZE:
                            feed_dict = {tf_train_dataset: batch_move, tf_train_labels: batch_target}
                            _, l, predictions = sess.run(
                            [optimizer, loss, train_prediction], feed_dict=feed_dict)
                            #print("train---------------------")
                            #if (gstep % 100 == 0):
                            gstep += 1
                            print('Game Step = %d BatchStep : %d (Minibatch) loss = %.4f' % (parsed,gstep, l), end=' ')
                            print(' accuracy = %.4f%%' % accuracy(predictions, batch_target))
                            if (gstep % 100 == 0):
                                saver.save(sess, 'logdir_model2/makruk', global_step=gstep)
                            batch_move = []
                            batch_target = []
                    except ValueError:
                        print("ValueError !=========================")
                        batch_target = []
                        batch_move = []
                        break
                    except TypeError:
                        print("TypeError !=========================")
                        batch_target = []
                        batch_move = []
                        break

                #print("Parsed = {}".format(parsed))
            parsed = parsed + 1            
        #if step % 10 == 0 and step > 0:
        #    saver.save(sess, 'logdir/makruk', global_step=step)


if __name__ == '__main__':
    main()
