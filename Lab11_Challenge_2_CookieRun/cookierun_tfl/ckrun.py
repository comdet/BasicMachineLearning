#!/usr/bin/env python
from __future__ import division, print_function, absolute_import
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
import numpy as np
import sys
import random
from os import listdir
from os.path import isfile, isdir, join
import os,datetime
import game

ACTIONS = 3 # not press , jump , slide , space 
input_size = 80
#   cv7   mp	cv5   mp	cv3   mp
#80 -> 74 -> 37 -> 33 -> 17 -> 15 -> 8x64(4096) -> 512 -> 1024 -> 2
def setup_network(input_size,output_class,learning_rate=0.001):
	network = input_data(shape=[None, input_size, input_size, 4], name='input')
	network = conv_2d(network, 32, 8, strides=4, activation='relu', regularizer="L2", padding="same")
	network = max_pool_2d(network, 2)
	network = local_response_normalization(network)
	network = conv_2d(network, 64, 4, strides=2, activation='relu', regularizer="L2", padding="same")
	#network = max_pool_2d(network, 2)	
	network = local_response_normalization(network)
	network = conv_2d(network, 64, 3, activation='relu', regularizer="L2", padding="same")
	#network = max_pool_2d(network, 2)
	network = fully_connected(network, 512, activation='tanh')
	#network = dropout(network, 0.5)
	network = fully_connected(network, output_class, activation='softmax')
	network = regression(network, optimizer='adam',loss='categorical_crossentropy',learning_rate=0.001,name='target')
	return network


def play(model):
	startPlay = False
	endPlay = False
	jump_pressing = False
	slide_pressing = False
	
	while(True):
		img, jump, slide, space, isPlay, isClash, _ = game.frame()
		img = np.reshape(img,[-1,80,80,1])
		#feed forward to get action
		readout_t = model.predict(img)[0]
		action_index = np.argmax(readout_t)		
		print(readout_t)
		if not startPlay and isPlay: #first time playing
			startPlay = True
			print("Game Started!")
		if startPlay and not isPlay: #game end			
			break;
		if isPlay: #while playing
			if action_index == 1 and not jump_pressing and not slide_pressing: #jump
				print('jump!')
				game.jump()
				jump_pressing = True
			elif action_index == 2 and not slide_pressing:
				print('\/Slide!')
				game.slideDown()
				slide_pressing = True
			elif action_index == 0 and jump_pressing:
				jump_pressing = False
			elif action_index == 0 and slide_pressing:
				slide_pressing = False
				print('/\Slide!')
				game.slideUp()

def main():
	net = setup_network(80,3)
	model = tflearn.DNN(net, tensorboard_verbose=3, tensorboard_dir="log/",checkpoint_path='model_checkpoint/ck_checkpoint')
	#model.load('./model_checkpoint/ck_checkpoint-1920')
	play(model)

if __name__ == "__main__":
	main()
