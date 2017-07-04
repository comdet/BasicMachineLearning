#!/usr/bin/env python
from __future__ import print_function
import sys
import random
import numpy as np
import time
from PIL import ImageGrab
from collections import deque
import pyautogui
from ctypes import *
from ctypes.wintypes import *
import tensorflow as tf
CK_D = deque()
P_LIFE = 500
P_CLASH = 0
life_base_address = 0x3A5167C4
game_pid = 15660

OpenProcess = windll.kernel32.OpenProcess
ReadProcessMemory = windll.kernel32.ReadProcessMemory
CloseHandle = windll.kernel32.CloseHandle
PROCESS_ALL_ACCESS = 0x1F0FFF

processHandle = OpenProcess(PROCESS_ALL_ACCESS, False, game_pid)

def get_life():
	data = c_float()
	bytesRead = c_float()
	if ReadProcessMemory(processHandle, life_base_address, byref(data), sizeof(data), byref(bytesRead)):
		return int(data.value)
	else:
		return -1;
def jump():
	#pyautogui.press("z")
	pyautogui.click(x=600, y=500)

def slideDown():
	#pyautogui.keyDown("/")
	pyautogui.mouseDown(x=1000,y=500);

def slideUp():
	#pyautogui.keyUp("/")
	pyautogui.mouseUp(x=1000,y=500);

def score(life):
	global P_LIFE
	if life == 0:
		return -1
	sc = -1 if (P_LIFE - life > 20 and life < P_LIFE) else 0.1	
	P_LIFE = life
	return sc

def clash(life):
	global P_CLASH
	c = True if(P_CLASH - life > 20 and life < P_CLASH) else False
	P_CLASH = life
	return c

def frame(debug=False):
	global CK_D
	img = ImageGrab.grab(bbox=(480,270,480+640,270+360)) #bbox specifies specific region (bbox= x,y,width,height *starts top-left)
	img = np.array(img)
	x_t = tf.image.rgb_to_grayscale(img)    
	img = x_t.eval()
	#life = np.argmax(img[45,47:570])+47 #location of white bar
	#lifep = np.amax(img[45,47:570])
	#life = life if (life > 50 and life < 540) else 0
	life = get_life()
	vscore = score(life)
	
	slide = True if img[317,563] < 140 else False
	jump = True if img[312,41] < 140 else False
	
	isClash = clash(life)

	isplay = (150 < img[21,310] <= 175) or (img[21,310] == 200)
	space = img[110,300] == img[110,330] == img[130,288] == img[150,288] == 180	
	#terminate = img[0,0] == 0
	#back section of touch prevent nn remember 
	img[270:0,0:118] = 0
	img[270:0,522:] = 0
	img = img[60:300,140:380].copy()
	
	#cv2.imshow('test',img)
	#cv2.waitKey(0)
	img = cv2.resize(img, (80, 80))
	#img = img.reshape([-1,80,80,1])
	#cv2.imshow('test',img)
	#cv2.waitKey(0)
	
	#if not CK_D: #queue empty
	#	CK_D.append(img)
	#	CK_D.append(img)
	#	CK_D.append(img)
	#	CK_D.append(img)
	#CK_D.append(img)
	#CK_D.popleft()
	#s_t = np.stack((CK_D[0], CK_D[1], CK_D[2], CK_D[3]), axis=2)
	
	if debug == True:
		print('Score : {0}, Jump {1}, Slide {2}, Terminate {3}'.format(vscore,jump,slide,terminate),end="")

	return img, jump, slide, space, isplay , isClash, vscore

def cast_test():	
	i = 0
	prv = ''
	while True:
		img, jump, slide, space, isPlay, isClash = frame()
		#wait until game start
		#if isplay:
		#	#game playing		
		#if isClash:
		#	i += 1
		now = '{}: Jump {}, Slide {}, Space {}, isPlay {}, isClash {}'.format(i,jump,slide,space,isPlay,isClash)
		if now != prv:
			print(now)
		prv = now