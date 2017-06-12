import sys
import os
import copy
import random

import gym
import tensorflow as tf
import numpy as np

gama = 0.8

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)


with tf.device('/gpu:0'):
	sess = tf.InteractiveSession()

	x = tf.placeholder("float", shape=[None, 4])
	y_ = tf.placeholder("float", shape=[None, 2])

	W_fc1 = weight_variable([4, 20])
	b_fc1 = bias_variable([20])
	h_fc1 = tf.nn.relu(tf.matmul(x, W_fc1) + b_fc1)
	
	W_fc2 = weight_variable([20, 20])
	b_fc2 = bias_variable([20])
	h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)

	W_fc3 = weight_variable([20, 2])
	b_fc3 = bias_variable([2])
	y_out = tf.matmul(h_fc2, W_fc3) + b_fc3
	
	loss = tf.reduce_sum(tf.square(y_ - y_out))
	train_step = tf.train.GradientDescentOptimizer(1e-3).minimize(loss)
	sess.run(tf.global_variables_initializer())
	
	
Epsilon = 0

def get_net_out(observation):
	x_in = np.zeros((1,4))
	x_in[0] = np.array(observation)
	return np.reshape( y_out.eval( feed_dict={x: x_in, y_: np.zeros((1,2))} ), (2) )

def get_action(net_out):
	r = random.uniform(0, 1)
	if r > Epsilon:
		return random.randint(0, 1)
	
	return np.argmax( net_out )
		
def get_max_q(net_out):
	return np.max( net_out )
	
batch_size = 128
def train(mem):
	bsize = min(len(mem), batch_size)
	sample_mem = random.sample(mem, bsize)
	x_list = np.zeros((bsize, 4), np.float32)
	y_list = np.zeros((bsize, 2), np.float32)
	for i in range(bsize):
		observation = sample_mem[i][0]
		action = sample_mem[i][1]
		reward = sample_mem[i][2]
		new_observation = sample_mem[i][3]
		net_out = sample_mem[i][4]
		
		x_list[i] = np.array(observation)
		
		new_net_out = get_net_out(new_observation)
		max_q = get_max_q(new_net_out)
		if action == 0:
			y_list[i, 0] = reward + gama * max_q
			y_list[i, 1] = net_out[1]
		else:
			y_list[i, 0] = net_out[0]
			y_list[i, 1] = reward + gama * max_q
		
	train_step.run(feed_dict={x: x_list, y_: y_list})
		
max_mem_len = 1024
mem = []
def remember(data):
	mem.append(data)
	if len(mem) > max_mem_len:
		del mem[0]

env = gym.make('CartPole-v0')
env = gym.wrappers.Monitor(env, '/home/alanmain/Reinforcement_Learning/out/cartpole-q')
max_iter = 400
for i_episode in range(max_iter):
	Epsilon += (1.5 / max_iter)
	observation = env.reset()
	for t in range(env.spec.timestep_limit):
		#env.render()
		net_out = get_net_out(observation)
		action = get_action(net_out)
		#print action
		new_observation, reward, done, info = env.step(action)
		if done and t < env.spec.timestep_limit - 1:
			reward = -1
		elif done and t == env.spec.timestep_limit - 1:
			reward = 1
		remember((observation, action, reward, new_observation, net_out))
		
		train(mem)
			
		observation = copy.copy(new_observation)
		if done:
			print "Epsilon: ", Epsilon
			print "Episode ", i_episode, "/", str(max_iter), " finished after ", t+1, " timesteps"
			break
            
	

