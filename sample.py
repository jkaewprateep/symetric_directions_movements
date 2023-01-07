"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
PY GAME

https://pygame-learning-environment.readthedocs.io/en/latest/user/games/snake.html

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""

import os
from os.path import exists

import tensorflow as tf

import ple
from ple import PLE
from ple.games.snake import Snake as Snake_Game

from pygame.constants import K_a, K_s, K_d, K_w, K_h

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
None
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)
print(physical_devices)
print(config)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: Variables
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
actions = { "none_1": K_h, "left_1": K_a, "down_1": K_s, "right1": K_d, "up___1": K_w }

steps = 0
reward = 0
gamescores = 0
nb_frames = 100000000000

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: Environment
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
game_console = Snake_Game(width=512, height=512, init_length=3)
p = PLE(game_console, fps=30, display_screen=True, reward_values={})
p.init()

obs = p.getScreenRGB()

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: Class / Functions
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
class AgentQueue:

	def __init__( self, PLE, momentum = 0.1, learning_rate = 0.001, batch_size = 500, epochs=1, actions={ "none_1": K_h, "left_1": K_a, "down_1": K_s, "right1": K_d, "up___1": K_w }):
		self.PLE = PLE
		self.previous_snake_head_x = 0
		self.previous_snake_head_y = 0
		self.model = tf.keras.models.Sequential([ ])
		self.momentum = momentum
		self.learning_rate = learning_rate
		self.batch_size = batch_size
		self.epochs = epochs
		self.optimizer = tf.keras.optimizers.SGD( learning_rate=self.learning_rate, momentum=self.momentum, nesterov=True, name='SGD', )
		self.lossfn = tf.keras.losses.MeanSquaredLogarithmicError(reduction=tf.keras.losses.Reduction.AUTO, name='mean_squared_logarithmic_error')
		self.history = []
		
		self.actions = { "none_1": K_h, "left_1": K_a, "down_1": K_s, "right1": K_d, "up___1": K_w }
		self.action = 0
		self.possible_actions = ( 1, 1, 1, 1, 1 )
		
		self.lives = 0
		self.reward = 0
		self.steps = 0
		self.gamescores = 0
		
		self.DATA = tf.zeros([1, 1, 1, 16 ], dtype=tf.float32)
		self.LABEL = tf.zeros([1, 1, 1, 1], dtype=tf.float32)
		for i in range(15):
			DATA_row = -9999 * tf.ones([1, 1, 1, 16 ], dtype=tf.float32)		
			self.DATA = tf.experimental.numpy.vstack([self.DATA, DATA_row])
			self.LABEL = tf.experimental.numpy.vstack([self.LABEL, tf.constant(0, shape=(1, 1, 1, 1))])
			
		for i in range(15):
			DATA_row = 9999 * tf.ones([1, 1, 1, 16 ], dtype=tf.float32)			
			self.DATA = tf.experimental.numpy.vstack([self.DATA, DATA_row])
			self.LABEL = tf.experimental.numpy.vstack([self.LABEL, tf.constant(9, shape=(1, 1, 1, 1))])	

		self.LABEL = self.LABEL[-500:,:,:,:]
		self.LABEL = self.LABEL[-500:,:,:,:]
		
		self.dataset = tf.data.Dataset.from_tensor_slices((self.DATA, self.LABEL))
		
		self.checkpoint_path = "F:\\models\\checkpoint\\" + os.path.basename(__file__).split('.')[0] + "\\TF_DataSets_01.h5"
		self.checkpoint_dir = os.path.dirname(self.checkpoint_path)

		if not exists(self.checkpoint_dir) : 
			os.mkdir(self.checkpoint_dir)
			print("Create directory: " + self.checkpoint_dir)
		
		return
		
	def build( self ):
	
		return
	
	def request_possible_action( self ):
	
		( width, height ) = self.PLE.getScreenDims()
		
		snake_head_x = self.read_current_state( 'snake_head_x' )
		snake_head_y = self.read_current_state( 'snake_head_y' )
		self.possible_actions = ( 1, 1, 1, 1, 1, 1 )
		
		"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
		# ( width, height, snake_head_x, snake_head_y )
		# {'none_1': 104, 'left_1': 97, 'down_1': 115, 'right1': 100, 'up___1': 119}
		
		# ( none, left, down, right, up, upper / lower )
		# ( 0, 0, 0, 0, 0, 0 )
		"""""""""""""""""""""""""""""""""""""""""""""""""""""""""

		stage_position = ( 0, snake_head_x, snake_head_y, 512 - snake_head_x, 512 - snake_head_y, 0 )
		stage_position = tf.where([tf.math.greater_equal(stage_position, 35 * tf.ones([6, ]))], [1], [0]).numpy()[0]



		# list_actions = [['left'], ['down'], ['right'], ['up']]
		# stage_position = ( 0, 5, 5, 512 - 5, 512 - 5 )								# ==> right and up			( 0, 0, 0, 1, 1 )	
		# stage_position = ( 0, 5, 512, 512 - 5, 512 - 512 )							# ==> right and down		( 0, 0, 1, 1, 0 )	
		# stage_position = ( 0, 512, 512, 512 - 512, 512 - 512 )						# ==> left and down			( 0, 1, 1, 0, 0 )	
		# stage_position = ( 0, 512, 5, 512 - 512, 512 - 5 )							# ==> left and up			( 0, 1, 0, 0, 1 )
		
		### upper path ###
		
		if snake_head_x == self.previous_snake_head_x and snake_head_y > self.previous_snake_head_y : 
			print( "step: " + str( self.steps ).zfill(6) + " condition 1: moving up" )
			stage_position[2] = 0
			stage_position[5] = 1
			
		### lower path ###	
			
		if snake_head_x == self.previous_snake_head_x and snake_head_y < self.previous_snake_head_y : 
			print( "step: " + str( self.steps ).zfill(6) + " condition 2: moving down" )
			stage_position[4] = 0
			stage_position[5] = -1
		
		### lower path ###
		
		if snake_head_y == self.previous_snake_head_y and snake_head_x > self.previous_snake_head_x : 
			print( "step: " + str( self.steps ).zfill(6) + " condition 3: moving right" )
			stage_position[1] = 0
			stage_position[5] = -1
			
		### upper path ###	
			
		if snake_head_y == self.previous_snake_head_y and snake_head_x < self.previous_snake_head_x : 
			print( "step: " + str( self.steps ).zfill(6) + " condition 4: moving left" )
			stage_position[3] = 0
			stage_position[5] = 1
		
		self.previous_snake_head_x = snake_head_x
		self.previous_snake_head_y = snake_head_y
	
		return stage_position
		
	def	read_current_state( self, string_gamestate ):
	
		GameState = self.PLE.getGameState()
		
		# print( GameState )
		# {'snake_head_x': 256.0, 'snake_head_y': 424.6666666666671, 'food_x': 92, 'food_y': 414, 'snake_body': [0.0, 8.518518518518533, 17.037037037037067], 
		# 'snake_body_pos': [[256.0, 424.6666666666671], [256.0, 416.14814814814855], [256.0, 407.62962962963]]}
		
		if string_gamestate in ['snake_head_x']:
			# temp = tf.cast( GameState[string_gamestate], dtype=tf.int32 )
			# temp = tf.cast( temp, dtype=tf.float32 )
			
			temp = 1.0 * GameState[string_gamestate]
			
			return temp
			
		elif string_gamestate in ['snake_head_y']:
			# temp = tf.cast( 512 - GameState[string_gamestate], dtype=tf.int32 )
			# temp = tf.cast( temp, dtype=tf.float32 )
			
			temp = 512.0 - GameState[string_gamestate]
			
			return temp
			
		elif string_gamestate in ['food_x']:
			temp = tf.cast( GameState[string_gamestate], dtype=tf.int32 )
			temp = tf.cast( temp, dtype=tf.float32 )
			
			temp = 1.0 * GameState[string_gamestate]
			
			return temp
			
		elif string_gamestate in ['food_y']:
			# temp = tf.cast( 512 - GameState[string_gamestate], dtype=tf.int32 )
			# temp = tf.cast( temp, dtype=tf.float32 )
			
			temp = 512.0 - GameState[string_gamestate]
			
			return temp
			
		elif string_gamestate in ['snake_body']:
			temp = tf.zeros([n_blocks * 1, ], dtype=tf.float32)
			return temp.numpy()[0]
			
		elif string_gamestate in ['snake_body_pos']:
			temp = tf.zeros([n_blocks * 2, ], dtype=tf.float32)
			return temp.numpy()[0]
			
		return None
		
	def random_action( self, possible_actions ): 

		snake_head_x = self.read_current_state('snake_head_x')
		snake_head_y = self.read_current_state('snake_head_y')
		food_x = self.read_current_state('food_x')
		food_y = self.read_current_state('food_y')

		distance = ( ( abs( snake_head_x - food_x ) + abs( snake_head_y - food_y ) + abs( food_x - snake_head_x ) + abs( food_y - snake_head_y ) ) / 4 )
		
		coeff_01 = distance
		coeff_02 = abs( snake_head_x - food_x )
		coeff_03 = abs( snake_head_y - food_y )
		coeff_04 = abs( food_x - snake_head_x )
		coeff_05 = abs( food_y - snake_head_y )
		
		temp = tf.constant( possible_actions, shape=(5, 1), dtype=tf.float32 )
		temp = tf.math.multiply(tf.constant([ coeff_01, coeff_02, coeff_03, coeff_04, coeff_05 ], shape=(5, 1), dtype=tf.float32), temp)
		
		action = tf.math.argmax(temp, axis=0)
		
		self.action = int(action)

		return int(action)

	def create_model( self ):
		input_shape = (1, 16)

		model = tf.keras.models.Sequential([
			tf.keras.layers.InputLayer(input_shape=input_shape),
			
			tf.keras.layers.Dense(512, activation='relu'),
			
			tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(448, return_sequences=True, return_state=False)),
			tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(448, return_sequences=True)),
			
			# tf.keras.layers.Dense(512, activation='relu'),
			# tf.keras.layers.Dropout(0.3),
			# tf.keras.layers.Dense(128, activation='relu'),
			# tf.keras.layers.Dropout(0.3),
			# tf.keras.layers.Dense(128, activation='relu'),
			# tf.keras.layers.Dropout(0.4),
		])
				
		model.add(tf.keras.layers.Flatten())
		model.add(tf.keras.layers.Dense(192))
		model.add(tf.keras.layers.Dense(5))
		model.summary()
		
		model.compile(optimizer=self.optimizer, loss=self.lossfn, metrics=['accuracy'])
		
		
		if exists( self.checkpoint_path ) :
			model.load_weights( self.checkpoint_path )
			print("model load: " + self.checkpoint_path)
			input("Press Any Key!")
			
		self.model = model

		return model

	def training( self ):
		self.history = model.fit(self.dataset, epochs=self.epochs, callbacks=[custom_callback])
		self.model.save_weights(self.checkpoint_path)
		
		return self.model

	def predict_action( self ):

		predictions = self.model.predict(tf.expand_dims(tf.squeeze(self.DATA), axis=1 ))
		
		self.action = int(tf.math.argmax(predictions[0]))
		
		
		##################### robots action prohibited #####################
		if ( self.possible_actions[self.action] == 0 ):
			print( 'conditions robots doing prohibited action' )
			self.reward = -1
			self.gamescores = self.gamescores + ( 5 * self.reward )
			self.update_DATA( self.action, self.reward, self.gamescores, to_action=False )
		####################################################################

		# action_from_list = list(actions.values())[self.action]

		return self.action

	def update_DATA( self, action, reward, gamescores, to_action=True ):
	
		self.steps = self.steps + 1
		self.reward = reward
		self.gamescores = gamescores
		self.action = action
		
		list_input = []
	
		snake_head_x = self.read_current_state('snake_head_x')
		snake_head_y = self.read_current_state('snake_head_y')
		food_x = self.read_current_state('food_x')
		food_y = self.read_current_state('food_y')
		
		if self.reward < 0 :
			self.steps = 0
			self.previous_snake_head_x = snake_head_x
			self.previous_snake_head_y = snake_head_y
		
		if ( to_action ) :
		
			self.possible_actions = self.request_possible_action()
			possible_actionname = []
			
			list_actions = [['none'], ['left'], ['down'], ['right'], ['up']]
			
			for i in range( len( self.possible_actions ) - 1 ) :
				if self.possible_actions[i] == 1 :
					possible_actionname.append( list_actions[i] ) 
				
			if self.possible_actions[5] == 1 :
				possible_actionname.append( ['upper'] )
			elif self.possible_actions[5] == -1 :
				possible_actionname.append( ['lower'] )
			
			print( 'possible_actions: ' + str( self.possible_actions ) + " to actions: " + str( possible_actionname ) )
		
		# distance = ( ( abs( snake_head_x - food_x ) + abs( snake_head_y - food_y ) + abs( food_x - snake_head_x ) + abs( food_y - snake_head_y ) ) / 4 )
		
		# upper_space = 512 - snake_head_y
		# right_space = 512 - snake_head_x
		
		# should be usable
		# contrl = upper_space
		# contr2 = snake_head_x - food_x
		# contr3 = right_space
		# contr4 = snake_head_y - food_y 
		
		contrl = snake_head_x
		contr2 = food_x
		contr3 = snake_head_y
		contr4 = food_y
		contr5 = 1 
		contr6 = 1
		contr7 = 1
		contr8 = 1
		contr9 = 1
		contr10 = 1
		contr11 = 1
		contr12 = 1
		contr13 = 1
		contr14 = 1
		contr15 = 1
		contr16 = ( -1 * self.possible_actions[5] * self.steps ) + self.gamescores
		
		list_input.append( contrl )
		list_input.append( contr2 )
		list_input.append( contr3 )
		list_input.append( contr4 )
		list_input.append( contr5 )
		list_input.append( contr6 )
		list_input.append( contr7 )
		list_input.append( contr8 )
		list_input.append( contr9 )
		list_input.append( contr10 )
		list_input.append( contr11 )
		list_input.append( contr12 )
		list_input.append( contr13 )
		list_input.append( contr14 )
		list_input.append( contr15 )
		list_input.append( contr16 )
		
		action_name = list(self.actions.values())[self.action]
		action_name = [ x for ( x, y ) in self.actions.items() if y == action_name]
		
		# print( "steps: " + str( self.steps ).zfill(6) + " action: " + str( action_name ) + " contrl: " + str(int(contrl)).zfill(6) + " contr2: " + str(int(contr2)).zfill(6) + " contr3: " +
			# str(int(contr3)).zfill(6) + " contr4: " + str(int(contr4)).zfill(6) + " contr5: " + str(int(contr5)).zfill(6) )
	
		# print( "steps: " + str( self.steps ).zfill(6) + " gamescores: " + str( self.gamescores ) + " reward: " + str(int( self.reward )).zfill(6)

		# )
		
		DATA_row = tf.constant([ list_input ], shape=(1, 1, 1, 16), dtype=tf.float32)	

		self.DATA = tf.experimental.numpy.vstack([self.DATA, DATA_row])
		self.DATA = self.DATA[-500:,:,:,:]
		
		self.LABEL = tf.experimental.numpy.vstack([self.LABEL, tf.constant(self.action, shape=(1, 1, 1, 1))])
		self.LABEL = self.LABEL[-500:,:,:,:]
		
		self.DATA = self.DATA[-500:,:,:,:]
		self.LABEL = self.LABEL[-500:,:,:,:]
	
		return self.DATA, self.LABEL, self.steps

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: Callback
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
class custom_callback(tf.keras.callbacks.Callback):

	def __init__(self, patience=0):
		self.best_weights = None
		self.best = 999999999999999
		self.patience = patience
	
	def on_train_begin(self, logs={}):
		self.best = 999999999999999
		self.wait = 0
		self.stopped_epoch = 0

	def on_epoch_end(self, epoch, logs={}):
		if(logs['accuracy'] == None) : 
			pass
		
		if logs['loss'] < self.best :
			self.best = logs['loss']
			self.wait = 0
			self.best_weights = self.model.get_weights()
		else :
			self.wait += 1
			if self.wait >= self.patience:
				self.stopped_epoch = epoch
				self.model.stop_training = True
				print("Restoring model weights from the end of the best epoch.")
				self.model.set_weights(self.best_weights)
		
		if self.wait > self.patience :
			self.model.stop_training = True

custom_callback = custom_callback(patience=8)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
: Tasks
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""
AgentQueue = AgentQueue( p )
model = AgentQueue.create_model()

for i in range(nb_frames):
	
	reward = 0
	steps = steps + 1
	
	if p.game_over():
		p.init()
		p.reset_game()
		steps = 0
		lives = 0
		reward = 0
		gamescores = 0
		
	# if ( steps == 0 ):
		# print('start ... ')

	action = AgentQueue.predict_action()
	action_from_list = list(actions.values())[action]
	
	print( "Seleted: " + str( list(actions.items())[action] ) )
	
	reward = p.act( action_from_list )
	gamescores = gamescores + ( 5 * reward )
	
	AgentQueue.update_DATA( action, reward, gamescores, to_action=True )
	
	if ( reward > 0 ):
		model = AgentQueue.training()
		
	if ( steps % 500 == 0 ):
		model = AgentQueue.training()
		
input('...')
