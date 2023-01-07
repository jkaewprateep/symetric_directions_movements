# symetric_directions_movements

For study symmetric directions movements, for 2D movements once simulation found the problems AI reverse action movement target result from changed in X, Y co-ordinates is reversed travel or it decreases of the value than increase along with our internal class step. 

Out goals is to study of the 2D movements and games rules within the game environment to simulates AI play from the third-eyes views to screen play. One problem we alway found is without central and distance as in the PyGame website example it has the problem circle talks in the corners when it cannot act correct since the action is reversed by the change from increate to decreate value. See the attached GIF image for example, the AI select to continue by go ```LEFT``` instead of ```RIGHT``` and ```UP``` instead of ```DOWN``` .


## X and Y Co-ordinates variances ##

In simulations to find the rims of the stage of the snake games, we found that AI figured out conditions but always missed at some conditions it play backside of the different actions. In example AI play ```UP``` when it should go ```DOWN``` and the AI play ```LEFT``` when it should go ```RIGHT```. 

Now from the picture we see that we can devided it into upper path and lower path where the AI work loads is 1/4 or less because it does not need to learn of the next action and the next action to return to lower path continue. It is not ```nesterov momentum``` in ```SGD``` and we still use the traditional SGD for the optimizer. ``` Optimizer = tf.keras.optimizers.SGD( learning_rate=self.learning_rate, momentum=self.momentum, nesterov=True, name='SGD', ) ```

![Diagonos line](https://github.com/jkaewprateep/symetric_directions_movements/blob/main/98.png "Diagonos line") 
![Diagonos line](https://github.com/jkaewprateep/symetric_directions_movements/blob/main/Snakes_Unsupervised_10minutes_learning_SGD%20%2B%20MSE%2002%20.gif "Diagonos line")

#### Out background logical for AI 2D movements in stage with 3rd eyes views ####

The ```( X, Y )``` is X and Y co-ordinates and turns each corner ```1. ( 512, 256)   ==> UP``` ,  ```2. ( 512, 512)   ==> LEFT``` ,  ```3. ( 0, 512)     ==> DOWN``` and ```( 0, 0 )      ==> RIGHT``` . The possible action number 5 is possible ```{ -1 | 0 | 1 }``` for ```{ upper path | none | lower path }``` when start step is increase and pass between zones it decrease.

```
( X, Y ); X and Y Co-ordinates for the snakes player.

( 512, 256)   ==> UP       ; X increase
( 512, 512)   ==> LEFT	   ; Y increase
( 0, 512)     ==> DOWN     ; X decrease
( 0, 0 )      ==> RIGHT    ; Y decrease

contrl = snake_head_x
contr2 = food_x
contr3 = snake_head_y
contr4 = food_y
...
contr16 = ( self.possible_actions[5] * self.steps ) + self.gamescores
```

```
def request_possible_action( self ):
    ( width, height ) = self.PLE.getScreenDims()
		
    snake_head_x = self.read_current_state( 'snake_head_x' )
    snake_head_y = self.read_current_state( 'snake_head_y' )

    stage_position = ( 0, snake_head_x, snake_head_y, 512 - snake_head_x, 512 - snake_head_y, 0 )
    stage_position = tf.where([tf.math.greater_equal(stage_position, 35 * tf.ones([6, ]))], [1], [0]).numpy()[0]

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
```

## Sample output ##

Prohibited actions are actions not allowed by the current stage and the robots had negative feedbacks by the ```negative scores result``` but allowed AI desire from its current state for the correct action.

```
step: 000001 condition 3: moving right
possible_actions: [ 0  0  1  1  1 -1] to actions: [['down'], ['right'], ['up'], ['lower']]
conditions robots doing prohibited action
Seleted: ('left_1', 97)
step: 000001 condition 3: moving right
possible_actions: [ 0  0  1  0  1 -1] to actions: [['down'], ['up'], ['lower']]
conditions robots doing prohibited action
Seleted: ('left_1', 97)
step: 000001 condition 3: moving right
possible_actions: [ 0  0  1  0  1 -1] to actions: [['down'], ['up'], ['lower']]
conditions robots doing prohibited action
Seleted: ('left_1', 97)
```

## Models ##

The ```LSTM cell size 448``` is a capable size from our experiments with the same input ```input_shape = (1, 16)``` with ```batch_sizes = 500``` .

```
model = tf.keras.models.Sequential([
tf.keras.layers.InputLayer(input_shape=input_shape),

tf.keras.layers.Dense(512, activation='relu'),

tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(448, return_sequences=True, return_state=False)),
tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(448, return_sequences=True)),

])

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(192))
model.add(tf.keras.layers.Dense(5))
model.summary()
```
