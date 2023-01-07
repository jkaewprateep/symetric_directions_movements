# symetric_directions_movements
For study symmetric directions movements


## X and Y Co-ordinates variances ##

In simulations to find the rims of the stage of the snake games, we found that AI figured out conditions but always missed at some conditions it play backside of the different actions. In example AI play ```UP``` when it should go ```DOWN``` and the AI play ```LEFT``` when it should go ```RIGHT```. 

Now from the picture we see that we can devided it into upper path and lower path where the AI work loads is 1/4 or less because it does not need to learn of the next action and the next action to return to lower path continue. It is not ```nesterov momentum``` in ```SGD``` and we still use the traditional SGD for the optimizer. ``` Optimizer = tf.keras.optimizers.SGD( learning_rate=self.learning_rate, momentum=self.momentum, nesterov=True, name='SGD', ) ```

![Diagonos line](https://github.com/jkaewprateep/symetric_directions_movements/blob/main/98.png "Diagonos line") 
![Diagonos line](https://github.com/jkaewprateep/symetric_directions_movements/blob/main/Snakes_Unsupervised_10minutes_learning_SGD%20%2B%20MSE%2002%20.gif "Diagonos line")

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
