# symetric_directions_movements
For study symmetric directions movements


## X and Y Co-ordinates variances ##

In simulations to find the rims of the stage of the snake games, we found that AI figured out conditions but always missed at some conditions it play backside of the different actions. In example AI play ```UP``` when it should go ```DOWN``` and the AI play ```LEFT``` when it should go ```RIGHT```. 

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
