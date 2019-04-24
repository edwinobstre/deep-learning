
3/25/2019

Problems:

Missing street sections
Discretization is not correct: locations on a trajectory are not discretized to correct graph nodes.  

Coding issue: 
1. separation of data manipulation and file io
2. low level programming intead of leveraging existing packages. 
3. flip the plot 

Plan: 
Further investigation of missing street sections. 
Focus on one street section, check which way type it is. 

Debug the code to generate plots. 


The interface of plotting function. One choice is to create an iterator with python "yield"
"for images in street_plottting:"

Another choice: "street_plotting.get_image_batch(50)"


Code the neural network



    











