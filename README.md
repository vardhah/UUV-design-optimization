# Sample efficeint and Surrogate based design optimization of UUV hull: 
In real world computer-aided design (CAD)  problem, some physics simulations are the computational bottleneck in optimization processes.
Hence, in order to do design optimization, one requires either an optimization framework that is highly sample-efficient or fast data-driven surrogate models for these complex simulations. 
Recent advances in AI can be leveraged in the field of design optimization involving complex computational physics -in both: direct optimization with numerical simulation in the loop and surrogate modeling/surrogate-based optimization. We take a use case of designing an optimal unmanned underwater vehicle (UUV) hull. For this purpose, Computational Fluid Dynamics (CFD) simulation is deployed in the optimization loop. First, we investigate and compare the sample efficiency and convergence behavior of different optimization techniques with a standard computational fluid dynamics (CFD) solver in the optimization loop.  We then develop a deep neural network (DNN) based surrogate model to approximate drag forces that would otherwise be computed via direct numerical simulation with the CFD solver.
<img src="./images/cfd_optimization.png" width="800" height="300" title="Employee Data title">

What we offer in this work : 
-  An evaluation of different optimization methods on a real-world design
optimization problem, which may be useful for practitioners considering adopting these algorithms.
-  Demonstration that a neural network can accurately capture a complex property of two-way coupled solid-fluid dynamics and when used in desing optimization it can get similar optimal design but  100,000 times faster than CFD in loop optimization.
-  A ready-to-use software package for drag-based design optimization for nautical and aeronautical hull. 


## How to reuse the code: 
For running you need to install two docker container 
1. Docker to run CFD: can be downloded from [here](https://hub.docker.com/r/kishorestevens/dexof/tags)
2. Docker to run CAD design and assembly from [here](https://hub.docker.com/r/vardhah/freecad) 

Once both dockers are installed, you need to install optimization algotrithms. Currently the tool supports following optimization frameworks: 
1. Bayesian optimization : For this we use [GpyOPT](http://sheffieldml.github.io/GPyOpt/)
2. [PyMoo](https://pymoo.org/): Most tradional algorithms are implemented in this python package. We used GA and Nealder Mead but extending it to other algorithms available in pyMoo is also very easy.  

Last thing we need is a parametic CAD seed design. We designed a parametric CAD with three body parts- nose, tail and cylindrical body. The seed is fully constrained in the CAD geometry and by changing the parameter various hull chapes can be designed.  For parametric purpose, we chose Myring hull as our design architecture.  
<img src="./images/myring.png" class="center" width="500" height="300" >

### Folder arrangements: 
Since there is a lot of tooling involved , we tried to keep our folder seperate process seperate until we get some free time to clean and make it more systematic (guess when? ). Currently there are three important folders:
1. cad_cfd_pipeline_surrogate_data_generation: This is used for data geeneration. SInce each CFD sim generated lot of data and it will fillyour hard drive sooner than you even think. we clean everything except the required minimal data. Given a deisgn space i.e. shape range paprameter and comutation resources and time, it gathers the data in tabular format and store it. 
2. Single_shot_optimization: From this folder, we run CFD in loop optimization and test different optimization methods on a given test problem. 
2. Trained_NN_surrogate_optim_in_loop:  From this folder, we first train our  surrogate based on collected data and then use it for different test probelms and evalute the same with CFD in loop. 

## Surrogate based design optimization:
These are some results : 

<img src="./images/picture1.png" class="center" width="392" height="240"><img src="./images/picture3.png" class="center" width="392" height="240" >
##