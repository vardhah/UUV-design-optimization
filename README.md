# Sample efficeint and Surrogate based design optimization of UUV hull: 
In real world computer-aided design (CAD)  problem, some physics simulations are the computational bottleneck in optimization processes.
Hence, in order to do design optimization, one requires either an optimization framework that is highly sample-efficient or fast data-driven surrogate models for these complex simulations. 
Recent advances in AI can be leveraged in the field of design optimization involving complex computational physics -in both: direct optimization with numerical simulation in the loop and surrogate modeling/surrogate-based optimization. We take a use case of designing an optimal unmanned underwater vehicle (UUV) hull. For this purpose, Computational Fluid Dynamics (CFD) simulation is deployed in the optimization loop. First, we investigate and compare the sample efficiency and convergence behavior of different optimization techniques with a standard computational fluid dynamics (CFD) solver in the optimization loop.  We then develop a deep neural network (DNN) based surrogate model to approximate drag forces that would otherwise be computed via direct numerical simulation with the CFD solver.
<img src="./images/cfd_optimization.png" width="800" height="300" title="Employee Data title">

## How to use the code: 
For running you need to install two docker container 
1. Docker to run CFD: can be downloded from [here](https://hub.docker.com/r/kishorestevens/dexof/tags)
2. Docker to run CAD design and assembly from [here](https://hub.docker.com/r/vardhah/freecad) 

Once both dockers are installed, you need to install optimization algotrithms. Currently the tool supports following optimization frameworks: 
1. Bayesian optimization : For this we use [GpyOPT](http://sheffieldml.github.io/GPyOpt/)
2. [PyMoo](https://pymoo.org/): Most tradional algorithms are implemented in this python package. We used GA and Nealder Mead but extending it to other algorithms available in pyMoo is also very easy.  

Last thing we need is a parametic CAD seed design. We designed a parametric CAD with three body parts- nose, tail and cylindrical body. The seed is fully constrained in the CAD geometry and by changing the parameter various hull chapes can be designed.  For parametric purpose, we chose Myring hull as our design architecture.  
<img src="./images/myring.png" width="800" height="300" title="Employee Data title">

### Folder arrangements: 
There are three important folders:
1. cad_cfd_pipeline_surrogate_data_generation: 
2. Single_shot_optimization:
2. Trained_NN_surrogate_optim_in_loop: 

## Surrogate based design optimization:
