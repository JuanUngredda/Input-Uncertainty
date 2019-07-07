# Input-Uncertainty

Hi Michael!!

Hi Juan!!!

## TestFuns
contains all code to do with simulators, their parameter ranges and uncertainty parameters (not quite ready yet).  All simultors and information sources go here. (No optimizer or modelling code)

## input_uncertainty_general.py
this is a customised version of the original code.

## Modular Version
This is a version of the original code with functions split up over files.
## IU_Optimizer
contains all files and code to do with KG and Delta loss  and gaussain processes and monte carlo etc. All ***modelling and optimizer*** code goes here.

## runner_(your favourite problem).py
each file loads the code from IU_optimizer, and a single test function from TestFuns, it then runs the optmizer and saves outputs (doesnt work yet)


## Michael's personal coding preferences
- code can be as ugly and messy as it needs ot be to get the job done, we want results plots
- therefore all functions starts with a docstring that contains a tiny descrition of function purpose, all arguments and returned values, this has saved me some much time and allows reuse of code etc. eg 
```def myfun(x,y,z,verbose=False):
      """ computes the product of inputs.
      ARGS
        x: float
        y: float
        z: float
        verbose: bool, print out function progress
      
      RETRUNS
        p: float, x*y*z
      """
      
      (here goes the code, can be as messy as as one you like, as
       long as the docstring makes clear what the function is doing)
```

- no line of code is crazy long (more than 80 characters wide), use multiple lines for long expresions!!! Again, even if code is hacky/messy this is the easiest way to make it easier to understand with no extra effort!! Its free! eg
```
# this is horrible to read
D = np.exp(-0.5\*np.sum(np.matmul(np.array([0,1,5,3,6,1,17]).reshape((3,2)), np.array([6,1,17,0,0,4]).reshape((2, 3)))))

# this is the same and soooo much easier to read at no cost!
A = np.array([0, 1, 5, 3, 6, 1, 17]).reshape((3, 2))
B = np.array([6, 1, 17, 0, 0, 4]).reshape((2, 3))
C = np.sum(np.matmul(A, B))
D = np.exp(-0.5*C)
```

- use assert statements to sanity check function inputs, this is a mega time saver as it stops errors propogating through code!!
```
def myfunc(x,y,z,vebose):
   "" comments on arguments etc""
   assert len(x.shape)==1; "x is not a 1D array!"
   assert len(y.shape)==1; "y is not a 1D array!"
   assert len(z.shape)==1; "z is not a 1D array!"
   assert len(z)==len(y) & len(x)==len(z); "x, y, z are not same lengths!"
   
   (put the most messy code you like here, as long as we know that the 
    inputs are legit it is not going to spread to other functions
   
   
```
- with 1. docstrings, 2. lines not too long, and 3. assert statements, these safety checks enable to use of hacky messy unstable code that you can share with others and reuse in the future!
