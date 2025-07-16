1. Create environment: 

conda pysammos_beta create -f pysammos-env.yml
conda activate pysammos_beta

2. Run CG: 

The executable file for now is "Coarse_Graining_Class.py"
Therein, in the "main" section of the script is the input configuration, slightly commented (I hope you can follow). Feel free to change inputs to fit your post-processing workflow. In the command line you can input:

python Coarse_Graining_Class.py

You can add NUMBA_NUM_THREADS parser argument to control the number of cores that Numba will use. If not specified will use the maximum available. 

NUMBA_NUM_THREADS=4 python Coarse_Graining_Class.py

