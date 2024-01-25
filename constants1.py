### Append stop word ###
STOP_WORD = True
### Capitalization
CAPITALIZATION = True

### small number
EPSILON = 1e-100

### Inference Types ###
GREEDY = 0
BEAM = 1; BEAM_K = 2
VITERBI = 2
INFERENCE = BEAM 

# NGRAMM
NGRAMM = 3

### Smoothing Types ###
LAPLACE = 0; LAPLACE_FACTOR = .2
INTERPOLATION = 1; LAMBDAS =  NGRAMM * [1/NGRAMM]
SMOOTHING = LAPLACE

### Append stop word ###
STOP_WORD = True

### Capitalization
CAPITALIZATION = True

## Handle unknown words TnT style
TNT_UNK = True
UNK_C = 10 #words with count to be considered
UNK_M = 10 #substring length to be considered