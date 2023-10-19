# GBD-PCG
GBD-PCG is a preconditioned conjugate gradient solver for optimal control problems. It solves linear systems of the form 

$\Phi^{-1} S \lambda = \Phi^{-1} \gamma$ 

where,

- $S$ is positive semidefinite
- $S$ and $\Phi^{-1}$ are square, block-tridiagonal matrices 
- Block dimension stateSize
- Matrix dimension stateSize * knotPoints.

## Requirements
Requires: CUDA 11.0+


## Datatypes


```
// compressed sparse row type
csr_t<typename T>{
    T *row_ptr;
    T *col_ind;
    T *val;
    T *nnz;
}

pcg_config<typename T>{
    T pcg_exit_tolerance;
    unsigned pcg_max_iter;
    dim3 pcg_grid;
    dim3 pcg_block;
	int empty_pinv;
}

```

## API functions

```
int pcg_solve<typename T>(cbtd_t *h_S, T *h_gamma, T *h_lambda, unsigned stateSize, unsigned knotPoints, bool warmStart = false, pcg_config *config = default_config);

``` 

h_S[I]: compressed block-tridiagonal format S

h_Pinv[I]: $\Phi^{-1}$

h_gamma[I]: $\gamma$

h_lambda[I/O]: $\lambda$ initial guess

stateSize[I]: state size

knotPoints[I]: knot points

warmStart[I]: if false $\lambda$ will be initialized to [0, ... 0] else h_lambda will be used as initial guess
config[I]: pcg parameters

```
int pcg_solve<typename T>(cbtd_t *d_S, cbtd_t *d_Pinv, T *d_gamma, T *d_lambda, unsigned stateSize, unsigned knotPoints, bool warmStart = false, pcg_config *config = default_config);

```

d_S: compressed block-tridiagonal format S device pointer

d_Pinv: compressed block-tridiagonal format $\Phi^{-1}$ device pointer

d_gamma: $\gamma$ device pointer

d_lambda: $\lambda$ device pointer



## Citing
To cite this work in your research, please use the following bibtex:
```
@misc{adabag2023mpcgpu,
      title={MPCGPU: Real-Time Nonlinear Model Predictive Control through Preconditioned Conjugate Gradient on the GPU}, 
      author={Emre Adabag and Miloni Atal and William Gerard and Brian Plancher},
      year={2023},
      eprint={2309.08079},
      archivePrefix={arXiv},
      primaryClass={cs.RO}
}
```
