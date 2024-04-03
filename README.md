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
int pcg_solve<typename T>(cbtd_t *h_S, T *h_gamma, T *h_lambda, unsigned stateSize, unsigned knotPoints, pcg_config *config = default_config);

``` 

h_S[I]: compressed block-tridiagonal format S

h_gamma[I]: $\gamma$

h_lambda[I/O]: $\lambda$ initial guess

stateSize[I]: state size

knotPoints[I]: knot points

## Compilation

```
nvcc -I../include -I../GLASS -DKNOT_POINTS=3 -DSTATE_SIZE=2 pcg_solve.cu -o pcg.exe 

```

Note: In addition to passing KNOT_POINTS and STATE_SIZE in the api, you need to pass it in while compiling. This is double declaration
will be removed in future work. It is currently required to have them as compile time constants.

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
