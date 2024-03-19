NVCC = nvcc
CFLAGS = --compiler-options -Wall  -O3 -I../include -I../GLASS  

examples: pcg.exe pcg_dp.exe pcg_many.exe


pcg.exe: pcg_solve.cu
	$(NVCC) $(CFLAGS) -DKNOT_POINTS=3 -DSTATE_SIZE=2 pcg_solve.cu -o pcg.exe 

pcg_dp.exe: pcg_solve_dp.cu
	$(NVCC) $(CFLAGS) -DKNOT_POINTS=3 -DSTATE_SIZE=2 pcg_solve.cu -o pcg_dp.exe 

pcg_many.exe: pcg_multisolve.cu
	$(NVCC) $(CFLAGS) -DKNOT_POINTS=30 -DSTATE_SIZE=2 pcg_multisolve.cu -o pcg_many.exe 

clean:
	rm -rf *.exe