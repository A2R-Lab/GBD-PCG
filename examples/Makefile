NVCC = nvcc
CFLAGS = --compiler-options -Wall  -O3 -I../include -I../GLASS  

examples: pcg.exe pcg_dp.exe


pcg.exe:
	$(NVCC) $(CFLAGS) -DKNOT_POINTS=3 -DSTATE_SIZE=2 pcg_solve.cu -o pcg.exe 

pcg_dp.exe:
	$(NVCC) $(CFLAGS) -DKNOT_POINTS=3 -DSTATE_SIZE=2 pcg_solve.cu -o pcg_dp.exe 

clean:
	rm -rf *.exe