
INC	:= -I"/usr/local/cuda-12.3/targets/x86_64-linux"/include -I.
LIB	:= -L"/usr/local/cuda-12.3/targets/x86_64-linux/lib" -lcudart -lcusolver

NVCCFLAGS	:= -lineinfo -arch=sm_86 --ptxas-options=-v --use_fast_math -Xlinker --verbose

all:	spectralClustering
		nvcc spectralClustering.cu -o spectralClustering $(INC) $(NVCCFLAGS) $(LIB)

spectralClustering: spectralClustering.cu Makefile

clean:
	rm -f spectralClustering