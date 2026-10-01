# Standalone build of the GBD-PCG examples. Sequential nvcc; never runs a benchmark.
#
# GLASS_DIR selects the GLASS checkout the headers compile against. The default is this
# repository's own submodule; a consumer that pins GLASS itself passes its path, for example
# MPCGPU: `make -C GBD-PCG/examples test GLASS_DIR=../../GLASS` (MPCGPU's attested route is
# tools/build.py, which puts its top-level GLASS on the include path the same way).
.NOTPARALLEL:
NVCC ?= nvcc
ARCH ?= sm_120
GLASS_DIR ?= ../GLASS
STATE_SIZE ?= 14
KNOT_POINTS ?= 32
EXTRA_FLAGS ?=
NVCCFLAGS = -std=c++17 -O3 -DNDEBUG -arch=$(ARCH) -I../include -I$(GLASS_DIR) \
            -DSTATE_SIZE=$(STATE_SIZE) -DKNOT_POINTS=$(KNOT_POINTS) $(EXTRA_FLAGS)
HEADERS = $(wildcard ../include/*.cuh)

.PHONY: examples test clean glass-check
examples: glass-check pcg.exe pcg_dp.exe test_spd.exe test_bdmv.exe

glass-check:
	@test -f "$(GLASS_DIR)/glass.cuh" || { echo "GLASS not found at GLASS_DIR=$(GLASS_DIR): run 'git submodule update --init' or pass GLASS_DIR=<path to a GLASS checkout>" >&2; exit 1; }

pcg.exe: test_api.cu $(HEADERS)
	$(NVCC) $(NVCCFLAGS) -DTEST_DOUBLE=0 $< -o $@
pcg_dp.exe: test_api.cu $(HEADERS)
	$(NVCC) $(NVCCFLAGS) -DTEST_DOUBLE=1 $< -o $@
test_spd.exe: test_pcg_spd.cu $(HEADERS)
	$(NVCC) $(NVCCFLAGS) $< -o $@
test_bdmv.exe: test_bdmv.cu $(HEADERS)
	$(NVCC) $(NVCCFLAGS) $< -o $@

# Self-contained known-solution gates (GPU required). test_bdmv.exe takes a strips file.
test: examples
	./pcg.exe host
	./pcg.exe device
	./pcg_dp.exe host
	./test_spd.exe

clean:
	rm -f *.exe
