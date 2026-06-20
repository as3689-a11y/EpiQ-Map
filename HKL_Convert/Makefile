# Build libhklBen.so next to hklBen.py.
#
#   make            optimized OpenMP build
#   make debug      with symbols and warnings
#   make clean
#
# At runtime, set OMP_NUM_THREADS to control the thread count, e.g.
#   OMP_NUM_THREADS=16 python autoRSM.py config.txt

CC      ?= gcc
CFLAGS  ?= -O3 -march=native -funroll-loops -fopenmp -fPIC -Wall
LDFLAGS ?= -shared -fopenmp

libhklBen.so: hklBen.c
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $<

debug: CFLAGS = -O0 -g -fopenmp -fPIC -Wall -Wextra
debug: libhklBen.so

clean:
	rm -f libhklBen.so

.PHONY: debug clean
