NVCC     := nvcc
CXX      := g++
CFLAGS   := -std=c++17 -arch=sm_121 -O3 --extended-lambda -lineinfo -Iparlaylib/include -Isrc
CXXFLAGS := -std=c++17 -O3 -march=native
TARGET   := bench

SRCDIR   := src
CU_SRCS  := $(SRCDIR)/main.cu $(SRCDIR)/bench_bandwidth.cu $(SRCDIR)/bench_latency.cu \
             $(SRCDIR)/bench_atomic_tput.cu $(SRCDIR)/bench_atomic_lat.cu

all: $(TARGET)

cpu_bw.o: $(SRCDIR)/cpu_bw.cpp $(SRCDIR)/cpu_bw.h
	$(CXX) $(CXXFLAGS) -c -o $@ $<

$(TARGET): $(CU_SRCS) $(SRCDIR)/common.cuh cpu_bw.o
	$(NVCC) $(CFLAGS) -o $@ $(CU_SRCS) cpu_bw.o

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET) cpu_bw.o results.csv

.PHONY: all clean run
