NVCC := nvcc
CXX := g++
CFLAGS := -std=c++17 -arch=sm_121 -O3 --extended-lambda -lineinfo -Iparlaylib/include
CXXFLAGS := -std=c++17 -O3 -march=native
TARGET := bench

all: $(TARGET)

cpu_bw.o: cpu_bw.cpp cpu_bw.h
	$(CXX) $(CXXFLAGS) -c -o $@ $<

$(TARGET): bench.cu cpu_bw.o
	$(NVCC) $(CFLAGS) -o $@ bench.cu cpu_bw.o

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET) cpu_bw.o results.csv

.PHONY: all clean run
