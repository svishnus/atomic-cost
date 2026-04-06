NVCC := nvcc
CFLAGS := -std=c++17 -arch=sm_121 -O3 --extended-lambda -lineinfo
TARGET := bench

all: $(TARGET)

$(TARGET): bench.cu
	$(NVCC) $(CFLAGS) -o $@ $<

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET) results.csv

.PHONY: all clean run
