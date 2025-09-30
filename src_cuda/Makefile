NVCC = nvcc

R = 5
NVFLAGS = -DR=$(R) -I../include

TARGET = main_cuda

all: $(TARGET)

$(TARGET): main_cuda.cu
	$(NVCC) $(NVFLAGS) -o $(TARGET) main_cuda.cu

clean:
	rm -f $(TARGET)
