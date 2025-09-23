CC = gcc
CFLAGS = -std=c99 -Wall -Wpedantic -fopenmp -I../include
LDFLAGS = -lm

R = 5
CFLAGS += -DR=$(R)

TARGET = main_omp

all: $(TARGET)

$(TARGET): main_omp.c
	$(CC) $(CFLAGS) -o $(TARGET) main_omp.c $(LDFLAGS)

clean:
	rm -f $(TARGET)
