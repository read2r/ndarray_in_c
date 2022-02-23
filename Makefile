CC = gcc
FLAG = -lm -g
OBJS = test_ndarray.o ndarray.o ndshape.o tester.o
TARGET = test_ndarray.out

all : $(OBJS)
	$(CC) $(OBJS) -o $(TARGET) $(FLAG)

%.o : %.c %.h
	$(CC) -c $< $(FLAG)

.PHONY: clean
clean:
	rm -f $(OBJS) $(TARGET)
