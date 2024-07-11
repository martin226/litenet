CC=g++
CFLAGS=-I.
DEPS = activations.h layers.h loss.h matrix.h model.h initializers.h
OBJ = activations.o layers.o loss.o matrix.o model.o initializers.o example_mnist.o

%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

example_mnist: $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS)
	./example_mnist