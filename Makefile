# Makefile

# Compiler and flags
CC = gcc
CFLAGS = -Wall -fPIC -I.

# Target shared library name
TARGET_LIB = libmylib.so

# Source files and object files
SRCS = src/main.c
OBJS = $(SRCS:.c=.o)

# Default target to build everything
all: $(TARGET_LIB)

# Rule to create the shared library from object files
$(TARGET_LIB): $(OBJS)
	$(CC) -shared -o $@ $^

# Rule to create object files from source files
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Clean up generated files
clean:
	rm -f $(OBJS) $(TARGET_LIB)

# Phony targets to avoid conflicts with files named 'clean', 'all', etc.
.PHONY: all clean
