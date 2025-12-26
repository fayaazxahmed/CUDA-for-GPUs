# Makefile for tensor_cuda project
# Usage:
#   make        # build the test program
#   make run    # run the test executable
#   make clean  # remove build artifacts

CXX = g++
CXXFLAGS = -std=c++11 -O2 -Wall


SRCDIR = .
SOURCES = $(SRCDIR)/tensor.cpp $(SRCDIR)/tensor_test.cpp
HEADERS = $(SRCDIR)/tensor.hpp
OBJECTS = $(SOURCES:.cpp=.o)
TARGET = tensor_test.exe

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJECTS)

%.o: %.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET) *.o
