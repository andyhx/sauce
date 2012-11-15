.PHONY: all compile run clean
CC=g++
LFLAGS=-lcppunit -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_ml -lstdc++ -lm
CFLAGS=-std=c++11
SOURCES=$(wildcard *.cpp)
OBJECTS=$(SOURCES:.cpp=.o)

all:	compile	

compile: $(OBJECTS)
	$(CC) $(OBJECTS) -o sauce $(LFLAGS)

%.o:	%.cpp
	$(CC) -c $(CFLAGS) $< -o $@
