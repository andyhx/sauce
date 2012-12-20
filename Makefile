.PHONY: all compile clean
CC=g++
LFLAGS=-lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_ml -lstdc++ -lm
CFLAGS=-std=c++0x
SOURCES=$(wildcard src/*.cpp)
OBJECTS=$(SOURCES:.cpp=.o)

all:	compile	

compile: $(OBJECTS)
	$(CC) $(OBJECTS) -o sauce $(LFLAGS)

%.o:	%.cpp
	$(CC) -c $(CFLAGS) $< -o $@

clean:	
	rm src/*.o
