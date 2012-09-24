.PHONY: all compile run clean
CC=g++
LFLAGS=-lcppunit -lopencv_core
SOURCES=$(wildcard tests/*.cpp)
OBJECTS=$(SOURCES:.cpp=.o)

all:	run	

compile: $(OBJECTS)
	$(CC) $(OBJECTS) -o run_tests $(LFLAGS)

%.o:	%.cpp
	$(CC) -c $< -o $@

run:	compile
	./run_tests

clean:
	rm -rf tests/*.o run_tests
