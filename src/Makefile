all:	compile
  
hog.o:	hog.cpp
	gcc -c -std=c++11 hog.cpp -o hog.o

compile:	hog.o
	gcc hog.o -lopencv_core -lopencv_imgproc -lopencv_highgui -lstdc++ -lm -o hog
