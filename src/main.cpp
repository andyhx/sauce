#include "global.h"
#include "hog.h"
#include "controller.h"
#include "descriptor.h"

int main(int argc, char** argv) {


  if(argc <= 1) {
    Controller::show_usage();
    return 0;
  }
  if(strcmp("generate", argv[1]) == 0) {
    char* dir = argv[2];
    char output[1024],width[1024],height[1024],hstride[1024],vstride[1024];
    int c;
    while ( (c = getopt(argc, argv, "o:w:h:x:y:")) != -1) {
      switch(c) {
        case 'o':
          strcpy(output, optarg);
          break;
        case 'w':
          strcpy(width, optarg);
          break;
        case 'h':
          strcpy(height, optarg);
          break;
        case 'x':
          strcpy(hstride, optarg);
          break;
        case 'y':
          strcpy(vstride, optarg);
          break;
      }
    }
    Controller::generate(dir, output, atoi(width), atoi(height), atoi(hstride), atoi(vstride));
  }
  else if(strcmp("extract", argv[1]) == 0) {
    char* dir = argv[2];
    char buf[1024];
    int c;
    while ( (c = getopt(argc, argv, "o:")) != -1) {
      switch(c) {
        case 'o':
          strcpy(buf, optarg);
          break;
      }
    }
    HOG *hog = new HOG();
    Controller::extract(hog, dir , buf);
  }
  else if(strcmp("train", argv[1]) == 0) {
    char pos[1024], neg[1024], output[1024];
    int c;
    while ( (c = getopt(argc, argv, "p:n:o:")) != -1) {
      switch(c) {
        case 'p':
          strcpy(pos, optarg);
          break;
        case 'n':
          strcpy(neg, optarg);
          break;
        case 'o':
          strcpy(output, optarg);
          break;
      }
    }
    Controller::train(pos, neg, output);
  }
  else if(strcmp("test", argv[1]) == 0) {
    char svm[1024],set[1024];
    int c;
    while ( (c = getopt(argc, argv, "s:c:")) != -1) {
      switch(c) {
        case 's':
          strcpy(set, optarg);
          break;
        case 'c':
          strcpy(svm, optarg);
          break;
      }
    }
    HOG* hog = new HOG();
    Controller::predict(hog, set, svm);
  }
  else {
    Controller::show_usage();
  }
  return 0;
}


