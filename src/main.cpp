#include "global.h"
#include "hog.h"
#include "cov.h"
#include "controller.h"
#include "descriptor.h"

int main(int argc, char** argv) {

  auto pick_descriptor = [](char* method) -> Descriptor* {
    Descriptor* desc;
    if(strcmp("hog", method) == 0) {
      desc = new HOG();
    }
    else {
      desc = new Cov();
    }
    return desc;
  };

  if(argc <= 1) {
    Controller::show_usage();
    return 0;
  }
  if(strcmp("fp", argv[1]) == 0) {
    char input[1024], method[1024], classifier[1024], output[1024];
    int c;
    while ( (c = getopt(argc, argv, "i:m:c:o:")) != -1) {
      switch(c) {
        case 'i':
          strcpy(input, optarg);
          break;
        case 'm':
          strcpy(method, optarg);
          break;
        case 'c':
          strcpy(classifier, optarg);
          break;
        case 'o':
          strcpy(output, optarg);
          break;
      }
    }

    Descriptor* desc = pick_descriptor(method); 
    Controller::false_positives(desc, classifier, input, output);
  }
  else if(strcmp("detect", argv[1]) == 0) {
    char input[1024], annotations[1024], method[1024], classifier[1024];
    int c;
    while ( (c = getopt(argc, argv, "i:a:m:c:")) != -1) {
      switch(c) {
        case 'i':
          strcpy(input, optarg);
          break;
        case 'a':
          strcpy(annotations, optarg);
          break;
        case 'm':
          strcpy(method, optarg);
          break;
        case 'c':
          strcpy(classifier, optarg);
          break;
      }
    }

    Descriptor* desc = pick_descriptor(method); 
    Controller::detect(desc, classifier, input, annotations);
  }
  else if(strcmp("generate", argv[1]) == 0) {
    char* dir = argv[2];
    char output[1024],width[1024],height[1024],hstride[1024],vstride[1024],n[1024];
    int c;
    while ( (c = getopt(argc, argv, "o:w:h:x:y:n:")) != -1) {
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
        case 'n':
          strcpy(n, optarg);
          break;
      }
    }
    Controller::generate(dir, output, atoi(width), atoi(height), atoi(hstride), atoi(vstride), atoi(n));
  }
  else if(strcmp("extract", argv[1]) == 0) {
    char* dir = argv[2];
    char buf[1024], method[1024], probability[1024];
    int c;
    while ( (c = getopt(argc, argv, "o:m:p:")) != -1) {
      switch(c) {
        case 'o':
          strcpy(buf, optarg);
          break;
        case 'm':
          strcpy(method, optarg);
          break;
        case 'p':
          strcpy(probability, optarg);
          break;
      }
    }

    float p = (float)atoi(probability)/100;
    Descriptor *desc = pick_descriptor(method);
    Controller::extract(desc, dir , buf, p);
  }
  else if(strcmp("join_sets", argv[1]) == 0) {
    char o[1024], a[1024], b[1024];
    int c;
    while ( (c = getopt(argc, argv, "o:a:b:")) != -1) {
      switch(c) {
        case 'o':
          strcpy(o, optarg);
          break;
        case 'a':
          strcpy(a, optarg);
          break;
        case 'b':
          strcpy(b, optarg);
          break;
      }
    }

    Controller::join_sets(a, b, o);
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
    char svm[1024],set[1024], method[1024];
    int c;
    while ( (c = getopt(argc, argv, "s:c:m:")) != -1) {
      switch(c) {
        case 's':
          strcpy(set, optarg);
          break;
        case 'c':
          strcpy(svm, optarg);
          break;
        case 'm':
          strcpy(method, optarg);
          break;
      }
    }

    Descriptor* desc = pick_descriptor(method);
    Controller::predict(desc, set, svm);
  }
  else {
    Controller::show_usage();
  }
  return 0;
}


