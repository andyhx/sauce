#include "global.h"
#include "hog.h"
#include "cov.h"
#include "lbp.h"
#include "controller.h"
#include "descriptor.h"

int main(int argc, char** argv) {

  auto pick_descriptor = [](char* method, char* b, char* k, char* n) -> Descriptor* {
    Descriptor* desc;
    if(strcmp("hog", method) == 0) {
      desc = new HOG();
      HOG::blockWidth = atoi(b);
      HOG::cellWidth = atoi(k);
      HOG::bins = atoi(b);
    }
    else if(strcmp("lbp", method) == 0) {
      desc = new LBP();
    }
    else {
      desc = new Cov();
      Cov::blockWidth = atoi(b);
      Cov::hStride = atoi(b);
      Cov::vStride = atoi(b);
    }
    return desc;
  };

  if(argc <= 1) {
    Controller::show_usage();
    return 0;
  }
  if(strcmp("fp", argv[1]) == 0) {
    char input[1024], classifier[1024], output[1024], width[1024], height[1024];
    char method[1024], b[1024], k[1024], n[1024];
    int c;
    while ( (c = getopt(argc, argv, "i:c:o:w:h:m:b:k:n:")) != -1) {
      switch(c) {
        case 'i':
          strcpy(input, optarg);
          break;
        case 'c':
          strcpy(classifier, optarg);
          break;
        case 'o':
          strcpy(output, optarg);
          break;
        case 'w':
          strcpy(width, optarg);
          break;
        case 'h':
          strcpy(height, optarg);
          break;
        case 'm':
          strcpy(method, optarg);
          break;
        case 'b':
          strcpy(b, optarg);
          break;
        case 'k':
          strcpy(k, optarg);
          break;
        case 'n':
          strcpy(n, optarg);
          break;
      }
    }

    Descriptor* desc = pick_descriptor(method, b, k, n); 
    Controller::false_positives(desc, classifier, input, output, atoi(width), atoi(height), atoi(width), atoi(height));
  }
  else if(strcmp("detect", argv[1]) == 0) {
    char input[1024], annotations[1024], classifier[1024], output[1024], width[1024], height[1024], x[1024], y[1024];
    char method[1024], b[1024], k[1024], n[1024];
    int c;
    while ( (c = getopt(argc, argv, "i:a:c:o:w:h:x:y:m:b:k:n:")) != -1) {
      switch(c) {
        case 'i':
          strcpy(input, optarg);
          break;
        case 'a':
          strcpy(annotations, optarg);
          break;
        case 'c':
          strcpy(classifier, optarg);
          break;
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
          strcpy(x, optarg);
          break;
        case 'y':
          strcpy(y, optarg);
          break;
        case 'm':
          strcpy(method, optarg);
          break;
        case 'b':
          strcpy(b, optarg);
          break;
        case 'k':
          strcpy(k, optarg);
          break;
        case 'n':
          strcpy(n, optarg);
          break;
      }
    }

    Descriptor* desc = pick_descriptor(method, b, k, n); 
    Controller::detect(desc, classifier, input, annotations, output, atoi(width), atoi(height), atoi(x), atoi(y));
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
    char buf[1024], probability[1024];
    char method[1024], b[1024], k[1024], n[1024];
    int c;
    while ( (c = getopt(argc, argv, "o:p:m:b:k:n:")) != -1) {
      switch(c) {
        case 'o':
          strcpy(buf, optarg);
          break;
        case 'p':
          strcpy(probability, optarg);
          break;
        case 'm':
          strcpy(method, optarg);
          break;
        case 'b':
          strcpy(b, optarg);
          break;
        case 'k':
          strcpy(k, optarg);
          break;
        case 'n':
          strcpy(n, optarg);
          break;
      }
    }

    float p = (float)atoi(probability)/100;
    Descriptor *desc = pick_descriptor(method, b, k, n);
    Controller::extract(desc, dir , buf, p);
  }
  else if(strcmp("join_sets", argv[1]) == 0) {
    char o[1024], a[1024], b[1024];
    int c;
    while ( (c = getopt(argc, argv, "o:a:")) != -1) {
      switch(c) {
        case 'o':
          strcpy(o, optarg);
          break;
        case 'a':
          strcpy(a, optarg);
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
    char svm[1024],set[1024];
    char method[1024], b[1024], k[1024], n[1024];
    int c;
    while ( (c = getopt(argc, argv, "s:c:m:b:k:n:")) != -1) {
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
        case 'b':
          strcpy(b, optarg);
          break;
        case 'k':
          strcpy(k, optarg);
          break;
        case 'n':
          strcpy(n, optarg);
          break;
      }
    }

    Descriptor* desc = pick_descriptor(method, b, k, n);
    RESULT res = Controller::predict(desc, set, svm);

    cout << ELEMENT(0, res) << endl;
    cout << ELEMENT(1, res) << endl;
    cout << ELEMENT(2, res) << endl;
  }
  else {
    Controller::show_usage();
  }
  return 0;
}


