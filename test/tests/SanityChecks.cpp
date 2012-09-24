#include "SanityChecks.h"

using namespace std;
using namespace cv;

void SanityChecks::setUp() {
}

void SanityChecks::tearDown() {
}

void SanityChecks::testOpenCv() {
    Mat matrix = (Mat_<double>(3, 3) << 5, 6, 7, 8, 9, -5, -7, -5, 100);
    Mat i = Mat::eye(3, 3, CV_64FC1);
    Mat diff = matrix*i != matrix;
    CPPUNIT_ASSERT(countNonZero(diff) == 0);
}

void SanityChecks::testTuples() {
}
