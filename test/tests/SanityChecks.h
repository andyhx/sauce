#include <cppunit/extensions/HelperMacros.h>
#include <iostream>
#include <opencv2/opencv.hpp>

class SanityChecks: public CppUnit::TestFixture {

  CPPUNIT_TEST_SUITE(SanityChecks);
  CPPUNIT_TEST(testOpenCv);
  CPPUNIT_TEST(testTuples);
  CPPUNIT_TEST_SUITE_END();

  public:
    void setUp();
    void tearDown();
    void testOpenCv();
    void testTuples();
};
