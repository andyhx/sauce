#include <cppunit/ui/text/TestRunner.h>
#include "SanityChecks.h"

int main(int argc, char **argv) {
    CppUnit::TextUi::TestRunner runner;
    runner.addTest(SanityChecks::suite());
    runner.run();
    return 0;
}
