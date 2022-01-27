#include <stdio.h>
#include "tester.h"

#define ANSI_COLOR_RED      "\x1b[31m"
#define ANSI_COLOR_GREEN    "\x1b[32m"
#define ANSI_COLOR_YELLOW   "\x1b[33m"
#define ANSI_COLOR_BLUE     "\x1b[34m"
#define ANSI_COLOR_MAGENTA  "\x1b[35m"
#define ANSI_COLOR_CYAN     "\x1b[36m"
#define ANSI_COLOR_RESET    "\x1b[0m"

void trace_start(const char *test_name) {
    printf(ANSI_COLOR_GREEN "[START] %s\n" ANSI_COLOR_RESET, test_name);
}

void trace_end(const char *test_name) {
    printf(ANSI_COLOR_GREEN "[END] %s\n\n" ANSI_COLOR_RESET, test_name);
}

void test(const char *test_name, void (*test_func)()) {
    trace_start(test_name);
    test_func();
    trace_end(test_name);
}
