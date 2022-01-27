#ifndef __TESTER_H__
#define __TESTER_H__

void trace_start(const char *test_name);
void trace_end(const char *test_name);
void test(const char *test_name, void (*test_func)());

#endif
