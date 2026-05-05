# CMake generated Testfile for 
# Source directory: /home/runner/work/ocrwkv.cpp/ocrwkv.cpp
# Build directory: /home/runner/work/ocrwkv.cpp/ocrwkv.cpp
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test([=[opencog_ml_tests]=] "/home/runner/work/ocrwkv.cpp/ocrwkv.cpp/bin/test_opencog_ml" "/home/runner/work/ocrwkv.cpp/ocrwkv.cpp/tests/tiny-rwkv-4v0-660K-FP32.bin")
set_tests_properties([=[opencog_ml_tests]=] PROPERTIES  _BACKTRACE_TRIPLES "/home/runner/work/ocrwkv.cpp/ocrwkv.cpp/CMakeLists.txt;182;add_test;/home/runner/work/ocrwkv.cpp/ocrwkv.cpp/CMakeLists.txt;0;")
subdirs("tests")
subdirs("extras")
subdirs("ggml")
