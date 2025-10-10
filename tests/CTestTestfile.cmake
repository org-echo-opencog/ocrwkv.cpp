# CMake generated Testfile for 
# Source directory: /home/runner/work/ocrwkv.cpp/ocrwkv.cpp/tests
# Build directory: /home/runner/work/ocrwkv.cpp/ocrwkv.cpp/tests
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test([=[test_ggml_basics]=] "/home/runner/work/ocrwkv.cpp/ocrwkv.cpp/bin/test_ggml_basics")
set_tests_properties([=[test_ggml_basics]=] PROPERTIES  _BACKTRACE_TRIPLES "/home/runner/work/ocrwkv.cpp/ocrwkv.cpp/tests/CMakeLists.txt;9;add_test;/home/runner/work/ocrwkv.cpp/ocrwkv.cpp/tests/CMakeLists.txt;51;rwkv_add_test;/home/runner/work/ocrwkv.cpp/ocrwkv.cpp/tests/CMakeLists.txt;0;")
add_test([=[test_quantized_matmul_on_gpu]=] "/home/runner/work/ocrwkv.cpp/ocrwkv.cpp/bin/test_quantized_matmul_on_gpu")
set_tests_properties([=[test_quantized_matmul_on_gpu]=] PROPERTIES  _BACKTRACE_TRIPLES "/home/runner/work/ocrwkv.cpp/ocrwkv.cpp/tests/CMakeLists.txt;9;add_test;/home/runner/work/ocrwkv.cpp/ocrwkv.cpp/tests/CMakeLists.txt;52;rwkv_add_test;/home/runner/work/ocrwkv.cpp/ocrwkv.cpp/tests/CMakeLists.txt;0;")
add_test([=[test_tiny_rwkv]=] "/home/runner/work/ocrwkv.cpp/ocrwkv.cpp/bin/test_tiny_rwkv")
set_tests_properties([=[test_tiny_rwkv]=] PROPERTIES  _BACKTRACE_TRIPLES "/home/runner/work/ocrwkv.cpp/ocrwkv.cpp/tests/CMakeLists.txt;9;add_test;/home/runner/work/ocrwkv.cpp/ocrwkv.cpp/tests/CMakeLists.txt;53;rwkv_add_test;/home/runner/work/ocrwkv.cpp/ocrwkv.cpp/tests/CMakeLists.txt;0;")
add_test([=[test_quantization_format_compatibility]=] "/home/runner/work/ocrwkv.cpp/ocrwkv.cpp/bin/test_quantization_format_compatibility")
set_tests_properties([=[test_quantization_format_compatibility]=] PROPERTIES  _BACKTRACE_TRIPLES "/home/runner/work/ocrwkv.cpp/ocrwkv.cpp/tests/CMakeLists.txt;9;add_test;/home/runner/work/ocrwkv.cpp/ocrwkv.cpp/tests/CMakeLists.txt;54;rwkv_add_test;/home/runner/work/ocrwkv.cpp/ocrwkv.cpp/tests/CMakeLists.txt;0;")
add_test([=[test_logit_calculation_skipping]=] "/home/runner/work/ocrwkv.cpp/ocrwkv.cpp/bin/test_logit_calculation_skipping")
set_tests_properties([=[test_logit_calculation_skipping]=] PROPERTIES  _BACKTRACE_TRIPLES "/home/runner/work/ocrwkv.cpp/ocrwkv.cpp/tests/CMakeLists.txt;9;add_test;/home/runner/work/ocrwkv.cpp/ocrwkv.cpp/tests/CMakeLists.txt;55;rwkv_add_test;/home/runner/work/ocrwkv.cpp/ocrwkv.cpp/tests/CMakeLists.txt;0;")
add_test([=[test_eval_sequence_in_chunks]=] "/home/runner/work/ocrwkv.cpp/ocrwkv.cpp/bin/test_eval_sequence_in_chunks")
set_tests_properties([=[test_eval_sequence_in_chunks]=] PROPERTIES  _BACKTRACE_TRIPLES "/home/runner/work/ocrwkv.cpp/ocrwkv.cpp/tests/CMakeLists.txt;9;add_test;/home/runner/work/ocrwkv.cpp/ocrwkv.cpp/tests/CMakeLists.txt;56;rwkv_add_test;/home/runner/work/ocrwkv.cpp/ocrwkv.cpp/tests/CMakeLists.txt;0;")
add_test([=[test_context_cloning]=] "/home/runner/work/ocrwkv.cpp/ocrwkv.cpp/bin/test_context_cloning")
set_tests_properties([=[test_context_cloning]=] PROPERTIES  _BACKTRACE_TRIPLES "/home/runner/work/ocrwkv.cpp/ocrwkv.cpp/tests/CMakeLists.txt;9;add_test;/home/runner/work/ocrwkv.cpp/ocrwkv.cpp/tests/CMakeLists.txt;57;rwkv_add_test;/home/runner/work/ocrwkv.cpp/ocrwkv.cpp/tests/CMakeLists.txt;0;")
