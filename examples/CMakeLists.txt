set(CMAKE_CXX_STANDARD 20)

set(EXAMPLE_SOURCES
    graph_demo.cpp
    basic_ops.cpp
    gradient_validation.cpp
    mlp_demo.cpp
    mnist_mlp.cpp
    mnist_tensor_autograd.cpp
)

foreach(example_src IN LISTS EXAMPLE_SOURCES)
    get_filename_component(example_name "${example_src}" NAME_WE)
    add_executable("${example_name}" "${example_src}")
    target_include_directories("${example_name}" PRIVATE ${PROJECT_SOURCE_DIR}/include)
endforeach()
