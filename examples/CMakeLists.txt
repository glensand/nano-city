set(BINARY nngd-examples)

set(CMAKE_CXX_STANDARD 20)

file(GLOB EXAMPLE_SOURCES
    *.cpp
)

add_executable(${BINARY} ${EXAMPLE_SOURCES})
target_include_directories(${BINARY} PRIVATE ${PROJECT_SOURCE_DIR}/include)
