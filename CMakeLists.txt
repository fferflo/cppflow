cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(cppflow LANGUAGES CXX)

include(GNUInstallDirs)
set(PACKAGE_NAME cppflow)
set(INSTALL_CONFIGDIR ${CMAKE_INSTALL_LIBDIR}/${PACKAGE_NAME}/cmake)

set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};${CMAKE_SOURCE_DIR}/cmake/modules")


# Library target
find_package(tensorflow REQUIRED)
add_library(cppflow INTERFACE)
target_include_directories(cppflow
  INTERFACE
    ${tensorflow_INCLUDE_DIRS}
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_compile_features(cppflow INTERFACE cxx_std_14)
target_link_libraries(cppflow INTERFACE
  ${tensorflow_LIBRARIES}
)
install(
  TARGETS cppflow
  EXPORT install_targets
)



# Build examples
add_executable(tensor examples/tensor/main.cpp)
target_link_libraries(tensor PUBLIC cppflow)

add_executable(efficientnet examples/efficientnet/main.cpp)
target_link_libraries(efficientnet PUBLIC cppflow)
target_compile_definitions(efficientnet PUBLIC
  CAT_PATH="${CMAKE_CURRENT_SOURCE_DIR}/examples/efficientnet/my_cat.jpg"
  MODEL_PATH="${CMAKE_CURRENT_BINARY_DIR}/efficientnet/model"
)
set_target_properties(efficientnet PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/efficientnet")
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/examples/efficientnet/create_model.py ${CMAKE_CURRENT_BINARY_DIR}/efficientnet/create_model.py COPYONLY)

find_package(Threads REQUIRED)
add_executable(eager_op_multithread examples/eager_op_multithread/main.cpp)
target_link_libraries(eager_op_multithread PUBLIC cppflow Threads::Threads)



# Install headers
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})


# Install targets file
install(EXPORT install_targets
  FILE
    ${PACKAGE_NAME}Targets.cmake
  NAMESPACE
    ${PACKAGE_NAME}::
  DESTINATION
    ${INSTALL_CONFIGDIR}
)


# Install cppflowConfig.cmake
include(CMakePackageConfigHelpers)
configure_package_config_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/${PACKAGE_NAME}Config.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/${PACKAGE_NAME}Config.cmake
  INSTALL_DESTINATION ${INSTALL_CONFIGDIR}
)
install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/${PACKAGE_NAME}Config.cmake
  DESTINATION ${INSTALL_CONFIGDIR}
)


# Install find modules
install(DIRECTORY cmake/modules/ DESTINATION ${INSTALL_CONFIGDIR}/modules)
