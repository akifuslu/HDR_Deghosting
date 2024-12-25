cmake_minimum_required(VERSION 3.25)
set(CMAKE_OSX_ARCHITECTURES arm64)
project(Deghosting)

set(CMAKE_CXX_STANDARD 17)

file(COPY ${CMAKE_SOURCE_DIR}/samples DESTINATION ${CMAKE_BINARY_DIR})

find_package(glm CONFIG REQUIRED)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_subdirectory(packages/BlooGraphics)
set(BlooGraphics_INSTALL_DIR "/packages/BlooGraphics")
include_directories("${BlooGraphics_INSTALL_DIR}/include")
link_directories("${BlooGraphics_INSTALL_DIR}/lib")
link_libraries(BlooGraphics)


# Find all shader source files in the shaders folder
file(GLOB_RECURSE SHADER_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/shaders/*.glsl")

# Create a custom command for compiling each shader
foreach(SHADER_SOURCE ${SHADER_SOURCES})
    get_filename_component(SHADER_NAME ${SHADER_SOURCE} NAME_WE)
    set(SHADER_OUTPUT "${CMAKE_BINARY_DIR}/shaders/${SHADER_NAME}.spv")

    add_custom_command(
            OUTPUT ${SHADER_OUTPUT}
            COMMAND glslc -o ${SHADER_OUTPUT} -fshader-stage=compute ${SHADER_SOURCE}
            DEPENDS ${SHADER_SOURCE}
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            COMMENT "Compiling ${SHADER_NAME}"
    )

    list(APPEND SHADER_OUTPUTS ${SHADER_OUTPUT})
endforeach()

# Create a custom target depending on all compiled shaders
add_custom_target(Shaders DEPENDS ${SHADER_OUTPUTS})


add_executable(Deghosting main.cpp)
target_link_libraries(Deghosting ${OpenCV_LIBS} glm::glm)
add_dependencies(Deghosting Shaders)
