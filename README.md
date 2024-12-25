# Implementation of "Ghost Removal in High Dynamic Range Images"

The original implementation of the paper, Ghost Removal in High Dynamic Range Images, was CPU-based. However, due to the performance-intensive and highly parallelizable nature of the task, I have reimplemented the algorithm in C++ using Vulkan and compute shaders. This approach leverages GPU acceleration to significantly enhance performance, making it well-suited for processing high-resolution images efficiently.
