#version 450

layout(set = 0, binding = 0, rgba32f) uniform readonly image2D image;
layout(set = 0, binding = 1, r32f) uniform writeonly image2D weight;

float Weight(float z){
    return 1 - pow(abs(2 * z - 1), 12);
}

float CalculateWeight(vec3 pixel) {
    return (Weight(pixel.x) + Weight(pixel.y) + Weight(pixel.z)) / 3.0f;
}

layout (local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
void main() {
    ivec2 id = ivec2(gl_GlobalInvocationID.xy);
    vec3 inputValue = vec3(imageLoad(image, id));
    imageStore(weight, id, vec4(CalculateWeight(inputValue)));
}