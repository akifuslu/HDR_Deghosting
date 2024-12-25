#version 450

// Declare the image2DArray bindings
layout(set = 0, binding = 0, rgba32f) uniform readonly image2DArray images;
layout(set = 0, binding = 1, r32f) uniform readonly image2DArray weights;
layout(set = 0, binding = 2, rgba32f) uniform writeonly image2D result;
layout(std430, binding = 3) buffer TimesBuffer {
    float times[];
};
layout(std430, binding = 4) buffer ResponseBuffer {
    float response[];
};

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;
void main() {
    ivec2 id = ivec2(gl_GlobalInvocationID.xy);

    vec3 s = vec3(0.0);
    vec3 t = vec3(0.0);
    vec4 r = vec4(0.0, 0.0, 0.0, 1.0);

    for (int k = 0; k < 9; ++k) {
        float wei = imageLoad(weights, ivec3(id, k)).r;
        vec3 cl = vec3(imageLoad(images, ivec3(id, k)));

        int z = int(cl.r * 255.0);
        s.r += wei * (response[z * 3] - log2(times[k]));
        t.r += wei;

        z = int(cl.g * 255.0);
        s.g += wei * (response[z * 3 + 1] - log2(times[k]));
        t.g += wei;

        z = int(cl.b * 255.0);
        s.b += wei * (response[z * 3 + 2] - log2(times[k]));
        t.b += wei;
    }

    r.r = s.r / max(t.r, 1.0);
    r.g = s.g / max(t.g, 1.0);
    r.b = s.b / max(t.b, 1.0);

    imageStore(result, id, r);
}
