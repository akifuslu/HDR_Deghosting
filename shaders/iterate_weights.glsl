#version 450

#define PI 3.14159265359f

layout(set = 0, binding = 0, rgba32f) uniform readonly image2DArray images;
layout(set = 0, binding = 1, r32f) uniform readonly image2DArray weights;
layout(set = 0, binding = 2, r32f) uniform writeonly image2D weight;

layout(push_constant) uniform PushConstants {
    int current;
    int imCount;
    int width;
    int height;
} constants;


vec3 rgb2xyz(vec3 c) {
    vec3 tmp;
    tmp.x = ( c.r > 0.04045 ) ? pow( ( c.r + 0.055 ) / 1.055, 2.4 ) : c.r / 12.92;
    tmp.y = ( c.g > 0.04045 ) ? pow( ( c.g + 0.055 ) / 1.055, 2.4 ) : c.g / 12.92;
    tmp.z = ( c.b > 0.04045 ) ? pow( ( c.b + 0.055 ) / 1.055, 2.4 ) : c.b / 12.92;
    const mat3 mat = mat3(
        0.4124, 0.3576, 0.1805,
        0.2126, 0.7152, 0.0722,
        0.0193, 0.1192, 0.9505
    );
    return 100.0 * (mat * tmp);
}

vec3 xyz2lab(vec3 c) {
    vec3 n = c / vec3(95.047, 100, 108.883);
    vec3 v;
    v.x = ( n.x > 0.008856 ) ? pow( n.x, 1.0 / 3.0 ) : ( 7.787 * n.x ) + ( 16.0 / 116.0 );
    v.y = ( n.y > 0.008856 ) ? pow( n.y, 1.0 / 3.0 ) : ( 7.787 * n.y ) + ( 16.0 / 116.0 );
    v.z = ( n.z > 0.008856 ) ? pow( n.z, 1.0 / 3.0 ) : ( 7.787 * n.z ) + ( 16.0 / 116.0 );
    return vec3(( 116.0 * v.y ) - 16.0, 500.0 * ( v.x - v.y ), 200.0 * ( v.y - v.z ));
}

vec3 rgb2lab(vec3 c)
{
    vec3 lab = xyz2lab(rgb2xyz(c));
    return vec3( lab.x / 100.0, 0.5 + 0.5 * ( lab.y / 127.0 ), 0.5 + 0.5 * ( lab.z / 127.0 ));
}

int nei3[8][2] = {
{-1, -1}, {-1, 0}, {-1, 1},
{ 0, -1},          { 0, 1},
{ 1, -1}, { 1, 0}, { 1, 1}
};

float KH(float X[5]) {
    float dot = 0;
    dot += 1 * X[0] * X[0];
    dot += 1 * X[1] * X[1];
    dot += 1 * X[2] * X[2];
    dot += 1 * X[3] * X[3];
    dot += 1 * X[4] * X[4];
    return exp(-0.5 * dot);// / sqrt(pow(2 * PI, 1));
}

float P(int x, int y, int k) {
    float nom = 0;
    float denom = 0;
    vec3 p = rgb2lab(imageLoad(images, ivec3(x, y, k)).rgb);
    for (int s = 0; s < constants.imCount; s++) {
        for (int n = 0; n < 8; n++) {
            int x1 = x + nei3[n][0];
            int y1 = y + nei3[n][1];
            x1 = clamp(x1, 0, constants.width - 1);
            y1 = clamp(y1, 0, constants.height - 1);
            float wei = imageLoad(weights, ivec3(x1, y1, s)).r;
            vec3 p1 = rgb2lab(imageLoad(images, ivec3(x1, y1, s)).rgb);
            float X[5] = {x - x1, y - y1, (p.x - p1.x) * 1, (p.y - p1.y) * 1, (p.z - p1.z) * 1};
            denom += wei;
            nom += wei * KH(X);
        }
    }
    return nom / (denom != 0.0 ? denom : 1.0f);
}

layout(local_size_x = 4, local_size_y = 4, local_size_z = 1) in;
void main() {
    ivec2 id = ivec2(gl_GlobalInvocationID.xy);
    float wei = imageLoad(weights, ivec3(id, constants.current)).r;
    float iterated = clamp(wei * P(id.x, id.y, constants.current), 0, 1);
    imageStore(weight, id, vec4(iterated));
}
