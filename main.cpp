#include <iostream>
#include <filesystem>
#include <vector>

#include "packages/BlooGraphics/Vulkan/GraphicsVK.h"
#include "packages/BlooGraphics/Graphics.h"

#include "opencv2/opencv.hpp"
#include <glm/vec4.hpp>

using namespace Bloo::Graphics;
using namespace Bloo::Graphics::Vulkan;

bool endsWith(const std::string& str, const std::string& suffix) {
    if (str.length() < suffix.length()) {
        return false;
    }
    return str.compare(str.length() - suffix.length(), suffix.length(), suffix) == 0;
}

std::vector<std::string> filterStrings(const std::vector<std::string>& strings, const std::string& suffix) {
    std::vector<std::string> filteredStrings;
    for (const auto& str : strings) {
        if (endsWith(str, suffix)) {
            filteredStrings.push_back(str);
        }
    }
    return filteredStrings;
}

void readPixels(uint8_t* data, uint32_t count, glm::vec4* pixels){
    for (int k = 0; k < count; k++) {
        pixels[k] = glm::vec4(data[k * 3], data[k * 3 + 1], data[k * 3 + 2], 1) / 255.0f;
    }
}

std::vector<std::string> readFiles(const std::string& path){
    std::vector<std::string> paths;
    for (const auto& entry : std::filesystem::directory_iterator(path)) {
        if (std::filesystem::is_regular_file(entry)) {
            paths.emplace_back(entry.path());
        }
    }
    return paths;
}

cv::Mat createImageFromVec4Array(const glm::vec4* pixelData, int width, int height) {
    // Create a CV_8UC4 Mat object with the given dimensions
    cv::Mat image(height, width, CV_32FC3);

    // Iterate over each pixel in the array and assign its values to the Mat object
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            // Get the index of the current pixel in the array
            int index = y * width + x;

            // Get the pixel value from the array
            glm::vec4 pixel = pixelData[index];

            // Assign the pixel value to the corresponding location in the Mat object
            image.at<cv::Vec3f>(y, x) = cv::Vec3f(pixel.r, pixel.g, pixel.b);
        }
    }

    return image;
}

struct ShaderConstants{
    int current;
    int imCount;
    int width;
    int height;
};

void khanDeghost(uint32_t x, uint32_t y, std::vector<uint8_t*> images, std::vector<float> times, std::vector<float> response){
    IGraphics* gfx = new GraphicsVK();

    auto imCount = (uint8_t)images.size();
    auto pixelCount = x * y;
    std::vector<glm::vec4*> pixels;
    std::vector<TextureHandle> inputImages;
    std::vector<TextureHandle> weightImages;

    TextureCreateOptions imageCreateOptions{
        .width = x,
        .height = y,
        .format = TextureFormat::R32G32B32A32_FLOAT,
        .dim = 2,
        .depth = 1,
        .layers = 1,
        .enableWrite = true
    };
    TextureCreateOptions weightCreateOptions{
        .width = x,
        .height = y,
        .format = TextureFormat::R32_FLOAT,
        .dim = 2,
        .depth = 1,
        .layers = 1,
        .enableWrite = true
    };

    // create initial textures
    for (int i = 0; i < imCount; ++i) {
        auto* px = new glm::vec4[pixelCount];
        readPixels(images[i], pixelCount, px);
        auto tex = gfx->CreateTexture(imageCreateOptions);
        gfx->WriteTexture(tex, px);
        auto weight = gfx->CreateTexture(weightCreateOptions);
        inputImages.emplace_back(tex);
        weightImages.emplace_back(weight);
        pixels.emplace_back(px);
    }
    // calculate initial weights
    auto shaderInitialWeights = gfx->CreateShader("shaders/initial_weights.spv", ShaderType::COMPUTE);

    for (int i = 0; i < imCount; ++i) {
        gfx->SetTexture(shaderInitialWeights, inputImages[i], 0);
        gfx->SetTexture(shaderInitialWeights, weightImages[i], 1);
        gfx->DispatchShader(shaderInitialWeights, "main", x / 8, y / 8, 1);
    }

    auto imageArray = gfx->CreateTexture({
        .width = x,
        .height = y,
        .format = TextureFormat::R32G32B32A32_FLOAT,
        .dim = 2,
        .depth = 1,
        .layers = imCount,
        .enableWrite = true
    });

    auto weightArray = gfx->CreateTexture({
        .width = x,
        .height = y,
        .format = TextureFormat::R32_FLOAT,
        .dim = 2,
        .depth = 1,
        .layers = imCount,
        .enableWrite = true
    });

    for (int i = 0; i < imCount; ++i) {
        gfx->CopyTexture(inputImages[i], imageArray, 0, i);
        gfx->CopyTexture(weightImages[i], weightArray,0,i);
    }

    // iterate weights
    auto shaderIterateWeights = gfx->CreateShader("shaders/iterate_weights.spv", ShaderType::COMPUTE);
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < imCount; ++j) {
            ShaderConstants constants{
                    .current = j,
                    .imCount = imCount,
                    .width = (int)x,
                    .height = (int)y
            };
            gfx->SetTexture(shaderIterateWeights, imageArray, 0);
            gfx->SetTexture(shaderIterateWeights, weightArray, 1);
            gfx->SetTexture(shaderIterateWeights, weightImages[j], 2);
            gfx->SetConstants(shaderIterateWeights, &constants, sizeof(constants));
            gfx->DispatchShader(shaderIterateWeights, "main", x / 4, y / 4, 1);
        }
        for (int j = 0; j < imCount; ++j) {
            gfx->CopyTexture(weightImages[j], weightArray,0,j);
        }
    }
    for (int j = 0; j < imCount; ++j) {
        gfx->CopyTexture(weightImages[j], weightArray,0,j);
    }

    // merge hdr
    auto timesBuffer = gfx->CreateBuffer(imCount, sizeof(float), BufferType::STORAGE);
    auto responseBuffer = gfx->CreateBuffer(256 * 3, sizeof(float), BufferType::STORAGE);
    gfx->WriteBuffer(timesBuffer, times.data());
    gfx->WriteBuffer(responseBuffer, response.data());

    auto resultImage = gfx->CreateTexture({
                                                  .width = x,
                                                  .height = y,
                                                  .format = TextureFormat::R32G32B32A32_FLOAT,
                                                  .dim = 2,
                                                  .depth = 1,
                                                  .layers = 1,
                                                  .enableWrite = true
                                          });

    auto shaderMergeHDR = gfx->CreateShader("shaders/merge_hdr.spv", ShaderType::COMPUTE);
    gfx->SetTexture(shaderMergeHDR, imageArray, 0);
    gfx->SetTexture(shaderMergeHDR, weightArray, 1);
    gfx->SetTexture(shaderMergeHDR, resultImage, 2);
    gfx->SetBuffer(shaderMergeHDR, timesBuffer, 3);
    gfx->SetBuffer(shaderMergeHDR, responseBuffer, 4);
    gfx->DispatchShader(shaderMergeHDR, "main", x / 8, y / 8, 1);

    auto resultPixels = new glm::vec4[pixelCount];
    gfx->ReadTexture(resultImage, resultPixels);

    auto img = createImageFromVec4Array(resultPixels, x, y);
    cv::imwrite("result.hdr", img);

    // cleanup
    delete gfx;
    for (int i = 0; i < imCount; ++i) {
        delete[] pixels[i];
    }
}


int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Path is required!" << std::endl;
        return 1;
    }

    std::string path = argv[1];

    auto files = readFiles(path);
    std::sort(files.begin(), files.end());
    auto imgFiles = filterStrings(files, ".tiff");
    std::vector<uint8_t*> data;
    std::vector<float> times;
    std::vector<cv::Mat> images;
    for (int i = 0; i < imgFiles.size(); ++i) {
        cv::Mat img = cv::imread(imgFiles[i]);
        images.emplace_back(img);
        data.push_back(img.data);
        times.push_back(std::powf(2, i) / 256);
    }
    std::vector<float> responses;
    cv::Mat response;
    auto dbv = cv::createCalibrateDebevec();
    dbv->process(images, response, times);
    for (int i = 0; i < response.rows; ++i) {
        auto r = response.at<cv::Vec3f>(i);
        for (int j = 0; j < 3; ++j) {
            responses.push_back(r[j]);
        }
    }
    khanDeghost(images[0].cols, images[0].rows, data, times, responses);
    return 0;
}
