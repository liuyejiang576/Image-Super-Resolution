#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <numeric>
#include <string>
#include <vector>

#include <ncnn/net.h>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "third_party/stb_image.h"
#include "third_party/stb_image_write.h"

#if defined(__ANDROID__)
#include <unistd.h>
#endif

static double median_ms(std::vector<double>& v)
{
    if (v.empty())
        return 0.0;
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    if (n % 2 == 1)
        return v[n / 2];
    return 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

static double percentile_ms(std::vector<double>& v, double p)
{
    if (v.empty())
        return 0.0;
    std::sort(v.begin(), v.end());
    double idx = (p / 100.0) * (v.size() - 1);
    size_t lo = static_cast<size_t>(idx);
    size_t hi = std::min(lo + 1, v.size() - 1);
    double frac = idx - lo;
    return v[lo] * (1.0 - frac) + v[hi] * frac;
}

static long read_vmhwm_kb()
{
#if defined(__ANDROID__) || defined(__linux__)
    std::ifstream f("/proc/self/status");
    std::string line;
    while (std::getline(f, line))
    {
        if (line.rfind("VmHWM:", 0) == 0)
        {
            long kb = 0;
            std::sscanf(line.c_str(), "VmHWM: %ld kB", &kb);
            return kb;
        }
    }
#endif
    return -1;
}

static void usage(const char* argv0)
{
    std::fprintf(stderr,
        "Usage (bench):\n"
        "  %s --param model.param --bin model.bin --input-w W --input-h H "
        "[--warmup 50] [--iters 300] [--fp16] [--vulkan] [--threads N]\n"
        "Usage (infer PNG):\n"
        "  %s --param model.param --bin model.bin --in lr.png --out hr.png "
        "[--input-w W] [--input-h H] [--warmup 1] [--fp16] [--vulkan] [--threads N]\n"
        "  If --input-w/--input-h set, LR is resized to that size before infer.\n"
        "  Prints latency_ms=... and a JSON summary line.\n",
        argv0, argv0);
}

static int load_lr_png(const std::string& path, int target_w, int target_h, ncnn::Mat& out)
{
    int w = 0, h = 0, c = 0;
    unsigned char* pixels = stbi_load(path.c_str(), &w, &h, &c, 3);
    if (!pixels)
    {
        std::fprintf(stderr, "Failed to load PNG: %s\n", path.c_str());
        return -1;
    }

    const int dst_w = target_w > 0 ? target_w : w;
    const int dst_h = target_h > 0 ? target_h : h;

    if (dst_w == w && dst_h == h)
        out = ncnn::Mat::from_pixels(pixels, ncnn::Mat::PIXEL_RGB, w, h);
    else
        out = ncnn::Mat::from_pixels_resize(pixels, ncnn::Mat::PIXEL_RGB, w, h, dst_w, dst_h);

    stbi_image_free(pixels);

    // Training / ONNX export use float RGB in [0, 1].
    const float norm_vals[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
    out.substract_mean_normalize(0, norm_vals);
    return 0;
}

static int save_hr_png(const std::string& path, const ncnn::Mat& sr)
{
    if (sr.empty() || sr.c < 3)
    {
        std::fprintf(stderr, "Invalid SR output mat (c=%d)\n", sr.c);
        return -1;
    }

    // sr is CHW float roughly in [0, 1]; write interleaved RGB8 with clamp.
    const int w = sr.w;
    const int h = sr.h;
    std::vector<unsigned char> pixels(static_cast<size_t>(w) * h * 3);
    const float* r = sr.channel(0);
    const float* g = sr.channel(1);
    const float* b = sr.channel(2);
    for (int i = 0; i < w * h; ++i)
    {
        auto to_u8 = [](float v) -> unsigned char {
            v = std::max(0.f, std::min(1.f, v)) * 255.f + 0.5f;
            return static_cast<unsigned char>(v);
        };
        pixels[static_cast<size_t>(i) * 3 + 0] = to_u8(r[i]);
        pixels[static_cast<size_t>(i) * 3 + 1] = to_u8(g[i]);
        pixels[static_cast<size_t>(i) * 3 + 2] = to_u8(b[i]);
    }

    if (!stbi_write_png(path.c_str(), w, h, 3, pixels.data(), w * 3))
    {
        std::fprintf(stderr, "Failed to write PNG: %s\n", path.c_str());
        return -1;
    }
    return 0;
}

static int run_infer(
    ncnn::Net& net,
    const std::string& in_blob,
    const std::string& out_blob,
    const std::string& in_path,
    const std::string& out_path,
    int input_w,
    int input_h,
    int warmup,
    bool use_fp16,
    bool use_vulkan,
    int threads)
{
    (void)use_fp16;
    (void)use_vulkan;
    (void)threads;

    ncnn::Mat in;
    if (load_lr_png(in_path, input_w, input_h, in) != 0)
        return 1;

    auto run_once = [&](ncnn::Mat& out) -> int {
        ncnn::Extractor ex = net.create_extractor();
        ex.set_light_mode(true);
        if (ex.input(in_blob.c_str(), in) != 0)
            return -1;
        if (ex.extract(out_blob.c_str(), out) != 0)
            return -1;
        return 0;
    };

    for (int i = 0; i < warmup; ++i)
    {
        ncnn::Mat out;
        if (run_once(out) != 0)
        {
            std::fprintf(stderr, "Inference failed during warmup (blobs: %s -> %s)\n",
                in_blob.c_str(), out_blob.c_str());
            return 1;
        }
    }

    ncnn::Mat out;
    auto t0 = std::chrono::steady_clock::now();
    if (run_once(out) != 0)
    {
        std::fprintf(stderr, "Inference failed (blobs: %s -> %s)\n", in_blob.c_str(), out_blob.c_str());
        return 1;
    }
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    if (save_hr_png(out_path, out) != 0)
        return 1;

    std::fprintf(stderr, "latency_ms=%.4f\n", ms);
    std::printf(
        "{"
        "\"mode\":\"infer\","
        "\"in\":\"%s\","
        "\"out\":\"%s\","
        "\"in_blob\":\"%s\","
        "\"out_blob\":\"%s\","
        "\"input_w\":%d,"
        "\"input_h\":%d,"
        "\"output_w\":%d,"
        "\"output_h\":%d,"
        "\"warmup\":%d,"
        "\"fp16\":%s,"
        "\"vulkan\":%s,"
        "\"latency_ms\":%.4f"
        "}\n",
        in_path.c_str(),
        out_path.c_str(),
        in_blob.c_str(),
        out_blob.c_str(),
        in.w,
        in.h,
        out.w,
        out.h,
        warmup,
        use_fp16 ? "true" : "false",
        use_vulkan ? "true" : "false",
        ms);
    return 0;
}

int main(int argc, char** argv)
{
    std::string param_path;
    std::string bin_path;
    std::string in_blob = "in0";
    std::string out_blob = "out0";
    std::string in_path;
    std::string out_path;
    int input_w = 180;
    int input_h = 180;
    int warmup = 50;
    int iters = 300;
    int threads = 4;
    bool use_fp16 = false;
    bool use_vulkan = false;
    bool input_size_set = false;

    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        auto need = [&](const char* name) -> std::string {
            if (i + 1 >= argc)
            {
                std::fprintf(stderr, "Missing value for %s\n", name);
                usage(argv[0]);
                std::exit(2);
            }
            return argv[++i];
        };
        if (arg == "--param")
            param_path = need("--param");
        else if (arg == "--bin")
            bin_path = need("--bin");
        else if (arg == "--in-blob")
            in_blob = need("--in-blob");
        else if (arg == "--out-blob")
            out_blob = need("--out-blob");
        else if (arg == "--in")
            in_path = need("--in");
        else if (arg == "--out")
            out_path = need("--out");
        else if (arg == "--input-w")
        {
            input_w = std::stoi(need("--input-w"));
            input_size_set = true;
        }
        else if (arg == "--input-h")
        {
            input_h = std::stoi(need("--input-h"));
            input_size_set = true;
        }
        else if (arg == "--warmup")
            warmup = std::stoi(need("--warmup"));
        else if (arg == "--iters")
            iters = std::stoi(need("--iters"));
        else if (arg == "--threads")
            threads = std::stoi(need("--threads"));
        else if (arg == "--fp16")
            use_fp16 = true;
        else if (arg == "--vulkan")
            use_vulkan = true;
        else if (arg == "--help" || arg == "-h")
        {
            usage(argv[0]);
            return 0;
        }
        else
        {
            std::fprintf(stderr, "Unknown arg: %s\n", arg.c_str());
            usage(argv[0]);
            return 2;
        }
    }

    if (param_path.empty() || bin_path.empty())
    {
        usage(argv[0]);
        return 2;
    }

    const bool infer_mode = !in_path.empty() || !out_path.empty();
    if (infer_mode && (in_path.empty() || out_path.empty()))
    {
        std::fprintf(stderr, "Infer mode requires both --in and --out\n");
        usage(argv[0]);
        return 2;
    }
    if (infer_mode && warmup == 50)
        warmup = 1; // default bench warmup is too heavy for a demo click

    ncnn::Net net;
    net.opt.num_threads = threads;
    net.opt.use_fp16_packed = use_fp16;
    net.opt.use_fp16_storage = use_fp16;
    net.opt.use_fp16_arithmetic = use_fp16;
    net.opt.use_vulkan_compute = use_vulkan;

    if (net.load_param(param_path.c_str()) != 0 || net.load_model(bin_path.c_str()) != 0)
    {
        std::fprintf(stderr, "Failed to load model\n");
        return 1;
    }

    if (infer_mode)
    {
        // If caller did not set size, keep defaults (180) so fixed graphs get a resize.
        // Pass input_w/h=0 only when we want native PNG size — not exposed; always resize to preset.
        (void)input_size_set;
        return run_infer(net, in_blob, out_blob, in_path, out_path, input_w, input_h, warmup,
            use_fp16, use_vulkan, threads);
    }

    ncnn::Mat in(input_w, input_h, 3);
    in.fill(0.5f);

    auto run_once = [&](ncnn::Mat& out) -> int {
        ncnn::Extractor ex = net.create_extractor();
        ex.set_light_mode(true);
        if (ex.input(in_blob.c_str(), in) != 0)
            return -1;
        if (ex.extract(out_blob.c_str(), out) != 0)
            return -1;
        return 0;
    };

    for (int i = 0; i < warmup; ++i)
    {
        ncnn::Mat out;
        if (run_once(out) != 0)
        {
            std::fprintf(stderr, "Inference failed (blobs: %s -> %s)\n", in_blob.c_str(), out_blob.c_str());
            return 1;
        }
    }

    std::vector<double> times_ms;
    times_ms.reserve(iters);
    for (int i = 0; i < iters; ++i)
    {
        auto t0 = std::chrono::steady_clock::now();
        ncnn::Mat out;
        if (run_once(out) != 0)
        {
            std::fprintf(stderr, "Inference failed during timed loop\n");
            return 1;
        }
        auto t1 = std::chrono::steady_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        times_ms.push_back(ms);
    }

    long vmhwm_kb = read_vmhwm_kb();
    double med = median_ms(times_ms);
    double p90 = percentile_ms(times_ms, 90.0);
    double fps = med > 0.0 ? 1000.0 / med : 0.0;

    std::printf(
        "{"
        "\"param\":\"%s\","
        "\"bin\":\"%s\","
        "\"in_blob\":\"%s\","
        "\"out_blob\":\"%s\","
        "\"input_w\":%d,"
        "\"input_h\":%d,"
        "\"warmup\":%d,"
        "\"iters\":%d,"
        "\"fp16\":%s,"
        "\"vulkan\":%s,"
        "\"threads\":%d,"
        "\"median_ms\":%.4f,"
        "\"p90_ms\":%.4f,"
        "\"fps\":%.2f,"
        "\"peak_memory_kb\":%ld"
        "}\n",
        param_path.c_str(),
        bin_path.c_str(),
        in_blob.c_str(),
        out_blob.c_str(),
        input_w,
        input_h,
        warmup,
        iters,
        use_fp16 ? "true" : "false",
        use_vulkan ? "true" : "false",
        threads,
        med,
        p90,
        fps,
        vmhwm_kb);

    return 0;
}
