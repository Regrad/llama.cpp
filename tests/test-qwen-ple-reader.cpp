#include "../src/models/qwen4exp-ple-reader.h"
#include "ggml-backend.h"

#include <memory>
#include <string>

static void require(bool ok, const char * message) {
    if (!ok) {
        throw std::runtime_error(message);
    }
}

static void test_reader(ggml_type type, unsigned threads) {
    constexpr int64_t dim = 256, rows = 257;
    ggml_init_params params = {4*1024*1024, nullptr, false};
    std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init(params), ggml_free);
    require(ctx != nullptr, "context allocation failed");
    ggml_tensor * table = ggml_new_tensor_2d(ctx.get(), type, dim, rows);
    std::vector<float> original(dim*rows);
    for (size_t i = 0; i < original.size(); ++i) {
        original[i] = (float) ((int) (i*31 % 113) - 56)/17.0f;
    }
    std::vector<float> importance(dim, 1.0f);
    if (type == GGML_TYPE_F32) {
        memcpy(table->data, original.data(), original.size()*sizeof(float));
    } else {
        ggml_quantize_chunk(type, original.data(), table->data, 0, rows, dim, importance.data());
    }
    std::unique_ptr<FILE, decltype(&fclose)> file(tmpfile(), fclose);
    require(file != nullptr, "temporary file allocation failed");
    const size_t offset = 37;
    std::vector<uint8_t> prefix(offset, 0x5a);
    require(fwrite(prefix.data(), 1, prefix.size(), file.get()) == prefix.size(), "prefix write failed");
    require(fwrite(table->data, 1, ggml_nbytes(table), file.get()) == ggml_nbytes(table), "table write failed");
    require(fflush(file.get()) == 0, "flush failed");
#ifdef _WIN32
    const int fd = _fileno(file.get());
#else
    const int fd = fileno(file.get());
#endif
    const size_t file_size = offset + ggml_nbytes(table);
    llama_qwen_ple_reader reader(fd, file_size, offset, table, threads);
    std::vector<int32_t> indices;
    for (int i = 0; i < 1137; ++i) {
        indices.push_back((i*73 + 19) % rows);
    }
    indices[0] = 0;
    indices[1] = rows - 1;
    ggml_tensor * index_tensor = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, indices.size());
    memcpy(index_tensor->data, indices.data(), indices.size()*sizeof(int32_t));
    ggml_tensor * reference = ggml_get_rows(ctx.get(), table, index_tensor);
    ggml_cgraph * graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, reference);
    std::unique_ptr<ggml_backend, decltype(&ggml_backend_free)> cpu(
            ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr), ggml_backend_free);
    require(cpu != nullptr, "CPU backend allocation failed");
    require(ggml_backend_graph_compute(cpu.get(), graph) == GGML_STATUS_SUCCESS, "GET_ROWS reference failed");
    auto check = [&]() {
        std::vector<float> result;
        for (int repeat = 0; repeat < 3; ++repeat) {
            reader.gather(indices, result);
            require(result.size()*sizeof(float) == ggml_nbytes(reference), "row size differs");
            require(memcmp(result.data(), reference->data, ggml_nbytes(reference)) == 0,
                    "direct rows differ bitwise from CPU GET_ROWS");
        }
    };
    check();
    std::exception_ptr concurrent_error;
    std::thread concurrent([&]() {
        try { check(); } catch (...) { concurrent_error = std::current_exception(); }
    });
    std::exception_ptr main_error;
    try { check(); } catch (...) { main_error = std::current_exception(); }
    concurrent.join();
    if (main_error) { std::rethrow_exception(main_error); }
    if (concurrent_error) { std::rethrow_exception(concurrent_error); }
    std::vector<float> result;
    reader.gather({}, result);
    require(result.empty(), "empty gather failed");
    for (int32_t bad : {int32_t(-1), int32_t(rows)}) {
        bool rejected = false;
        try { reader.gather({bad}, result); } catch (const std::runtime_error &) { rejected = true; }
        require(rejected, "invalid row accepted");
        check();
    }
    bool rejected = false;
    try { llama_qwen_ple_reader truncated(fd, file_size - 1, offset, table, 1); }
    catch (const std::runtime_error &) { rejected = true; }
    require(rejected, "truncated table accepted");
#ifdef _WIN32
    require(_chsize_s(fd, file_size - 1) == 0, "test file truncate failed");
#else
    require(ftruncate(fd, (off_t) (file_size - 1)) == 0, "test file truncate failed");
#endif
    rejected = false;
    try { reader.gather({int32_t(rows - 1)}, result); }
    catch (const std::runtime_error &) { rejected = true; }
    require(rejected, "short positional read accepted");
    require(fseek(file.get(), 0, SEEK_END) == 0, "test file seek failed");
    const uint8_t last = ((const uint8_t *) table->data)[ggml_nbytes(table) - 1];
    require(fwrite(&last, 1, 1, file.get()) == 1 && fflush(file.get()) == 0, "test file restore failed");
    check();
    printf("PASS %s threads=%u: exact GET_ROWS, duplicate/boundary/empty rows, repeated/concurrent gathers, bounds errors\n",
            ggml_type_name(type), threads);
}

int main() {
    try {
        ggml_backend_load_all();
        for (ggml_type type : {GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_Q8_0, GGML_TYPE_IQ4_NL}) {
            for (unsigned threads : {1u, 16u}) {
                test_reader(type, threads);
            }
        }
        puts("PLE_READER_CORRECTNESS_PASSED");
    } catch (const std::exception & error) {
        fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
    return 0;
}
