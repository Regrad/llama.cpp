#pragma once

#include "ggml.h"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <exception>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <io.h>
#else
#include <cerrno>
#include <unistd.h>
#endif

// Read the same packed rows as GET_ROWS, without demand-faulting the whole table.
class llama_qwen_ple_reader {
public:
    llama_qwen_ple_reader(int file_id, size_t file_size, size_t offset, const ggml_tensor * table,
                          unsigned n_threads = 16) :
        offset(offset), row_size(ggml_row_size(table->type, table->ne[0])),
        row_count(table->ne[1]), row_dim(table->ne[0]), type(table->type),
        to_float(ggml_get_type_traits(table->type)->to_float) {
        if (file_id < 0 || !ggml_is_contiguous(table) || table->ne[2] != 1 || table->ne[3] != 1 ||
                row_dim <= 0 || row_count <= 0 || row_size == 0 ||
                row_size > std::numeric_limits<uint32_t>::max() ||
                offset > file_size || (uint64_t) row_count > (file_size - offset)/row_size ||
                (uint64_t) row_dim > SIZE_MAX/sizeof(float) || (type != GGML_TYPE_F32 && !to_float)) {
            throw std::runtime_error("invalid PLE direct reader table or file bounds");
        }
#ifdef _WIN32
        file = ReOpenFile((HANDLE) _get_osfhandle(file_id), GENERIC_READ,
                FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                FILE_FLAG_OVERLAPPED | FILE_FLAG_RANDOM_ACCESS);
        if (file == INVALID_HANDLE_VALUE) {
            throw std::runtime_error("cannot open PLE file for positional reads");
        }
#else
        file = dup(file_id);
        if (file < 0) {
            throw std::runtime_error("cannot duplicate PLE file for positional reads");
        }
#endif
        n_threads = std::max(1u, std::min(n_threads, 32u));
        try {
            for (unsigned i = 0; i < n_threads; ++i) {
                workers.emplace_back([this]() { worker(); });
            }
        } catch (...) {
            stop();
            close_file();
            throw;
        }
    }

    ~llama_qwen_ple_reader() {
        stop();
        close_file();
    }

    llama_qwen_ple_reader(const llama_qwen_ple_reader &) = delete;
    llama_qwen_ple_reader & operator=(const llama_qwen_ple_reader &) = delete;

    void gather(const std::vector<int32_t> & indices, std::vector<float> & output) {
        std::lock_guard<std::mutex> serial(gather_mutex);
        if (indices.size() > SIZE_MAX/(size_t) row_dim/sizeof(float)) {
            throw std::runtime_error("PLE gathered rows are too large");
        }
        order.resize(indices.size());
        for (size_t i = 0; i < indices.size(); ++i) {
            if (indices[i] < 0 || indices[i] >= row_count) {
                throw std::runtime_error("PLE row index is outside the table");
            }
            order[i] = {indices[i], i};
        }
        std::sort(order.begin(), order.end(), [](const row_request & a, const row_request & b) {
            return a.row < b.row;
        });
        groups.clear();
        for (size_t i = 0; i < order.size(); ++i) {
            if (i == 0 || order[i].row != order[i - 1].row) {
                groups.push_back(i);
            }
        }
        groups.push_back(order.size());
        output.resize(indices.size() * (size_t) row_dim);
        if (indices.empty()) {
            return;
        }
        std::unique_lock<std::mutex> lock(work_mutex);
        result = output.data();
        next.store(0);
        error = nullptr;
        pending = workers.size();
        ++generation;
        ready.notify_all();
        done.wait(lock, [this]() { return pending == 0; });
        result = nullptr;
        if (error) {
            std::rethrow_exception(error);
        }
    }

private:
    struct row_request { int32_t row; size_t destination; };
    size_t offset, row_size;
    int64_t row_count, row_dim;
    ggml_type type;
    ggml_to_float_t to_float;
#ifdef _WIN32
    HANDLE file = INVALID_HANDLE_VALUE;
#else
    int file = -1;
#endif
    std::vector<std::thread> workers;
    std::mutex gather_mutex, work_mutex;
    std::condition_variable ready, done;
    bool stopping = false;
    uint64_t generation = 0;
    size_t pending = 0;
    std::atomic<size_t> next{0};
    std::exception_ptr error;
    std::vector<row_request> order;
    std::vector<size_t> groups;
    float * result = nullptr;

    struct worker_event {
#ifdef _WIN32
        HANDLE handle = nullptr;
        ~worker_event() { if (handle) { CloseHandle(handle); } }
#endif
    };

    void worker() {
        uint64_t seen = 0;
        // uint64_t storage preserves alignment for quantized block decoders.
        std::vector<uint64_t> packed;
        std::vector<float> decoded;
        worker_event event;
        for (;;) {
            std::unique_lock<std::mutex> lock(work_mutex);
            ready.wait(lock, [this, &seen]() { return stopping || generation != seen; });
            if (stopping) {
                return;
            }
            seen = generation;
            lock.unlock();
            try {
                packed.resize((row_size + sizeof(uint64_t) - 1)/sizeof(uint64_t));
                decoded.resize((size_t) row_dim);
#ifdef _WIN32
                if (!event.handle) {
                    event.handle = CreateEventW(nullptr, TRUE, FALSE, nullptr);
                    if (!event.handle) {
                        throw std::runtime_error("cannot create PLE read event");
                    }
                }
#endif
                for (;;) {
                    const size_t group = next.fetch_add(1);
                    if (group + 1 >= groups.size()) {
                        break;
                    }
                    const size_t begin = groups[group], end = groups[group + 1];
                    const size_t pos = offset + (size_t) order[begin].row * row_size;
                    read_row(packed.data(), pos, event);
                    if (type == GGML_TYPE_F32) {
                        std::memcpy(decoded.data(), packed.data(), (size_t) row_dim * sizeof(float));
                    } else {
                        to_float(packed.data(), decoded.data(), row_dim);
                    }
                    for (size_t i = begin; i < end; ++i) {
                        std::memcpy(result + order[i].destination * (size_t) row_dim,
                                decoded.data(), (size_t) row_dim * sizeof(float));
                    }
                }
            } catch (...) {
                std::lock_guard<std::mutex> guard(work_mutex);
                if (!error) {
                    error = std::current_exception();
                }
            }
            lock.lock();
            if (--pending == 0) {
                done.notify_one();
            }
        }
    }

    void read_row(void * data, size_t pos, [[maybe_unused]] const worker_event & event) const {
#ifdef _WIN32
        OVERLAPPED request = {};
        request.Offset = (DWORD) pos;
        request.OffsetHigh = (DWORD) ((uint64_t) pos >> 32);
        // A private event keeps concurrent reads on one file handle independent.
        request.hEvent = event.handle;
        ResetEvent(request.hEvent);
        DWORD count = 0;
        const BOOL started = ReadFile(file, data, (DWORD) row_size, nullptr, &request);
        const DWORD start_error = started ? ERROR_SUCCESS : GetLastError();
        const BOOL completed = (started || start_error == ERROR_IO_PENDING) &&
                GetOverlappedResult(file, &request, &count, TRUE);
        if (!completed || count != row_size) {
            throw std::runtime_error("PLE positional read failed or reached end of file");
        }
#else
        size_t count = 0;
        while (count < row_size) {
            const ssize_t n = pread(file, (char *) data + count, row_size - count, (off_t) (pos + count));
            if (n < 0 && errno == EINTR) {
                continue;
            }
            if (n <= 0) {
                throw std::runtime_error("PLE positional read failed or reached end of file");
            }
            count += (size_t) n;
        }
#endif
    }

    void stop() {
        {
            std::lock_guard<std::mutex> lock(work_mutex);
            stopping = true;
        }
        ready.notify_all();
        for (auto & thread : workers) {
            thread.join();
        }
    }

    void close_file() {
#ifdef _WIN32
        CloseHandle(file);
#else
        close(file);
#endif
    }
};
