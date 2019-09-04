//
// Uber, Inc. (c) 2019
//

#include <chrono>
#include <numeric>
#include <vector>

// Times a lambda with a specified resolution
// Runs `fn` for `warmup` iterations without storing timing information
// and then runs it for `iterations` iterations while storing timing information
// Returns the mean time that `fn` took to execute
template<typename Resolution, typename T>
float time_lambda(size_t warmup, size_t iterations, T fn)
{
    std::vector<size_t> times;

    for (int i = 0; i < warmup + iterations; i++)
    {
        auto start = std::chrono::high_resolution_clock::now();

        fn();

        auto end = std::chrono::high_resolution_clock::now();

        // Ignore the warmup period
        if (i > warmup)
        {
            times.emplace_back(std::chrono::duration_cast<Resolution>(end - start).count());
        }
    }

    return std::accumulate(times.begin(), times.end(), 0.0) / times.size();
}
