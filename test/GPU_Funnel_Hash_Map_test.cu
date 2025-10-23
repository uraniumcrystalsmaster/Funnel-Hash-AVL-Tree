//
// Test file for FunnelHashMapGPU using Catch2
//
// ====================================================================
// IMPORTANT: How to Compile
// ====================================================================
//
// 1. You need the Catch2 header (e.g., catch_amalgamated.hpp).
//    Download it from: https://github.com/catchorg/Catch2
//
// 2. You must compile this file with NVCC (NVIDIA's CUDA Compiler).
//
// 3. Example compile command:
//    nvcc -std=c++17 -o test_runner test_gpu_funnel_map.cu -I/path/to/catch2/include
//
// 4. Run the executable:
//    ./test_runner
//
// ====================================================================

// This defines the main() function for Catch2
#define CATCH_CONFIG_MAIN
#include "catch2/catch_test_macros.hpp"

// Include the CUDA hash map implementation
#include "GPU_Funnel_Hash_Map.h"

#include <vector>
#include <numeric>
#include <iostream>
#include <map>

// ====================================================================
// Test Helper Functions & Scoped Allocators
// ====================================================================

// A simple RAII wrapper for device memory
template <typename T>
struct DeviceVector {
    T* d_ptr = nullptr;
    size_t num_items;

    DeviceVector(size_t n) : num_items(n) {
        if (n == 0) return; // Allow zero-sized vectors
        checkCudaErrors(cudaMalloc(&d_ptr, n * sizeof(T)));
    }

    ~DeviceVector() {
        cudaFree(d_ptr);
    }

    // Disable copy
    DeviceVector(const DeviceVector&) = delete;
    DeviceVector& operator=(const DeviceVector&) = delete;

    // Load data from host
    void copy_from_host(const std::vector<T>& h_vec) {
        if (h_vec.size() != num_items) {
             throw std::runtime_error("Host/Device vector size mismatch.");
        }
        if (num_items == 0) return;

        // FIX: Special handling for std::vector<bool>
        if constexpr (std::is_same_v<T, bool>) {
            // 1. Create a temporary buffer that has .data()
            std::vector<uint8_t> temp_buffer(num_items);

            // 2. Manually copy from the std::vector<bool> to the temp buffer
            for (size_t i = 0; i < num_items; ++i) {
                temp_buffer[i] = static_cast<uint8_t>(h_vec[i]);
            }

            // 3. Copy from temporary buffer to device
            checkCudaErrors(cudaMemcpy(d_ptr, temp_buffer.data(), num_items * sizeof(uint8_t), cudaMemcpyHostToDevice));

        } else {
            // This is the normal path for all other types
            checkCudaErrors(cudaMemcpy(d_ptr, h_vec.data(), num_items * sizeof(T), cudaMemcpyHostToDevice));
        }
    }

    // Copy data to host
    std::vector<T> copy_to_host() {
        std::vector<T> h_vec(num_items);
        if (num_items == 0) return h_vec;

        // FIX: Special handling for std::vector<bool>
        if constexpr (std::is_same_v<T, bool>) {
            // 1. Create a temporary buffer that has .data()
            std::vector<uint8_t> temp_buffer(num_items);

            // 2. Copy from device to the temporary buffer
            // On device, bool is 1 byte, so this is correct.
            checkCudaErrors(cudaMemcpy(temp_buffer.data(), d_ptr, num_items * sizeof(T), cudaMemcpyDeviceToHost));

            // 3. Manually copy from the temp buffer to the std::vector<bool>
            for (size_t i = 0; i < num_items; ++i) {
                h_vec[i] = static_cast<T>(temp_buffer[i]);
            }
        } else {
            // This is the normal path for all other types
            checkCudaErrors(cudaMemcpy(h_vec.data(), d_ptr, num_items * sizeof(T), cudaMemcpyDeviceToHost));
        }
        return h_vec;
    }

    // Get raw device pointer
    T* data() { return d_ptr; }
    const T* data() const { return d_ptr; }
};


// =================================T===================================
// Test Cases
// ====================================================================

using TestKey = unsigned int;
using TestValue = int;

TEST_CASE("1. Construction and Initial State", "[FunnelHashMapGPU]") {
    size_t num_items = 1000;
    FunnelHashMapGPU<TestKey, TestValue> gpu_map(num_items);

    REQUIRE(gpu_map.size() == 0);
    REQUIRE(gpu_map.get_capacity() > 0);
    // Capacity should be at least num_items / (1.0 - delta)
    REQUIRE(gpu_map.get_capacity() >= static_cast<size_t>(num_items / 0.9));

    SECTION("Construction with zero items") {
        FunnelHashMapGPU<TestKey, TestValue> zero_map(0);
        REQUIRE(zero_map.size() == 0);
        REQUIRE(zero_map.get_capacity() > 0); // Still allocates a minimal map
    }
}

TEST_CASE("2. Insert, Find (Hit), Erase, Find (Miss)", "[FunnelHashMapGPU]") {
    const size_t num_items = 1024;
    FunnelHashMapGPU<TestKey, TestValue> gpu_map(num_items);

    std::vector<TestKey> h_keys(num_items);
    std::vector<TestValue> h_values(num_items);
    for (size_t i = 0; i < num_items; ++i) {
        h_keys[i] = i * 10; // Use non-contiguous keys
        h_values[i] = static_cast<TestValue>(i);
    }

    // Allocate device memory
    DeviceVector<TestKey> d_keys(num_items);
    DeviceVector<TestValue> d_values(num_items);
    DeviceVector<TestValue> d_out_values(num_items);
    DeviceVector<bool> d_out_mask(num_items);

    // Copy to device
    d_keys.copy_from_host(h_keys);
    d_values.copy_from_host(h_values);

    SECTION("Insert and Find (Hit)") {
        gpu_map.insert_batch(d_keys.data(), d_values.data(), num_items);
        checkCudaErrors(cudaDeviceSynchronize());

        REQUIRE(gpu_map.size() == num_items);

        // Find the keys we just inserted
        gpu_map.find_batch(d_keys.data(), d_out_values.data(), d_out_mask.data(), num_items);
        checkCudaErrors(cudaDeviceSynchronize());

        // Copy results back to host
        std::vector<TestValue> h_out_values = d_out_values.copy_to_host();
        std::vector<bool> h_out_mask = d_out_mask.copy_to_host();

        for (size_t i = 0; i < num_items; ++i) {
            REQUIRE(h_out_mask[i] == true);
            REQUIRE(h_out_values[i] == h_values[i]);
        }
    }

    SECTION("Find (Miss)") {
        // Create keys that don't exist
        std::vector<TestKey> h_miss_keys(num_items);
        for (size_t i = 0; i < num_items; ++i) {
            h_miss_keys[i] = i * 10 + 1; // e.g., 1, 11, 21, ...
        }
        DeviceVector<TestKey> d_miss_keys(num_items);
        d_miss_keys.copy_from_host(h_miss_keys);

        gpu_map.find_batch(d_miss_keys.data(), d_out_values.data(), d_out_mask.data(), num_items);
        checkCudaErrors(cudaDeviceSynchronize());

        std::vector<bool> h_out_mask = d_out_mask.copy_to_host();

        for (size_t i = 0; i < num_items; ++i) {
            REQUIRE(h_out_mask[i] == false);
        }
    }

    SECTION("Erase and Find (Miss)") {
        // First, insert them
        gpu_map.insert_batch(d_keys.data(), d_values.data(), num_items);
        checkCudaErrors(cudaDeviceSynchronize());
        REQUIRE(gpu_map.size() == num_items);

        // Now, erase them
        DeviceVector<bool> d_erase_mask(num_items);
        gpu_map.erase_batch(d_keys.data(), d_erase_mask.data(), num_items);
        checkCudaErrors(cudaDeviceSynchronize());

        // Check that erase was successful
        std::vector<bool> h_erase_mask = d_erase_mask.copy_to_host();
        for (size_t i = 0; i < num_items; ++i) {
            REQUIRE(h_erase_mask[i] == true);
        }

        REQUIRE(gpu_map.size() == 0);

        // Try to find the keys again
        gpu_map.find_batch(d_keys.data(), d_out_values.data(), d_out_mask.data(), num_items);
        checkCudaErrors(cudaDeviceSynchronize());

        // Check that they are all gone
        std::vector<bool> h_find_mask = d_out_mask.copy_to_host();
        for (size_t i = 0; i < num_items; ++i) {
            REQUIRE(h_find_mask[i] == false);
        }
    }
}

TEST_CASE("4. Re-insert after erase (Tombstone test)", "[FunnelHashMapGPU]") {
    const size_t num_items = 1024;
    FunnelHashMapGPU<TestKey, TestValue> gpu_map(num_items);

    std::vector<TestKey> h_keys(num_items);
    std::vector<TestValue> h_values(num_items);
    for (size_t i = 0; i < num_items; ++i) {
        h_keys[i] = i;
        h_values[i] = static_cast<TestValue>(i);
    }

    DeviceVector<TestKey> d_keys(num_items);
    DeviceVector<TestValue> d_values(num_items);
    DeviceVector<bool> d_mask(num_items);
    DeviceVector<TestValue> d_out_values(num_items);

    d_keys.copy_from_host(h_keys);
    d_values.copy_from_host(h_values);

    // 1. Insert all
    gpu_map.insert_batch(d_keys.data(), d_values.data(), num_items);
    checkCudaErrors(cudaDeviceSynchronize());
    REQUIRE(gpu_map.size() == num_items);

    // 2. Erase all (this creates tombstones)
    gpu_map.erase_batch(d_keys.data(), d_mask.data(), num_items);
    checkCudaErrors(cudaDeviceSynchronize());
    REQUIRE(gpu_map.size() == 0);

    // 3. Re-insert all (this must reclaim the tombstones)
    gpu_map.insert_batch(d_keys.data(), d_values.data(), num_items);
    checkCudaErrors(cudaDeviceSynchronize());

    // 4. Check size (this tests the tombstone atomicAdd)
    REQUIRE(gpu_map.size() == num_items);

    // 5. Verify all keys are findable
    gpu_map.find_batch(d_keys.data(), d_out_values.data(), d_mask.data(), num_items);
    checkCudaErrors(cudaDeviceSynchronize());

    std::vector<bool> h_find_mask = d_mask.copy_to_host();
    for (size_t i = 0; i < num_items; ++i) {
        REQUIRE(h_find_mask[i] == true);
    }
}


TEST_CASE("5. Large Insert (Stress Test) and Clear", "[FunnelHashMapGPU]") {
    const size_t num_items = 100000; // 100k items
    FunnelHashMapGPU<TestKey, TestValue> gpu_map(num_items);

    std::vector<TestKey> h_keys(num_items);
    std::vector<TestValue> h_values(num_items);
    for (size_t i = 0; i < num_items; ++i) {
        h_keys[i] = i;
        h_values[i] = static_cast<TestValue>(i % 1000); // Values can repeat
    }

    DeviceVector<TestKey> d_keys(num_items);
    DeviceVector<TestValue> d_values(num_items);
    d_keys.copy_from_host(h_keys);
    d_values.copy_from_host(h_values);

    // Insert
    gpu_map.insert_batch(d_keys.data(), d_values.data(), num_items);
    checkCudaErrors(cudaDeviceSynchronize());

    REQUIRE(gpu_map.size() == num_items);

    // Find
    DeviceVector<TestValue> d_out_values(num_items);
    DeviceVector<bool> d_out_mask(num_items);
    gpu_map.find_batch(d_keys.data(), d_out_values.data(), d_out_mask.data(), num_items);
    checkCudaErrors(cudaDeviceSynchronize());

    // Spot check a few results (copying all 100k back is slow)
    std::vector<bool> h_out_mask = d_out_mask.copy_to_host();
    REQUIRE(h_out_mask[0] == true);
    REQUIRE(h_out_mask[num_items / 2] == true);
    REQUIRE(h_out_mask[num_items - 1] == true);

    SECTION("Clear") {
        gpu_map.clear();
        checkCudaErrors(cudaDeviceSynchronize());

        REQUIRE(gpu_map.size() == 0);

        // Verify they are gone
        gpu_map.find_batch(d_keys.data(), d_out_values.data(), d_out_mask.data(), num_items);
        checkCudaErrors(cudaDeviceSynchronize());

        std::vector<bool> h_cleared_mask = d_out_mask.copy_to_host();
        for (size_t i = 0; i < num_items; ++i) {
            REQUIRE(h_cleared_mask[i] == false);
        }
    }
}

TEST_CASE("6. Update Existing Values", "[FunnelHashMapGPU]") {
    const size_t num_items = 1024;
    FunnelHashMapGPU<TestKey, TestValue> gpu_map(num_items);

    std::vector<TestKey> h_keys(num_items);
    std::vector<TestValue> h_values(num_items);
    std::vector<TestValue> h_new_values(num_items);

    for (size_t i = 0; i < num_items; ++i) {
        h_keys[i] = i;
        h_values[i] = static_cast<TestValue>(i);
        h_new_values[i] = static_cast<TestValue>(i * 2 + 1); // e.g., 1, 3, 5, ...
    }

    DeviceVector<TestKey> d_keys(num_items);
    DeviceVector<TestValue> d_values(num_items);
    DeviceVector<TestValue> d_new_values(num_items);
    DeviceVector<TestValue> d_out_values(num_items);
    DeviceVector<bool> d_mask(num_items);

    d_keys.copy_from_host(h_keys);
    d_values.copy_from_host(h_values);
    d_new_values.copy_from_host(h_new_values);

    // 1. Insert original values
    gpu_map.insert_batch(d_keys.data(), d_values.data(), num_items);
    checkCudaErrors(cudaDeviceSynchronize());
    REQUIRE(gpu_map.size() == num_items);

    // 2. Insert new values for the same keys
    gpu_map.insert_batch(d_keys.data(), d_new_values.data(), num_items);
    checkCudaErrors(cudaDeviceSynchronize());

    // 3. Size should not change
    REQUIRE(gpu_map.size() == num_items);

    // 4. Find keys and check for *new* values
    gpu_map.find_batch(d_keys.data(), d_out_values.data(), d_mask.data(), num_items);
    checkCudaErrors(cudaDeviceSynchronize());

    std::vector<TestValue> h_out_values = d_out_values.copy_to_host();
    std::vector<bool> h_mask = d_mask.copy_to_host();

    for (size_t i = 0; i < num_items; ++i) {
        REQUIRE(h_mask[i] == true);
        REQUIRE(h_out_values[i] == h_new_values[i]); // Check for the updated value
    }
}

TEST_CASE("7. Erase Non-Existent Keys", "[FunnelHashMapGPU]") {
    const size_t num_items = 1024;
    FunnelHashMapGPU<TestKey, TestValue> gpu_map(num_items);

    std::vector<TestKey> h_keys(num_items);
    std::vector<TestValue> h_values(num_items);
    std::vector<TestKey> h_other_keys(num_items);

    for (size_t i = 0; i < num_items; ++i) {
        h_keys[i] = i * 2; // e.g., 0, 2, 4
        h_values[i] = static_cast<TestValue>(i);
        h_other_keys[i] = i * 2 + 1; // e.g., 1, 3, 5
    }

    DeviceVector<TestKey> d_keys(num_items);
    DeviceVector<TestValue> d_values(num_items);
    DeviceVector<TestKey> d_other_keys(num_items);
    DeviceVector<bool> d_mask(num_items);

    d_keys.copy_from_host(h_keys);
    d_values.copy_from_host(h_values);
    d_other_keys.copy_from_host(h_other_keys);

    // 1. Insert original keys
    gpu_map.insert_batch(d_keys.data(), d_values.data(), num_items);
    checkCudaErrors(cudaDeviceSynchronize());
    REQUIRE(gpu_map.size() == num_items);

    // 2. Try to erase keys that don't exist
    gpu_map.erase_batch(d_other_keys.data(), d_mask.data(), num_items);
    checkCudaErrors(cudaDeviceSynchronize());

    // 3. Check that size is unchanged
    REQUIRE(gpu_map.size() == num_items);

    // 4. Check that mask is all false
    std::vector<bool> h_mask = d_mask.copy_to_host();
    for (size_t i = 0; i < num_items; ++i) {
        REQUIRE(h_mask[i] == false);
    }
}

TEST_CASE("8. Insert Duplicate Keys (Concurrency)", "[FunnelHashMapGPU]") {
    const size_t num_unique_keys = 10000;
    const size_t num_duplicates = 10;
    const size_t total_items = num_unique_keys * num_duplicates; // 100k items

    FunnelHashMapGPU<TestKey, TestValue> gpu_map(total_items);

    std::vector<TestKey> h_keys(total_items);
    std::vector<TestValue> h_values(total_items);
    // Use std::map to store the "last" value for each key
    std::map<TestKey, TestValue> h_final_values;

    for (size_t i = 0; i < total_items; ++i) {
        TestKey key = i % num_unique_keys;
        TestValue value = static_cast<TestValue>(i);
        h_keys[i] = key;
        h_values[i] = value;
        h_final_values[key] = value; // Last value wins
    }

    DeviceVector<TestKey> d_keys(total_items);
    DeviceVector<TestValue> d_values(total_items);
    d_keys.copy_from_host(h_keys);
    d_values.copy_from_host(h_values);

    // 1. Insert all 100k items (with 10 duplicates per key)
    gpu_map.insert_batch(d_keys.data(), d_values.data(), total_items);
    checkCudaErrors(cudaDeviceSynchronize());

    // 2. Final size should be the number of *unique* keys
    REQUIRE(gpu_map.size() == num_unique_keys);

    // 3. Verify the final values are correct (last one wins)
    std::vector<TestKey> h_unique_keys(num_unique_keys);
    std::vector<TestValue> h_expected_values(num_unique_keys);
    size_t i = 0;
    for (const auto& pair : h_final_values) {
        h_unique_keys[i] = pair.first;
        h_expected_values[i] = pair.second;
        i++;
    }

    DeviceVector<TestKey> d_unique_keys(num_unique_keys);
    DeviceVector<TestValue> d_out_values(num_unique_keys);
    DeviceVector<bool> d_mask(num_unique_keys);

    d_unique_keys.copy_from_host(h_unique_keys);

    gpu_map.find_batch(d_unique_keys.data(), d_out_values.data(), d_mask.data(), num_unique_keys);
    checkCudaErrors(cudaDeviceSynchronize());

    std::vector<TestValue> h_out_values = d_out_values.copy_to_host();
    std::vector<bool> h_mask = d_mask.copy_to_host();

    for (size_t j = 0; j < num_unique_keys; ++j) {
        REQUIRE(h_mask[j] == true);
        REQUIRE(h_out_values[j] == h_expected_values[j]);
    }
}

