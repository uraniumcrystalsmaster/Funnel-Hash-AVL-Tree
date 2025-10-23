#ifndef GPU_FUNNEL_HASH_MAP_H
#define GPU_FUNNEL_HASH_MAP_H

/*
This C++ implementation of the Funnel Hashing algorithm is based on the
Python implementation originally written by Matthew Stern. Find at:
https://github.com/sternma/optopenhash

The algorithm is described in the paper "Optimal Bounds for Open
Addressing Without Reordering" by Farach-Colton, Krapivin, and Kuszmaul,
available at: https://arxiv.org/abs/2501.02305
 */
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <chrono>
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <new>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

// CUDA Error Checking Utility
static void HandleCudaError(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        // Fix for potential null pointer crash on invalid error codes
        const char* error_str = cudaGetErrorString(err);
        std::string msg;
        if (error_str == nullptr) {
            msg = "Unknown CUDA Error (Invalid Error Code: " + std::to_string(static_cast<int>(err)) + ")";
        } else {
            msg = "CUDA Error: " + std::string(error_str);
        }
        throw std::runtime_error(msg + " in " + file + " at line " + std::to_string(line));
    }
}

#define checkCudaErrors(err) (HandleCudaError(err, __FILE__, __LINE__))

template <typename Key, typename Value>
struct DeviceSlot {
    Key key;
    Value value;
    int slot_state; // 0 = Empty, 1 = Tombstone, 2 = Occupied
};

template <typename Key, typename Value>
struct FunnelMapDeviceView {
    // Pointers to GPU global memory
    DeviceSlot<Key, Value>* d_all_slots;
    size_t* d_level_bucket_counts;
    size_t* d_level_salts;
    size_t* d_level_offsets; // Start index of each level in d_all_slots
    unsigned int* d_num_inserts;   // Pointer to a single unsigned int for atomic updates

    // Map geometry (passed by value)
    size_t capacity;
    size_t alpha; // # of levels
    size_t beta;  // Size of buckets
    size_t special_array_offset; // Start index of the special array in d_all_slots
    size_t special_salt;

    // --- Device-Side Hash Functions ---

    // MurmurHash3_x64_128 finalizer
    __device__ inline size_t fmix64(size_t k) const {
        k ^= k >> 33;
        k *= 0xff51afd7ed558ccdULL;
        k ^= k >> 33;
        k *= 0xc4ceb9fe1a85ec53ULL;
        k ^= k >> 33;
        return k;
    }

    __device__ inline size_t hash_bytes(const char* bytes, size_t len) const {
        // FNV-1a 64-bit hash
        // These are the standard FNV primes and offsets.
        const size_t fnv_prime = 0x100000001b3ULL;
        const size_t fnv_offset_basis = 0xcbf29ce484222325ULL;

        size_t hash = fnv_offset_basis;
        for (size_t i = 0; i < len; ++i) {
            hash ^= static_cast<size_t>(bytes[i]);
            hash *= fnv_prime;
        }
        return hash;
    }

    // Simple hash combine, usable on device
    template <class T>
    __device__ inline void hash_combine(size_t& seed, const T& v) const {
        // using FNV-1a before mixing into the seed.
        const char* bytes = reinterpret_cast<const char*>(&v);
        size_t v_hash = hash_bytes(bytes, sizeof(T));

        // Combine the new hash into the existing seed (Boost's hash_combine)
        seed ^= v_hash + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }

    // Hash function for the primary levels
    __device__ inline size_t _hash_level(const Key& key, size_t level_index) const {
        size_t seed = d_level_salts[level_index];
        hash_combine(seed, key);
        return seed;
    }

    // Hash function for the special fallback array
    __device__ inline size_t _hash_special(const Key& key) const {
        size_t seed = special_salt;
        hash_combine(seed, key);
        return seed;
    }
};

template <typename Key, typename Value>
__global__ void kernel_init_slots(FunnelMapDeviceView<Key, Value> view) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < view.capacity) {
        view.d_all_slots[idx].slot_state = 0; // 0 = Empty
        // Use (Key)-1 as the reserved value for an available slot
        view.d_all_slots[idx].key = (Key)-1;
    }
}

template <typename Key, typename Value>
__global__ void kernel_insert_batch(FunnelMapDeviceView<Key, Value> view,
                                    const Key* d_keys,
                                    const Value* d_values,
                                    size_t num_items) {

    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= num_items) {
        return;
    }

    const Key& key = d_keys[idx];
    const Value& value = d_values[idx];
    const Key EMPTY_KEY = (Key)-1;

    // Try to insert into primary levels
    for (size_t i = 0; i < view.alpha; ++i) {
        if (view.d_level_bucket_counts[i] == 0) continue;

        size_t bucket_index = view._hash_level(key, i) % view.d_level_bucket_counts[i];
        size_t start = view.d_level_offsets[i] + (bucket_index * view.beta);
        size_t end = start + view.beta;

        for (size_t slot_idx = start; slot_idx < end; ++slot_idx) {
            DeviceSlot<Key, Value>* slot = &view.d_all_slots[slot_idx];

            // --- FIX: Race-free logic using atomicCAS on key ---
            Key current_key = slot->key;

            // 1. Check for update
            if (current_key == key) {
                slot->value = value;
                return;
            }

            // 2. Check for available slot
            if (current_key == EMPTY_KEY) {
                // Try to claim this slot by writing our key
                Key old_key = atomicCAS(&slot->key, EMPTY_KEY, key);

                if (old_key == EMPTY_KEY) {
                    // SUCCESS: We claimed a new slot.
                    slot->value = value;

                    // Now, update state (0->2 or 1->2)
                    int old_state = atomicCAS(&slot->slot_state, 0, 2);
                    if (old_state == 1) { // It was a tombstone
                        atomicCAS(&slot->slot_state, 1, 2);
                    }

                    // We claimed a slot that wasn't counted (0 or 1), so add size.
                    atomicAdd(view.d_num_inserts, 1U);
                    return;

                } else if (old_key == key) {
                    // We lost the race, but to a thread with the *same key*.
                    // This is just an update.
                    slot->value = value;
                    return;
                }
                // Lost race to a *different* key. Continue probing.
            }
            // 3. Slot is occupied by another key. Continue probing.
            // --- END FIX ---
        }
    }

    // Try to insert into special array (using linear probing)
    size_t special_size = view.capacity - view.special_array_offset;
    if (special_size > 0) {
        size_t start_hash = view._hash_special(key) % special_size;

        for (size_t j = 0; j < special_size; ++j) {
            size_t slot_idx = view.special_array_offset + ((start_hash + j) % special_size);
            DeviceSlot<Key, Value>* slot = &view.d_all_slots[slot_idx];

            // --- FIX: Applying the same race-free logic here ---
            Key current_key = slot->key;

            // 1. Check for update
            if (current_key == key) {
                slot->value = value;
                return;
            }

            // 2. Check for available slot
            if (current_key == EMPTY_KEY) {
                Key old_key = atomicCAS(&slot->key, EMPTY_KEY, key);

                if (old_key == EMPTY_KEY) {
                    // SUCCESS: We claimed a new slot.
                    slot->value = value;

                    // Now, update state (0->2 or 1->2)
                    int old_state = atomicCAS(&slot->slot_state, 0, 2);
                    if (old_state == 1) { // It was a tombstone
                        atomicCAS(&slot->slot_state, 1, 2);
                    }

                    // We claimed a slot that wasn't counted (0 or 1), so add size.
                    atomicAdd(view.d_num_inserts, 1U);
                    return;

                } else if (old_key == key) {
                    // We lost the race, but to a thread with the *same key*.
                    // This is just an update.
                    slot->value = value;
                    return;
                }
                // Lost race to a different key. Continue probing.
            }
            // 3. Slot is occupied by another key. Continue probing.
            // --- END FIX ---
        }
    }
    // Table is full error would be thrown here, but we should not throw exceptions on the GPU for efficency.
}

template <typename Key, typename Value>
__global__ void kernel_find_batch(FunnelMapDeviceView<Key, Value> view,
                                  const Key* d_keys,
                                  Value* d_out_values,
                                  bool* d_out_found_mask,
                                  size_t num_items) {

    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= num_items) {
        return;
    }

    const Key& key = d_keys[idx];

    // Search primary levels (read-only, no atomics needed)
    for (size_t i = 0; i < view.alpha; ++i) {
        if (view.d_level_bucket_counts[i] == 0) continue;

        size_t bucket_index = view._hash_level(key, i) % view.d_level_bucket_counts[i];
        size_t start = view.d_level_offsets[i] + (bucket_index * view.beta);
        size_t end = start + view.beta;

        for (size_t slot_idx = start; slot_idx < end; ++slot_idx) {
            const DeviceSlot<Key, Value>* slot = &view.d_all_slots[slot_idx];

            // Read state with a memory fence (volatile)
            const volatile int slot_state = slot->slot_state;
            const volatile Key current_key = slot->key;

            if (current_key == key) {
                // Key matches, but is it *really* there? (not a tombstone)
                if (slot_state == 2) {
                    d_out_values[idx] = slot->value;
                    d_out_found_mask[idx] = true;
                    return;
                }
            }
            // If we hit an empty (not tombstone) slot, key can't be here.
            if (slot_state == 0) {
                break;
            }
        }
    }

    // Search special array
    size_t special_size = view.capacity - view.special_array_offset;
    if (special_size > 0) {
        size_t start_hash = view._hash_special(key) % special_size;

        for (size_t j = 0; j < special_size; ++j) {
            size_t slot_idx = view.special_array_offset + ((start_hash + j) % special_size);
            const DeviceSlot<Key, Value>* slot = &view.d_all_slots[slot_idx];

            const volatile int slot_state = slot->slot_state;
            const volatile Key current_key = slot->key;

            if (current_key == key) {
                if (slot_state == 2) {
                    d_out_values[idx] = slot->value;
                    d_out_found_mask[idx] = true;
                    return;
                }
            }
            if (slot_state == 0) {
                break;
            }
        }
    }

    // Not found
    d_out_found_mask[idx] = false;
}

template <typename Key, typename Value>
__global__ void kernel_erase_batch(FunnelMapDeviceView<Key, Value> view,
                                   const Key* d_keys,
                                   bool* d_out_erased_mask,
                                   size_t num_items) {

    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= num_items) {
        return;
    }

    const Key& key = d_keys[idx];
    d_out_erased_mask[idx] = false;
    const Key EMPTY_KEY = (Key)-1;

    // Search primary levels
    for (size_t i = 0; i < view.alpha; ++i) {
        if (view.d_level_bucket_counts[i] == 0) continue;

        size_t bucket_index = view._hash_level(key, i) % view.d_level_bucket_counts[i];
        size_t start = view.d_level_offsets[i] + (bucket_index * view.beta);
        size_t end = start + view.beta;

        for (size_t slot_idx = start; slot_idx < end; ++slot_idx) {
            DeviceSlot<Key, Value>* slot = &view.d_all_slots[slot_idx];

            // Read state with a memory fence
            const volatile int slot_state = slot->slot_state;
            const volatile Key current_key = slot->key;

            if (current_key == key) {
                // Atomically change state from 2 (Occupied) to 1 (Tombstone)
                int old_state = atomicCAS(&slot->slot_state, 2, 1);
                if (old_state == 2) {
                    // We were the thread that successfully erased it
                    __threadfence(); // Ensure state write is visible
                    slot->key = EMPTY_KEY; // Free the key slot
                    atomicSub(view.d_num_inserts, 1U);
                    d_out_erased_mask[idx] = true;
                }
                return; // Key found (even if we lost the race), stop searching
            }
            if (slot_state == 0) {
                break;
            }
        }
    }

    // Search special array
    size_t special_size = view.capacity - view.special_array_offset;
    if (special_size > 0) {
        size_t start_hash = view._hash_special(key) % special_size;

        for (size_t j = 0; j < special_size; ++j) {
            size_t slot_idx = view.special_array_offset + ((start_hash + j) % special_size);
            DeviceSlot<Key, Value>* slot = &view.d_all_slots[slot_idx];

            const volatile int slot_state = slot->slot_state;
            const volatile Key current_key = slot->key;

            if (current_key == key) {
                int old_state = atomicCAS(&slot->slot_state, 2, 1);
                if (old_state == 2) {
                    __threadfence();
                    slot->key = EMPTY_KEY;
                    atomicSub(view.d_num_inserts, 1U);
                    d_out_erased_mask[idx] = true;
                }
                return;
            }
            if (slot_state == 0) {
                break;
            }
        }
    }
}

template <typename Key, typename Value>
class FunnelHashMapGPU {
private:
    // --- Device-Side Data Pointers ---
    DeviceSlot<Key, Value>* d_all_slots = nullptr;
    size_t* d_level_bucket_counts = nullptr;
    size_t* d_level_salts = nullptr;
    size_t* d_level_offsets = nullptr;
    unsigned int* d_num_inserts = nullptr;

    // --- Host-Side Map Geometry ---
    size_t capacity;
    size_t alpha; // # of levels
    size_t beta;  // Size of buckets
    size_t special_array_offset;
    size_t special_salt;

    // --- Host-Side Helper Data (for constructor) ---
    std::vector<size_t> h_level_bucket_counts;
    std::vector<size_t> h_level_salts;
    std::vector<size_t> h_level_offsets;

    // --- Private Helper ---

    /**
     * @brief Creates the device-side "view" struct to be passed to kernels.
     */
    FunnelMapDeviceView<Key, Value> create_device_view() {
        FunnelMapDeviceView<Key, Value> view;
        view.d_all_slots = d_all_slots;
        view.d_level_bucket_counts = d_level_bucket_counts;
        view.d_level_salts = d_level_salts;
        view.d_level_offsets = d_level_offsets;
        view.d_num_inserts = d_num_inserts;
        view.capacity = capacity;
        view.alpha = alpha;
        view.beta = beta;
        view.special_array_offset = special_array_offset;
        view.special_salt = special_salt;
        return view;
    }

    void get_launch_params(size_t num_items, dim3& grid, dim3& block) {
        //Calculates optimal CUDA launch parameters (grid/block)
        const int threads_per_block = 256;
        int blocks = (num_items + threads_per_block - 1) / threads_per_block;

        grid = dim3(blocks);
        block = dim3(threads_per_block);
    }


public:
    explicit FunnelHashMapGPU(size_t num_items_to_insert, double delta = 0.1) {
        if (num_items_to_insert == 0) num_items_to_insert = 1;
        if (!(delta > 0 && delta < 1)) {
            throw std::invalid_argument("delta must be between 0 and 1.");
        }

        // 1. Calculate map geometry on the host
        double load_factor = 1.0 - delta;
        this->capacity = static_cast<size_t>(std::ceil(static_cast<double>(num_items_to_insert) / load_factor));

        this->alpha = static_cast<size_t>(std::ceil(4 * std::log2(1 / delta) + 10));
        this->beta = static_cast<size_t>(std::ceil(2 * std::log2(1 / delta)));
        if (this->beta == 0) this->beta = 1;

        size_t special_size = std::max(1.0, std::floor(3 * delta * this->capacity / 4));
        size_t primary_size = (this->capacity > special_size) ? this->capacity - special_size : 0;

        if (primary_size > 0 && this->beta > 0) {
            size_t remainder = primary_size % this->beta;
            primary_size -= remainder;
            special_size += remainder;
        }

        this->special_array_offset = primary_size;
        size_t total_buckets = (primary_size > 0 && beta > 0) ? primary_size / beta : 0;
        double a1 = (alpha > 0 && total_buckets > 0) ?
                    static_cast<double>(total_buckets) / (4.0 * (1.0 - std::pow(0.75, alpha))) :
                    static_cast<double>(total_buckets);

        std::mt19937 rng(std::chrono::high_resolution_clock::now().time_since_epoch().count());
        std::uniform_int_distribution<size_t> dist(0, std::numeric_limits<size_t>::max());

        size_t remaining_buckets = total_buckets;
        size_t current_offset = 0;
        for (size_t i = 0; i < alpha; ++i) {
            size_t a_i = std::max(1.0, std::round(a1 * std::pow(0.75, i)));
            if (remaining_buckets == 0 || a_i == 0) break;

            a_i = std::min(a_i, remaining_buckets);
            h_level_bucket_counts.push_back(a_i);

            size_t level_size = a_i * beta;
            h_level_offsets.push_back(current_offset);
            current_offset += level_size;

            h_level_salts.push_back(dist(rng));
            remaining_buckets -= a_i;
        }
        // Update alpha to the *actual* number of levels we created
        this->alpha = h_level_offsets.size();

        if (remaining_buckets > 0 && !h_level_bucket_counts.empty()) {
            h_level_bucket_counts.back() += remaining_buckets;
        }
        this->special_salt = dist(rng);

        // 2. Allocate memory on the GPU
        try {
            checkCudaErrors(cudaMalloc(&d_all_slots, capacity * sizeof(DeviceSlot<Key, Value>)));
            if (alpha > 0) {
                checkCudaErrors(cudaMalloc(&d_level_bucket_counts, alpha * sizeof(size_t)));
                checkCudaErrors(cudaMalloc(&d_level_salts, alpha * sizeof(size_t)));
                checkCudaErrors(cudaMalloc(&d_level_offsets, alpha * sizeof(size_t)));
            }
            // --- FIX: Use 32-bit unsigned int ---
            checkCudaErrors(cudaMalloc(&d_num_inserts, sizeof(unsigned int)));
        } catch (...) {
            // Free any partially allocated memory before re-throwing
            cudaFree(d_all_slots);
            cudaFree(d_level_bucket_counts);
            cudaFree(d_level_salts);
            cudaFree(d_level_offsets);
            cudaFree(d_num_inserts);
            throw;
        }

        // 3. Copy geometry vectors from Host to Device
        if (alpha > 0) {
            checkCudaErrors(cudaMemcpy(d_level_bucket_counts, h_level_bucket_counts.data(), alpha * sizeof(size_t), cudaMemcpyHostToDevice));
            checkCudaErrors(cudaMemcpy(d_level_salts, h_level_salts.data(), alpha * sizeof(size_t), cudaMemcpyHostToDevice));
            checkCudaErrors(cudaMemcpy(d_level_offsets, h_level_offsets.data(), alpha * sizeof(size_t), cudaMemcpyHostToDevice));
        }

        // 4. Initialize device memory
        // --- FIX: Use 32-bit unsigned int ---
        checkCudaErrors(cudaMemset(d_num_inserts, 0, sizeof(unsigned int)));

        dim3 grid, block;
        get_launch_params(capacity, grid, block);
        kernel_init_slots<Key, Value><<<grid, block>>>(create_device_view());
        checkCudaErrors(cudaGetLastError());
        checkCudaErrors(cudaDeviceSynchronize());
    }

    /**
     * @brief Destructor frees all GPU memory.
     */
    ~FunnelHashMapGPU() {
        cudaFree(d_all_slots);
        cudaFree(d_level_bucket_counts);
        cudaFree(d_level_salts);
        cudaFree(d_level_offsets);
        cudaFree(d_num_inserts);
    }

    // No copy/move constructors for simplicity
    FunnelHashMapGPU(const FunnelHashMapGPU&) = delete;
    FunnelHashMapGPU& operator=(const FunnelHashMapGPU&) = delete;

    /**
     * @brief Returns the number of items currently in the map.
     */
    size_t size() const {
        // --- FIX: Read 32-bit unsigned int, not 64-bit size_t ---
        unsigned int h_num_inserts = 0;
        checkCudaErrors(cudaMemcpy(&h_num_inserts, d_num_inserts, sizeof(unsigned int), cudaMemcpyDeviceToHost));
        return h_num_inserts;
    }

/**
     * @brief Returns the total storage capacity of the map.
     */
    size_t get_capacity() const {
        return capacity;
    }

    /**
     * @brief Inserts a batch of (key, value) pairs into the map.
     * @param d_keys Device pointer to an array of keys.
     * @param d_values Device pointer to an array of values.
     * @param num_items The number of items to insert.
     */
    void insert_batch(const Key* d_keys, const Value* d_values, size_t num_items) {
        if (num_items == 0) return;
        dim3 grid, block;
        get_launch_params(num_items, grid, block);
        kernel_insert_batch<Key, Value><<<grid, block>>>(create_device_view(), d_keys, d_values, num_items);
        checkCudaErrors(cudaGetLastError());
    }

    /**
     * @brief Finds a batch of keys.
     * @param d_keys Device pointer to an array of keys to find.
     * @param d_out_values Device pointer to an array where found values will be written.
     * @param d_out_found_mask Device pointer to a bool array (1/0) indicating if a key was found.
     * @param num_items The number of items to find.
     */
    void find_batch(const Key* d_keys, Value* d_out_values, bool* d_out_found_mask, size_t num_items) {
        if (num_items == 0) return;
        dim3 grid, block;
        get_launch_params(num_items, grid, block);
        kernel_find_batch<Key, Value><<<grid, block>>>(create_device_view(), d_keys, d_out_values, d_out_found_mask, num_items);
        checkCudaErrors(cudaGetLastError());
    }

    /**
     * @brief Erases a batch of keys from the map.
     * @param d_keys Device pointer to an array of keys to erase.
     * @param d_out_erased_mask Device pointer to a bool array (1/0) indicating if a key was found and erased.
     * @param num_items The number of items to erase.
     */
    void erase_batch(const Key* d_keys, bool* d_out_erased_mask, size_t num_items) {
        if (num_items == 0) return;
        dim3 grid, block;
        get_launch_params(num_items, grid, block);
        kernel_erase_batch<Key, Value><<<grid, block>>>(create_device_view(), d_keys, d_out_erased_mask, num_items);
        checkCudaErrors(cudaGetLastError());
    }

    /**
     * @brief Clears the map by resetting all slots to "Empty".
     */
    void clear() {
        // --- FIX: Use 32-bit unsigned int ---
        checkCudaErrors(cudaMemset(d_num_inserts, 0, sizeof(unsigned int)));

        dim3 grid, block;
        get_launch_params(capacity, grid, block);
        kernel_init_slots<Key, Value><<<grid, block>>>(create_device_view());
        checkCudaErrors(cudaGetLastError());
    }
};

#endif //GPU_FUNNEL_HASH_MAP_H


