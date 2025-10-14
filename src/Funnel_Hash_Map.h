/*
This C++ implementation of the Funnel Hashing algorithm is based on the
Python implementation originally written by Matthew Stern. Find at:
https://github.com/sternma/optopenhash

The algorithm is described in the paper "Optimal Bounds for Open
Addressing Without Reordering" by Farach-Colton, Krapivin, and Kuszmaul,
available at: https://arxiv.org/abs/2501.02305
 */
#ifndef FUNNEL_HASH_MAP_H
#define FUNNEL_HASH_MAP_H
#include <vector>
#include <cmath>
#include <iostream>
#include <utility>
#include <functional>
#include <stdexcept>
#include <random>
#include <chrono>
#include <limits>
#include <new>

template <typename Key, typename Value>
class Funnel_Hash_Map {
public:
    // A Slot can be occupied, deleted (tombstone), or empty.
    struct Slot {
        std::pair<const Key, Value> data;
        bool is_occupied = false;
        bool is_deleted = false;

        Slot() : data(Key{}, Value{}), is_occupied(false), is_deleted(false) {}
        Slot(const Key& key, const Value& value) : data(key, value), is_occupied(true), is_deleted(false) {}
    };

private:
    size_t capacity;
    double delta;
    size_t max_inserts;
    size_t num_inserts;

    size_t alpha; // # of levels
    size_t beta;  // Size of buckets within levels

    // Each level is a separate vector of Slots
    std::vector<std::vector<Slot>> levels;
    std::vector<size_t> level_bucket_counts;
    std::vector<size_t> level_salts;

    // Fallback array for when all levels fail
    std::vector<Slot> special_array;
    size_t special_salt;

    // Helper to combine hashes, since C++ doesn't hash tuples directly
    template <class T>
    inline void hash_combine(std::size_t& seed, const T& v) const {
        std::hash<T> hasher;
        seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }

    // Hash function for the primary levels, depends on the level index
    size_t _hash_level(const Key& key, size_t level_index) const {
        size_t seed = level_salts[level_index];
        hash_combine(seed, key);
        return seed;
    }

    // Hash function for the special fallback array
    size_t _hash_special(const Key& key) const {
        size_t seed = special_salt;
        hash_combine(seed, key);
        return seed;
    }

    // Helper to perform the actual insertion
    void place_item(Slot& slot, const Key& key, const Value& value) {
        if (!slot.is_occupied) {
            num_inserts++;
        }
        slot.~Slot();
        new (&slot) Slot(key, Value(value)); // Use placement new for const correctness
    }

public:
    // Implementing a full-featured iterator class
    class iterator {
        Funnel_Hash_Map* map_ptr;
        size_t level_idx;
        size_t slot_idx;
        bool in_special_array;

        void advance() {
            if (in_special_array) {
                slot_idx++;
                while (slot_idx < map_ptr->special_array.size() && !map_ptr->special_array[slot_idx].is_occupied) {
                    slot_idx++;
                }
            } else {
                slot_idx++;
                while (level_idx < map_ptr->levels.size()) {
                    if (slot_idx < map_ptr->levels[level_idx].size()) {
                        if (map_ptr->levels[level_idx][slot_idx].is_occupied) {
                            return; // Found next item
                        }
                        slot_idx++;
                    } else {
                        level_idx++;
                        slot_idx = 0;
                    }
                }
                // If we fall out of the levels, switch to the special array
                in_special_array = true;
                slot_idx = 0;
                while (slot_idx < map_ptr->special_array.size() && !map_ptr->special_array[slot_idx].is_occupied) {
                    slot_idx++;
                }
            }
        }

    public:
        using iterator_category = std::forward_iterator_tag;
        using difference_type = std::ptrdiff_t;
        using value_type = std::pair<const Key, Value>;
        using pointer = value_type*;
        using reference = value_type&;

        iterator(Funnel_Hash_Map* map, size_t l_idx, size_t s_idx, bool special)
            : map_ptr(map), level_idx(l_idx), slot_idx(s_idx), in_special_array(special) {}

        reference operator*() const {
            if (in_special_array) {
                return map_ptr->special_array[slot_idx].data;
            }
            return map_ptr->levels[level_idx][slot_idx].data;
        }

        pointer operator->() const {
            if (in_special_array) {
                return &map_ptr->special_array[slot_idx].data;
            }
            return &map_ptr->levels[level_idx][slot_idx].data;
        }

        iterator& operator++() {
            advance();
            return *this;
        }

        iterator operator++(int) {
            iterator temp = *this;
            advance();
            return temp;
        }

        bool operator==(const iterator& other) const {
            return map_ptr == other.map_ptr && level_idx == other.level_idx &&
                   slot_idx == other.slot_idx && in_special_array == other.in_special_array;
        }

        bool operator!=(const iterator& other) const {
            return !(*this == other);
        }
    };

    explicit Funnel_Hash_Map(size_t num_items_to_insert, double delta = 0.1) {
        if (num_items_to_insert == 0) num_items_to_insert = 1; // Handle 0 case
        if (!(delta > 0 && delta < 1)) {
            throw std::invalid_argument("delta must be between 0 and 1.");
        }

        this->delta = delta;
        this->num_inserts = 0;

        // Calculate the required total capacity to store the requested number of items
        // at the given load factor (1 - delta).
        double load_factor = 1.0 - delta;
        size_t required_capacity = static_cast<size_t>(std::ceil(static_cast<double>(num_items_to_insert) / load_factor));
        this->capacity = required_capacity;
        this->max_inserts = num_items_to_insert;

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

        size_t total_buckets = (primary_size > 0 && beta > 0) ? primary_size / beta : 0;
        double a1 = (alpha > 0 && total_buckets > 0) ?
                    static_cast<double>(total_buckets) / (4.0 * (1.0 - std::pow(0.75, alpha))) :
                    static_cast<double>(total_buckets);

        std::mt19937 rng(std::chrono::high_resolution_clock::now().time_since_epoch().count());
        std::uniform_int_distribution<size_t> dist(0, std::numeric_limits<size_t>::max());

        size_t remaining_buckets = total_buckets;
        for (size_t i = 0; i < alpha; ++i) {
            size_t a_i = std::max(1.0, std::round(a1 * std::pow(0.75, i)));
            if (remaining_buckets == 0 || a_i == 0) break;

            a_i = std::min(a_i, remaining_buckets);
            level_bucket_counts.push_back(a_i);

            size_t level_size = a_i * beta;
            levels.emplace_back(level_size);
            level_salts.push_back(dist(rng));

            remaining_buckets -= a_i;
        }

        if (remaining_buckets > 0 && !levels.empty()) {
            size_t extra_slots = remaining_buckets * beta;
            levels.back().resize(levels.back().size() + extra_slots);
            level_bucket_counts.back() += remaining_buckets;
        }

        special_array.resize(special_size);
        special_salt = dist(rng);
    }

    size_t size() const {
        return num_inserts;
    }

    bool insert(const std::pair<Key, Value>& pair) {
        return insert(pair.first, pair.second);
    }

    bool insert(const Key& key, const Value& value) {
        for (size_t i = 0; i < levels.size(); ++i) {
            if (level_bucket_counts[i] == 0) continue;

            size_t bucket_index = _hash_level(key, i) % level_bucket_counts[i];
            size_t start = bucket_index * beta;
            size_t end = start + beta;

            for (size_t idx = start; idx < end; ++idx) {
                if (!levels[i][idx].is_occupied) {
                    place_item(levels[i][idx], key, value);
                    return true;
                }
                if (levels[i][idx].data.first == key) {
                    place_item(levels[i][idx], key, value);
                    return true;
                }
            }
        }

        if (!special_array.empty()) {
            size_t special_size = special_array.size();
            for (size_t j = 0; j < special_size; ++j) {
                size_t idx = (_hash_special(key) + j) % special_size;
                if (!special_array[idx].is_occupied) {
                    place_item(special_array[idx], key, value);
                    return true;
                }
                if (special_array[idx].data.first == key) {
                    place_item(special_array[idx], key, value);
                    return true;
                }
            }
        }

        throw std::runtime_error("Insertion failed: the hash table is full.");
    }

    iterator find(const Key& key) {
        for (size_t i = 0; i < levels.size(); ++i) {
             if (level_bucket_counts[i] == 0) continue;

            size_t bucket_index = _hash_level(key, i) % level_bucket_counts[i];
            size_t start = bucket_index * beta;
            size_t end = start + beta;

            for (size_t idx = start; idx < end; ++idx) {
                const Slot& slot = levels[i][idx];
                if (slot.is_occupied && slot.data.first == key) {
                    return iterator(this, i, idx, false);
                }
                if (!slot.is_occupied && !slot.is_deleted) {
                    break;
                }
            }
        }

         if (!special_array.empty()) {
            size_t special_size = special_array.size();
            for (size_t j = 0; j < special_size; ++j) {
                size_t idx = (_hash_special(key) + j) % special_size;
                const Slot& slot = special_array[idx];
                if (slot.is_occupied && slot.data.first == key) {
                     return iterator(this, levels.size(), idx, true);
                }
                 if (!slot.is_occupied && !slot.is_deleted) {
                    break;
                }
            }
        }

        return end();
    }

    bool erase(const Key& key) {
        // Search primary levels
        for (size_t i = 0; i < levels.size(); ++i) {
            if (level_bucket_counts[i] == 0) continue;

            size_t bucket_index = _hash_level(key, i) % level_bucket_counts[i];
            size_t start = bucket_index * beta;
            size_t end = start + beta;

            for (size_t idx = start; idx < end; ++idx) {
                Slot& slot = levels[i][idx];
                if (slot.is_occupied && slot.data.first == key) {
                    slot.is_occupied = false;
                    slot.is_deleted = true;
                    num_inserts--;
                    return true;
                }
                if (!slot.is_occupied && !slot.is_deleted) {
                    break;
                }
            }
        }

        // Search special array
        if (!special_array.empty()) {
            size_t special_size = special_array.size();
            for (size_t j = 0; j < special_size; ++j) {
                size_t idx = (_hash_special(key) + j) % special_size;
                Slot& slot = special_array[idx];
                if (slot.is_occupied && slot.data.first == key) {
                    slot.is_occupied = false;
                    slot.is_deleted = true;
                    num_inserts--;
                    return true;
                }
                if (!slot.is_occupied && !slot.is_deleted) {
                    break;
                }
            }
        }

        return false; // Key not found
    }


    bool contains(const Key& key) {
        return find(key) != end();
    }

    iterator begin() {
        for (size_t i = 0; i < levels.size(); ++i) {
            for (size_t j = 0; j < levels[i].size(); ++j) {
                if (levels[i][j].is_occupied) {
                    return iterator(this, i, j, false);
                }
            }
        }
        for (size_t i = 0; i < special_array.size(); ++i) {
            if (special_array[i].is_occupied) {
                return iterator(this, levels.size(), i, true);
            }
        }
        return end();
    }

    iterator end() {
        return iterator(this, levels.size(), special_array.size(), true);
    }
};

#endif // FUNNEL_HASH_MAP_H

