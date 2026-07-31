#pragma once

#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace thiran
{

using String = std::string;

template<typename T>
using Vector = std::vector<T>;

template<typename K, typename V>
using HashMap = std::unordered_map<K, V>;

template<typename T>
using HashSet = std::unordered_set<T>;

template<typename T>
using Ptr = std::shared_ptr<T>;

}