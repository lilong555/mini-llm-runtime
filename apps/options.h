#pragma once

#include <charconv>
#include <cstdint>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>

class Options {
public:
    Options(int argc, char** argv, std::set<std::string> values,
            std::set<std::string> flags = {"--help"}) {
        for (int i = 1; i < argc; ++i) {
            const std::string name = argv[i];
            if (arguments_.contains(name)) {
                throw std::invalid_argument("duplicate option: " + name);
            }
            if (flags.contains(name)) {
                arguments_[name] = "true";
            } else if (values.contains(name) && i + 1 < argc) {
                arguments_[name] = argv[++i];
            } else {
                throw std::invalid_argument("unknown option or missing value: " + name);
            }
        }
    }
    bool has(const std::string& key) const { return arguments_.contains(key); }
    std::string get(const std::string& key, const std::string& fallback = "") const {
        const auto value = arguments_.find(key);
        return value == arguments_.end() ? fallback : value->second;
    }
    std::int64_t integer(const std::string& key, std::int64_t fallback,
                         std::int64_t minimum, std::int64_t maximum) const {
        if (!has(key)) {
            return fallback;
        }
        const auto text = get(key);
        std::int64_t value = 0;
        const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
        if (error != std::errc{} || end != text.data() + text.size() ||
            value < minimum || value > maximum) {
            throw std::invalid_argument("out-of-range integer: " + key);
        }
        return value;
    }
private:
    std::map<std::string, std::string> arguments_;
};
