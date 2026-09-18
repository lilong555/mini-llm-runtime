#pragma once

#include <cmath>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace test {

struct Case {
    const char* name;
    std::function<void()> run;
};

inline std::vector<Case>& cases() {
    static std::vector<Case> value;
    return value;
}

struct Register {
    Register(const char* name, std::function<void()> run) { cases().push_back({name, std::move(run)}); }
};

inline void check(bool value, const char* expression, const char* file, int line) {
    if (!value) {
        throw std::runtime_error(std::string(file) + ":" + std::to_string(line) + ": " + expression);
    }
}

template<class Error, class Function>
void throws(Function&& function) {
    try {
        function();
    } catch (const Error&) {
        return;
    }
    throw std::runtime_error("expected exception was not thrown");
}

inline int run() {
    std::size_t failed = 0;
    for (const auto& item : cases()) {
        try {
            item.run();
            std::cout << "[PASS] " << item.name << '\n';
        } catch (const std::exception& error) {
            ++failed;
            std::cerr << "[FAIL] " << item.name << ": " << error.what() << '\n';
        }
    }
    std::cout << cases().size() - failed << "/" << cases().size() << " tests passed\n";
    return failed ? 1 : 0;
}

} // namespace test

#define CHECK(expression) test::check(static_cast<bool>(expression), #expression, __FILE__, __LINE__)
#define TEST(name) static void name(); static test::Register registration_##name(#name, name); static void name()
