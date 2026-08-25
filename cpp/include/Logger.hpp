#pragma once

#include <string_view>

class Logger {
public:
    static void log(std::string_view message);
    static void error(std::string_view message);
};
