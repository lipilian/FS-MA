#include "fs/core/Logger.hpp"

#include <iostream>

void Logger::log(std::string_view message) {
    std::cout << message << '\n';
}

void Logger::error(std::string_view message) {
    std::cerr << message << '\n';
}
