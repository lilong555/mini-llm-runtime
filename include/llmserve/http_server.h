#pragma once

#include <string>

namespace llmserve {

class Engine;
bool serve_http(Engine& engine, int port, const std::string& shutdown_file);

} // namespace llmserve
