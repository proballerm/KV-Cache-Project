#include <chrono>
#include <cctype>
#include <iostream>
#include <string>
#include <thread>

#include "httplib.h"
#include "json.hpp"

using json = nlohmann::json;

std::string to_lower(std::string text) {
    for (char& c : text) {
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return text;
}

int main() {
    httplib::Server app;

    app.Options(".*", [](const httplib::Request&, httplib::Response& res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_header("Access-Control-Allow-Credentials", "true");
        res.set_header("Access-Control-Allow-Methods", "*");
        res.set_header("Access-Control-Allow-Headers", "*");
        res.status = 204;
    });

    app.Get("/", [](const httplib::Request&, httplib::Response& res) {
        res.set_header("Access-Control-Allow-Origin", "*");

        json response = {
            {"status", "Backend is running"}
        };

        res.set_content(response.dump(), "application/json");
    });

    app.Post("/chat", [](const httplib::Request& req, httplib::Response& res) {
        auto start_time = std::chrono::high_resolution_clock::now();

        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_header("Access-Control-Allow-Credentials", "true");
        res.set_header("Access-Control-Allow-Methods", "*");
        res.set_header("Access-Control-Allow-Headers", "*");

        json request = json::parse(req.body);

        std::string message = request.value("message", "");
        bool use_cache = request.value("use_cache", false);

        if (use_cache) {
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(1800));
        }

        std::string user_message = to_lower(message);
        std::string npc_response;

        if (user_message.find("dragon") != std::string::npos) {
            npc_response =
                "The dragon guards the cave because it protects an ancient crystal that powers the entire valley.";
        } else if (user_message.find("cave") != std::string::npos) {
            npc_response =
                "The cave is old and filled with glowing stones. Many travelers enter, but not all return.";
        } else if (user_message.find("castle") != std::string::npos) {
            npc_response =
                "The castle gates are closed because strange lights appeared near the mountain last night.";
        } else if (user_message.find("treasure") != std::string::npos) {
            npc_response =
                "The real treasure is not gold. It is a memory stone that stores the forgotten history of this world.";
        } else {
            npc_response =
                "That is a great question. This world has many secrets, and your choices will shape what happens next.";
        }

        auto end_time = std::chrono::high_resolution_clock::now();

        int latency_ms = static_cast<int>(
            std::chrono::duration_cast<std::chrono::milliseconds>(
                end_time - start_time
            ).count()
        );

        json response = {
            {"response", npc_response},
            {"latency_ms", latency_ms},
            {"tokens_per_second", use_cache ? 55 : 15},
            {"use_cache", use_cache}
        };

        res.set_content(response.dump(), "application/json");
    });

    std::cout << "Backend is running on http://localhost:8000" << std::endl;
    app.listen("0.0.0.0", 8000);

    return 0;
}