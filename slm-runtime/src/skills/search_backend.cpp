#include "skills/search_backend.hpp"

#include <functional>
#include <sstream>
#include <stdexcept>

namespace slm {

std::string SearchResponse::to_text() const {
    std::ostringstream text;
    text << "[search provider=" << provider << (is_mock ? " MOCK" : "")
         << "] query: " << query << '\n';
    for (std::size_t index = 0; index < documents.size(); ++index) {
        const auto& document = documents[index];
        text << (index + 1) << ". " << document.title << " — " << document.snippet;
        if (!document.url.empty()) text << " (" << document.url << ')';
        text << '\n';
    }
    return text.str();
}

namespace {

class UnconfiguredSearchBackend final : public SearchBackend {
public:
    SearchResponse search(const std::string&) override {
        throw SearchBackendUnavailable(
            "search skill has no retrieval provider configured; pass "
            "--search-backend mock for offline orchestration tests");
    }
    const char* name() const noexcept override { return "unconfigured"; }
};

class MockSearchBackend final : public SearchBackend {
public:
    SearchResponse search(const std::string& query) override {
        if (query.empty()) throw std::invalid_argument("search query must not be empty");
        const auto key = std::hash<std::string>{}(query) % 100000;
        SearchResponse response;
        response.provider = name();
        response.query = query;
        response.is_mock = true;
        for (int rank = 1; rank <= 2; ++rank) {
            response.documents.push_back(
                {"mock result " + std::to_string(rank) + " for: " + query,
                 "deterministic placeholder text; no real retrieval was performed",
                 "mock://search/" + std::to_string(key) + "/" + std::to_string(rank)});
        }
        return response;
    }
    const char* name() const noexcept override { return "mock"; }
};

}  // namespace

std::unique_ptr<SearchBackend> make_unconfigured_search_backend() {
    return std::make_unique<UnconfiguredSearchBackend>();
}

std::unique_ptr<SearchBackend> make_mock_search_backend() {
    return std::make_unique<MockSearchBackend>();
}

}  // namespace slm
