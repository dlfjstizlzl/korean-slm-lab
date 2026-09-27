#pragma once

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace slm {

// Contract of the `search` skill: external information retrieval.
// The query is the Manager-written task instruction; search takes no sources.
// No real provider is connected in this repository. `MockSearchBackend` is a
// deterministic test double and says so in every result.
struct SearchDocument {
    std::string title;
    std::string snippet;
    std::string url;
};

struct SearchResponse {
    std::string provider;
    std::string query;
    std::vector<SearchDocument> documents;
    bool is_mock = false;

    // Deterministic plain-text rendering handed to downstream skills.
    std::string to_text() const;
};

class SearchBackend {
public:
    virtual ~SearchBackend() = default;
    virtual SearchResponse search(const std::string& query) = 0;
    virtual const char* name() const noexcept = 0;
};

class SearchBackendUnavailable : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

// Default: fails loudly instead of inventing results.
std::unique_ptr<SearchBackend> make_unconfigured_search_backend();
// Deterministic, offline, clearly labelled mock results.
std::unique_ptr<SearchBackend> make_mock_search_backend();

}  // namespace slm
