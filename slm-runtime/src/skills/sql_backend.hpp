#pragma once

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace slm {

// Contract of the `sql` skill. The Manager passes only a natural-language
// instruction. Schema/connection context is owned by the Runtime and handed
// to the backend here; it is never copied into a TaskPlan. SQL generation,
// validation and execution are the backend's responsibility.
struct SqlContext {
    std::string schema_description;
};

struct SqlResponse {
    std::string provider;
    std::string instruction;
    std::string generated_sql;  // optional, filled by a real SQL skill
    std::vector<std::string> columns;
    std::vector<std::vector<std::string>> rows;
    bool is_mock = false;

    std::string to_text() const;
};

class SqlBackend {
public:
    virtual ~SqlBackend() = default;
    virtual SqlResponse run(const std::string& instruction,
                            const SqlContext& context) = 0;
    virtual const char* name() const noexcept = 0;
};

class SqlBackendUnavailable : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

std::unique_ptr<SqlBackend> make_unconfigured_sql_backend();
std::unique_ptr<SqlBackend> make_mock_sql_backend();

}  // namespace slm
