#include "skills/sql_backend.hpp"

#include <functional>
#include <sstream>

namespace slm {

std::string SqlResponse::to_text() const {
    std::ostringstream text;
    text << "[sql provider=" << provider << (is_mock ? " MOCK" : "")
         << "] request: " << instruction << '\n';
    if (!generated_sql.empty()) text << "sql: " << generated_sql << '\n';
    for (std::size_t index = 0; index < columns.size(); ++index) {
        text << (index ? " | " : "") << columns[index];
    }
    text << '\n';
    for (const auto& row : rows) {
        for (std::size_t index = 0; index < row.size(); ++index) {
            text << (index ? " | " : "") << row[index];
        }
        text << '\n';
    }
    return text.str();
}

namespace {

class UnconfiguredSqlBackend final : public SqlBackend {
public:
    SqlResponse run(const std::string&, const SqlContext&) override {
        throw SqlBackendUnavailable(
            "sql skill has no database backend configured; pass "
            "--sql-backend mock for offline orchestration tests");
    }
    const char* name() const noexcept override { return "unconfigured"; }
};

class MockSqlBackend final : public SqlBackend {
public:
    SqlResponse run(const std::string& instruction, const SqlContext& context) override {
        if (instruction.empty()) {
            throw std::invalid_argument("sql instruction must not be empty");
        }
        SqlResponse response;
        response.provider = name();
        response.instruction = instruction;
        response.is_mock = true;
        response.columns = {"row", "value", "schema_context"};
        const auto key = std::hash<std::string>{}(instruction) % 1000;
        const std::string schema = context.schema_description.empty()
                                       ? "none"
                                       : "provided(" +
                                             std::to_string(context.schema_description.size()) +
                                             " bytes)";
        for (int row = 1; row <= 3; ++row) {
            response.rows.push_back({std::to_string(row),
                                     "mock-" + std::to_string(key + row), schema});
        }
        return response;
    }
    const char* name() const noexcept override { return "mock"; }
};

}  // namespace

std::unique_ptr<SqlBackend> make_unconfigured_sql_backend() {
    return std::make_unique<UnconfiguredSqlBackend>();
}

std::unique_ptr<SqlBackend> make_mock_sql_backend() {
    return std::make_unique<MockSqlBackend>();
}

}  // namespace slm
