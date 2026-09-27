#include "manager/task_plan.hpp"

#include <charconv>
#include <cctype>
#include <cstdint>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace slm {
namespace {

struct JsonValue {
    enum class Type { Null, Boolean, Integer, String, Array, Object };

    Type type = Type::Null;
    bool boolean = false;
    std::int64_t integer = 0;
    std::string string;
    std::vector<JsonValue> array;
    std::map<std::string, JsonValue> object;
};

class JsonParser {
public:
    explicit JsonParser(std::string_view input) : input_(input) {}

    JsonValue parse() {
        auto result = parse_value(0);
        skip_space();
        if (position_ != input_.size()) {
            fail("unexpected trailing JSON content");
        }
        return result;
    }

private:
    [[noreturn]] void fail(const std::string& reason) const {
        throw std::invalid_argument("invalid manager JSON at byte " +
                                    std::to_string(position_) + ": " + reason);
    }

    void skip_space() {
        while (position_ < input_.size() &&
               std::isspace(static_cast<unsigned char>(input_[position_]))) {
            ++position_;
        }
    }

    bool consume(char expected) {
        skip_space();
        if (position_ < input_.size() && input_[position_] == expected) {
            ++position_;
            return true;
        }
        return false;
    }

    void expect(char expected) {
        if (!consume(expected)) {
            fail(std::string("expected '") + expected + "'");
        }
    }

    bool consume_literal(std::string_view literal) {
        skip_space();
        if (input_.substr(position_, literal.size()) == literal) {
            position_ += literal.size();
            return true;
        }
        return false;
    }

    JsonValue parse_value(std::size_t depth) {
        if (depth > 32) fail("JSON nesting is too deep");
        skip_space();
        if (position_ >= input_.size()) fail("unexpected end of input");

        if (input_[position_] == '{') return parse_object(depth + 1);
        if (input_[position_] == '[') return parse_array(depth + 1);
        if (input_[position_] == '"') {
            JsonValue value;
            value.type = JsonValue::Type::String;
            value.string = parse_string();
            return value;
        }
        if (input_[position_] == '-' ||
            std::isdigit(static_cast<unsigned char>(input_[position_]))) {
            return parse_integer();
        }
        if (consume_literal("true")) {
            JsonValue value;
            value.type = JsonValue::Type::Boolean;
            value.boolean = true;
            return value;
        }
        if (consume_literal("false")) {
            JsonValue value;
            value.type = JsonValue::Type::Boolean;
            return value;
        }
        if (consume_literal("null")) return {};
        fail("unexpected value");
    }

    JsonValue parse_object(std::size_t depth) {
        JsonValue value;
        value.type = JsonValue::Type::Object;
        expect('{');
        if (consume('}')) return value;
        while (true) {
            skip_space();
            if (position_ >= input_.size() || input_[position_] != '"') {
                fail("object key must be a string");
            }
            auto key = parse_string();
            expect(':');
            auto [_, inserted] =
                value.object.emplace(std::move(key), parse_value(depth));
            if (!inserted) fail("duplicate object key");
            if (consume('}')) break;
            expect(',');
        }
        return value;
    }

    JsonValue parse_array(std::size_t depth) {
        JsonValue value;
        value.type = JsonValue::Type::Array;
        expect('[');
        if (consume(']')) return value;
        while (true) {
            value.array.push_back(parse_value(depth));
            if (consume(']')) break;
            expect(',');
        }
        return value;
    }

    static int hex_digit(char value) {
        if (value >= '0' && value <= '9') return value - '0';
        if (value >= 'a' && value <= 'f') return value - 'a' + 10;
        if (value >= 'A' && value <= 'F') return value - 'A' + 10;
        return -1;
    }

    std::uint32_t parse_hex_quad() {
        if (position_ + 4 > input_.size()) fail("incomplete unicode escape");
        std::uint32_t result = 0;
        for (int index = 0; index < 4; ++index) {
            const int digit = hex_digit(input_[position_++]);
            if (digit < 0) fail("invalid unicode escape");
            result = (result << 4U) | static_cast<std::uint32_t>(digit);
        }
        return result;
    }

    static void append_utf8(std::string& output, std::uint32_t codepoint) {
        if (codepoint <= 0x7f) {
            output.push_back(static_cast<char>(codepoint));
        } else if (codepoint <= 0x7ff) {
            output.push_back(static_cast<char>(0xc0 | (codepoint >> 6)));
            output.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
        } else if (codepoint <= 0xffff) {
            output.push_back(static_cast<char>(0xe0 | (codepoint >> 12)));
            output.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3f)));
            output.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
        } else {
            output.push_back(static_cast<char>(0xf0 | (codepoint >> 18)));
            output.push_back(static_cast<char>(0x80 | ((codepoint >> 12) & 0x3f)));
            output.push_back(static_cast<char>(0x80 | ((codepoint >> 6) & 0x3f)));
            output.push_back(static_cast<char>(0x80 | (codepoint & 0x3f)));
        }
    }

    std::string parse_string() {
        expect('"');
        std::string result;
        while (position_ < input_.size()) {
            const unsigned char character =
                static_cast<unsigned char>(input_[position_++]);
            if (character == '"') return result;
            if (character < 0x20) fail("unescaped control character");
            if (character != '\\') {
                result.push_back(static_cast<char>(character));
                continue;
            }
            if (position_ >= input_.size()) fail("incomplete string escape");
            const char escaped = input_[position_++];
            switch (escaped) {
                case '"': result.push_back('"'); break;
                case '\\': result.push_back('\\'); break;
                case '/': result.push_back('/'); break;
                case 'b': result.push_back('\b'); break;
                case 'f': result.push_back('\f'); break;
                case 'n': result.push_back('\n'); break;
                case 'r': result.push_back('\r'); break;
                case 't': result.push_back('\t'); break;
                case 'u': {
                    std::uint32_t codepoint = parse_hex_quad();
                    if (codepoint >= 0xd800 && codepoint <= 0xdbff) {
                        if (position_ + 2 > input_.size() ||
                            input_[position_] != '\\' ||
                            input_[position_ + 1] != 'u') {
                            fail("missing low unicode surrogate");
                        }
                        position_ += 2;
                        const std::uint32_t low = parse_hex_quad();
                        if (low < 0xdc00 || low > 0xdfff) {
                            fail("invalid low unicode surrogate");
                        }
                        codepoint = 0x10000 +
                                    ((codepoint - 0xd800) << 10U) +
                                    (low - 0xdc00);
                    } else if (codepoint >= 0xdc00 && codepoint <= 0xdfff) {
                        fail("unexpected low unicode surrogate");
                    }
                    append_utf8(result, codepoint);
                    break;
                }
                default: fail("unsupported string escape");
            }
        }
        fail("unterminated string");
    }

    JsonValue parse_integer() {
        skip_space();
        const std::size_t start = position_;
        if (input_[position_] == '-') ++position_;
        if (position_ >= input_.size() ||
            !std::isdigit(static_cast<unsigned char>(input_[position_]))) {
            fail("invalid integer");
        }
        if (input_[position_] == '0') {
            ++position_;
        } else {
            while (position_ < input_.size() &&
                   std::isdigit(static_cast<unsigned char>(input_[position_]))) {
                ++position_;
            }
        }
        if (position_ < input_.size() &&
            (input_[position_] == '.' || input_[position_] == 'e' ||
             input_[position_] == 'E')) {
            fail("task plan numbers must be integers");
        }
        JsonValue value;
        value.type = JsonValue::Type::Integer;
        const auto text = input_.substr(start, position_ - start);
        const auto conversion = std::from_chars(
            text.data(), text.data() + text.size(), value.integer);
        if (conversion.ec != std::errc{} ||
            conversion.ptr != text.data() + text.size()) {
            fail("integer is out of range");
        }
        return value;
    }

    std::string_view input_;
    std::size_t position_ = 0;
};

const JsonValue& required_field(const JsonValue& object, const char* name,
                                JsonValue::Type type) {
    const auto found = object.object.find(name);
    if (found == object.object.end()) {
        throw std::invalid_argument(std::string("task plan is missing '") + name +
                                    "'");
    }
    if (found->second.type != type) {
        throw std::invalid_argument(std::string("task plan field '") + name +
                                    "' has the wrong type");
    }
    return found->second;
}

bool blank(const std::string& value) {
    for (const unsigned char character : value) {
        if (!std::isspace(character)) return false;
    }
    return true;
}

std::string escape_json(const std::string& value) {
    std::ostringstream result;
    for (const unsigned char character : value) {
        switch (character) {
            case '\\': result << "\\\\"; break;
            case '"': result << "\\\""; break;
            case '\b': result << "\\b"; break;
            case '\f': result << "\\f"; break;
            case '\n': result << "\\n"; break;
            case '\r': result << "\\r"; break;
            case '\t': result << "\\t"; break;
            default:
                if (character < 0x20) {
                    constexpr char digits[] = "0123456789abcdef";
                    result << "\\u00" << digits[character >> 4U]
                           << digits[character & 0x0fU];
                } else {
                    result << static_cast<char>(character);
                }
        }
    }
    return result.str();
}

}  // namespace

TaskPlan TaskPlan::from_manager_output(const std::string& output) {
    const auto first = output.find('{');
    const auto last = output.rfind('}');
    if (first == std::string::npos || last == std::string::npos || last < first) {
        throw std::invalid_argument("manager output contains no JSON object");
    }

    const JsonValue root =
        JsonParser(std::string_view(output).substr(first, last - first + 1)).parse();
    if (root.type != JsonValue::Type::Object) {
        throw std::invalid_argument("task plan root must be an object");
    }
    if (root.object.size() != 1 || !root.object.contains("tasks")) {
        throw std::invalid_argument("task plan root may contain only 'tasks'");
    }
    const JsonValue& tasks =
        required_field(root, "tasks", JsonValue::Type::Array);

    TaskPlan plan;
    plan.tasks.reserve(tasks.array.size());
    for (const auto& item : tasks.array) {
        if (item.type != JsonValue::Type::Object) {
            throw std::invalid_argument("each task must be an object");
        }
        for (const auto& [key, ignored] : item.object) {
            (void)ignored;
            if (key != "id" && key != "skill" && key != "input" &&
                key != "input_from") {
                throw std::invalid_argument("unknown task field: '" + key + "'");
            }
        }

        const auto& id = required_field(item, "id", JsonValue::Type::Integer);
        const auto& skill =
            required_field(item, "skill", JsonValue::Type::String);
        if (id.integer < 0 ||
            id.integer > std::numeric_limits<int>::max()) {
            throw std::invalid_argument("task id is out of range");
        }
        Task task;
        task.id = static_cast<int>(id.integer);
        task.skill = skill.string;
        if (const auto found = item.object.find("input");
            found != item.object.end()) {
            if (found->second.type != JsonValue::Type::String) {
                throw std::invalid_argument("task input must be a string");
            }
            task.input = found->second.string;
        }
        if (const auto found = item.object.find("input_from");
            found != item.object.end()) {
            if (found->second.type != JsonValue::Type::Integer ||
                found->second.integer < 0 ||
                found->second.integer > std::numeric_limits<int>::max()) {
                throw std::invalid_argument(
                    "task input_from must be a non-negative integer");
            }
            task.input_from = static_cast<int>(found->second.integer);
        }
        plan.tasks.push_back(std::move(task));
    }
    return plan;
}

void TaskPlan::validate(
    const std::unordered_set<std::string>& available_skills,
    std::size_t max_tasks) const {
    if (tasks.empty()) {
        throw std::invalid_argument("task plan must contain at least one task");
    }
    if (tasks.size() > max_tasks) {
        throw std::invalid_argument("task plan exceeds the maximum task count");
    }
    for (std::size_t index = 0; index < tasks.size(); ++index) {
        const auto& task = tasks[index];
        if (task.id != static_cast<int>(index)) {
            throw std::invalid_argument(
                "task ids must be unique and ordered from zero");
        }
        if (!available_skills.contains(task.skill)) {
            throw std::invalid_argument("unsupported skill: '" + task.skill + "'");
        }
        if (task.input.has_value() == task.input_from.has_value()) {
            throw std::invalid_argument(
                "each task requires exactly one of input or input_from");
        }
        if (task.input && blank(*task.input)) {
            throw std::invalid_argument("task input must not be empty");
        }
        if (task.input_from &&
            (*task.input_from < 0 || *task.input_from >= task.id)) {
            throw std::invalid_argument(
                "input_from must reference an earlier task");
        }
    }
}

std::string TaskPlan::to_json() const {
    validate();
    std::ostringstream json;
    json << "{\"tasks\":[";
    for (std::size_t index = 0; index < tasks.size(); ++index) {
        if (index > 0) json << ',';
        const auto& task = tasks[index];
        json << "{\"id\":" << task.id << ",\"skill\":\""
             << escape_json(task.skill) << "\",";
        if (task.input) {
            json << "\"input\":\"" << escape_json(*task.input) << "\"}";
        } else {
            json << "\"input_from\":" << *task.input_from << '}';
        }
    }
    json << "]}";
    return json.str();
}

}  // namespace slm
