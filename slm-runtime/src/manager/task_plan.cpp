#include "manager/task_plan.hpp"

#include <charconv>
#include <cctype>
#include <cstdint>
#include <limits>
#include <map>
#include <sstream>
#include <string_view>
#include <unordered_set>
#include <utility>

namespace slm {

const char* plan_status_name(PlanStatus status) noexcept {
    switch (status) {
        case PlanStatus::Ok: return "ok";
        case PlanStatus::Clarify: return "clarify";
        case PlanStatus::Unsupported: return "unsupported";
    }
    return "ok";
}

const char* plan_error_code_name(PlanErrorCode code) noexcept {
    switch (code) {
        case PlanErrorCode::NoJsonObject: return "NO_JSON_OBJECT";
        case PlanErrorCode::JsonParseError: return "JSON_PARSE_ERROR";
        case PlanErrorCode::InvalidText: return "INVALID_TEXT";
        case PlanErrorCode::SchemaError: return "SCHEMA_ERROR";
        case PlanErrorCode::UnknownField: return "UNKNOWN_FIELD";
        case PlanErrorCode::MissingField: return "MISSING_FIELD";
        case PlanErrorCode::TypeError: return "TYPE_ERROR";
        case PlanErrorCode::InvalidStatus: return "INVALID_STATUS";
        case PlanErrorCode::StatusSemantics: return "STATUS_SEMANTICS";
        case PlanErrorCode::EmptyPlan: return "EMPTY_PLAN";
        case PlanErrorCode::TooManyTasks: return "TOO_MANY_TASKS";
        case PlanErrorCode::InvalidTaskId: return "INVALID_TASK_ID";
        case PlanErrorCode::InvalidSkill: return "INVALID_SKILL";
        case PlanErrorCode::EmptyInstruction: return "EMPTY_INSTRUCTION";
        case PlanErrorCode::EmptyMessage: return "EMPTY_MESSAGE";
        case PlanErrorCode::InvalidSource: return "INVALID_SOURCE";
        case PlanErrorCode::SelfReference: return "SELF_REFERENCE";
        case PlanErrorCode::FutureReference: return "FUTURE_REFERENCE";
        case PlanErrorCode::DuplicateSource: return "DUPLICATE_SOURCE";
        case PlanErrorCode::SourceCount: return "SOURCE_COUNT";
        case PlanErrorCode::EmptyOutputs: return "EMPTY_OUTPUTS";
        case PlanErrorCode::InvalidOutput: return "INVALID_OUTPUT";
        case PlanErrorCode::DuplicateOutput: return "DUPLICATE_OUTPUT";
    }
    return "UNKNOWN";
}

TaskPlanError::TaskPlanError(PlanErrorCode code, const std::string& message)
    : std::invalid_argument(std::string(plan_error_code_name(code)) + ": " +
                            message),
      code_(code) {}

namespace {

[[noreturn]] void reject(PlanErrorCode code, const std::string& message) {
    throw TaskPlanError(code, message);
}

// Strict UTF-8: no overlong forms, no surrogates, nothing above U+10FFFF.
bool valid_utf8(std::string_view text) {
    std::size_t index = 0;
    while (index < text.size()) {
        const auto lead = static_cast<unsigned char>(text[index]);
        std::size_t length = 0;
        std::uint32_t codepoint = 0;
        if (lead < 0x80) {
            ++index;
            continue;
        } else if (lead >= 0xc2 && lead <= 0xdf) {
            length = 2;
            codepoint = lead & 0x1fU;
        } else if (lead >= 0xe0 && lead <= 0xef) {
            length = 3;
            codepoint = lead & 0x0fU;
        } else if (lead >= 0xf0 && lead <= 0xf4) {
            length = 4;
            codepoint = lead & 0x07U;
        } else {
            return false;
        }
        if (index + length > text.size()) return false;
        for (std::size_t offset = 1; offset < length; ++offset) {
            const auto next = static_cast<unsigned char>(text[index + offset]);
            if ((next & 0xc0U) != 0x80U) return false;
            codepoint = (codepoint << 6U) | (next & 0x3fU);
        }
        if ((length == 3 && codepoint < 0x800) ||
            (length == 4 && (codepoint < 0x10000 || codepoint > 0x10ffff)) ||
            (codepoint >= 0xd800 && codepoint <= 0xdfff)) {
            return false;
        }
        index += length;
    }
    return true;
}

std::vector<std::uint32_t> codepoints(std::string_view text) {
    std::vector<std::uint32_t> result;
    std::size_t index = 0;
    while (index < text.size()) {
        const auto lead = static_cast<unsigned char>(text[index]);
        std::size_t length = lead < 0x80 ? 1 : lead < 0xe0 ? 2 : lead < 0xf0 ? 3 : 4;
        std::uint32_t value = length == 1 ? lead
                              : length == 2 ? (lead & 0x1fU)
                              : length == 3 ? (lead & 0x0fU)
                                            : (lead & 0x07U);
        for (std::size_t offset = 1; offset < length && index + offset < text.size(); ++offset) {
            value = (value << 6U) |
                    (static_cast<unsigned char>(text[index + offset]) & 0x3fU);
        }
        result.push_back(value);
        index += length;
    }
    return result;
}

bool is_space_codepoint(std::uint32_t value) {
    return (value >= 0x09 && value <= 0x0d) || value == 0x20 || value == 0xa0 ||
           value == 0x1680 || (value >= 0x2000 && value <= 0x200a) ||
           value == 0x2028 || value == 0x2029 || value == 0x202f ||
           value == 0x205f || value == 0x3000 || value == 0xfeff;
}

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
        throw TaskPlanError(PlanErrorCode::JsonParseError,
                            "invalid manager JSON at byte " +
                                std::to_string(position_) + ": " + reason);
    }

    // RFC 8259 whitespace only (std::isspace would also accept \v and \f).
    void skip_space() {
        while (position_ < input_.size() &&
               (input_[position_] == ' ' || input_[position_] == '\t' ||
                input_[position_] == '\n' || input_[position_] == '\r')) {
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


const JsonValue* field(const JsonValue& object, const char* name) {
    const auto found = object.object.find(name);
    return found == object.object.end() ? nullptr : &found->second;
}

const JsonValue& required(const JsonValue& object, const char* name,
                          JsonValue::Type type) {
    const JsonValue* value = field(object, name);
    if (value == nullptr) {
        reject(PlanErrorCode::MissingField,
               std::string("task plan is missing '") + name + "'");
    }
    if (value->type != type) {
        reject(PlanErrorCode::TypeError,
               std::string("task plan field '") + name + "' has the wrong type");
    }
    return *value;
}

bool out_of_int_range(std::int64_t value) {
    return value < 0 || value > std::numeric_limits<int>::max();
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

const std::unordered_set<std::string> kRootFields = {"status", "message",
                                                     "tasks", "outputs"};
const std::unordered_set<std::string> kTaskFields = {"id", "skill",
                                                     "instruction", "sources"};

}  // namespace

bool plan_text_is_invalid(const std::string& value) {
    for (const auto point : codepoints(value)) {
        if (point < 0x20 || point == 0x7f || (point >= 0x80 && point <= 0x9f) ||
            point == 0xfffd) {
            return true;
        }
    }
    return false;
}

bool plan_text_is_blank(const std::string& value) {
    for (const auto point : codepoints(value)) {
        if (!is_space_codepoint(point)) return false;
    }
    return true;
}

TaskPlan TaskPlan::ok(std::vector<Task> tasks, std::vector<int> outputs) {
    TaskPlan plan;
    plan.status = PlanStatus::Ok;
    plan.tasks = std::move(tasks);
    plan.outputs = std::move(outputs);
    return plan;
}

TaskPlan TaskPlan::clarify(std::string message) {
    TaskPlan plan;
    plan.status = PlanStatus::Clarify;
    plan.message = std::move(message);
    plan.has_message = true;
    return plan;
}

TaskPlan TaskPlan::unsupported(std::string message) {
    TaskPlan plan;
    plan.status = PlanStatus::Unsupported;
    plan.message = std::move(message);
    plan.has_message = true;
    return plan;
}

TaskPlan TaskPlan::from_manager_output(const std::string& output) {
    if (!valid_utf8(output)) {
        reject(PlanErrorCode::InvalidText, "manager output is not valid UTF-8");
    }
    const auto first = output.find('{');
    const auto last = output.rfind('}');
    if (first == std::string::npos || last == std::string::npos || last < first) {
        reject(PlanErrorCode::NoJsonObject, "manager output contains no JSON object");
    }
    const JsonValue root =
        JsonParser(std::string_view(output).substr(first, last - first + 1)).parse();
    if (root.type != JsonValue::Type::Object) {
        reject(PlanErrorCode::SchemaError, "task plan root must be an object");
    }
    for (const auto& [key, ignored] : root.object) {
        (void)ignored;
        if (!kRootFields.contains(key)) {
            reject(PlanErrorCode::UnknownField, "unknown task plan field: '" + key + "'");
        }
    }

    TaskPlan plan;
    const auto& status = required(root, "status", JsonValue::Type::String);
    if (status.string == "ok") {
        plan.status = PlanStatus::Ok;
    } else if (status.string == "clarify") {
        plan.status = PlanStatus::Clarify;
    } else if (status.string == "unsupported") {
        plan.status = PlanStatus::Unsupported;
    } else {
        reject(PlanErrorCode::InvalidStatus, "unknown status: '" + status.string + "'");
    }
    const auto& tasks = required(root, "tasks", JsonValue::Type::Array);
    const auto& outputs = required(root, "outputs", JsonValue::Type::Array);
    if (const JsonValue* message = field(root, "message")) {
        if (message->type != JsonValue::Type::String) {
            reject(PlanErrorCode::TypeError, "task plan field 'message' has the wrong type");
        }
        plan.message = message->string;
        plan.has_message = true;
    }

    plan.tasks.reserve(tasks.array.size());
    for (const auto& item : tasks.array) {
        if (item.type != JsonValue::Type::Object) {
            reject(PlanErrorCode::SchemaError, "each task must be an object");
        }
        for (const auto& [key, ignored] : item.object) {
            (void)ignored;
            if (!kTaskFields.contains(key)) {
                reject(PlanErrorCode::UnknownField, "unknown task field: '" + key + "'");
            }
        }
        const auto& id = required(item, "id", JsonValue::Type::Integer);
        if (out_of_int_range(id.integer)) {
            reject(PlanErrorCode::InvalidTaskId, "task id is out of range");
        }
        Task task;
        task.id = static_cast<int>(id.integer);
        task.skill = required(item, "skill", JsonValue::Type::String).string;
        task.instruction =
            required(item, "instruction", JsonValue::Type::String).string;
        const auto& sources = required(item, "sources", JsonValue::Type::Array);
        for (const auto& source : sources.array) {
            if (source.type == JsonValue::Type::String) {
                if (source.string != "user") {
                    reject(PlanErrorCode::InvalidSource,
                           "string source must be \"user\", got '" + source.string + "'");
                }
                task.sources.push_back(SourceRef::user());
            } else if (source.type == JsonValue::Type::Integer) {
                if (out_of_int_range(source.integer)) {
                    reject(PlanErrorCode::InvalidSource, "task source id is out of range");
                }
                task.sources.push_back(SourceRef::task(static_cast<int>(source.integer)));
            } else {
                reject(PlanErrorCode::TypeError,
                       "a source must be \"user\" or an integer task id");
            }
        }
        plan.tasks.push_back(std::move(task));
    }
    for (const auto& output : outputs.array) {
        if (output.type != JsonValue::Type::Integer) {
            reject(PlanErrorCode::TypeError, "outputs must contain integer task ids");
        }
        if (out_of_int_range(output.integer)) {
            reject(PlanErrorCode::InvalidOutput, "output task id is out of range");
        }
        plan.outputs.push_back(static_cast<int>(output.integer));
    }
    return plan;
}

void TaskPlan::validate(const SkillRegistry& registry, std::size_t max_tasks) const {
    if (status != PlanStatus::Ok) {
        if (!tasks.empty() || !outputs.empty()) {
            reject(PlanErrorCode::StatusSemantics,
                   std::string(plan_status_name(status)) +
                       " plan must have empty tasks and outputs");
        }
        if (!has_message) {
            reject(PlanErrorCode::StatusSemantics,
                   std::string(plan_status_name(status)) + " plan requires a message");
        }
        if (plan_text_is_invalid(message)) {
            reject(PlanErrorCode::InvalidText, "message contains invalid text");
        }
        if (plan_text_is_blank(message)) {
            reject(PlanErrorCode::EmptyMessage, "message must not be empty");
        }
        return;
    }
    if (has_message) {
        reject(PlanErrorCode::StatusSemantics, "ok plan must not carry a message");
    }
    if (tasks.empty()) {
        reject(PlanErrorCode::EmptyPlan, "ok plan must contain at least one task");
    }
    if (outputs.empty()) {
        reject(PlanErrorCode::EmptyOutputs, "ok plan must name at least one output");
    }
    if (tasks.size() > max_tasks) {
        reject(PlanErrorCode::TooManyTasks, "task plan exceeds the maximum task count");
    }
    for (std::size_t index = 0; index < tasks.size(); ++index) {
        const auto& task = tasks[index];
        if (task.id != static_cast<int>(index)) {
            reject(PlanErrorCode::InvalidTaskId,
                   "task ids must be unique and ordered from zero");
        }
        const SkillSpec* spec = registry.find(task.skill);
        if (spec == nullptr) {
            reject(PlanErrorCode::InvalidSkill, "unsupported skill: '" + task.skill + "'");
        }
        if (plan_text_is_invalid(task.instruction)) {
            reject(PlanErrorCode::InvalidText, "instruction contains invalid text");
        }
        if (plan_text_is_blank(task.instruction)) {
            reject(PlanErrorCode::EmptyInstruction, "task instruction must not be empty");
        }
        std::vector<SourceRef> seen;
        for (const auto& source : task.sources) {
            if (!source.is_user()) {
                if (source.task_id == task.id) {
                    reject(PlanErrorCode::SelfReference, "task references its own output");
                }
                if (source.task_id > task.id) {
                    reject(PlanErrorCode::FutureReference,
                           "source must reference an earlier task");
                }
            }
            for (const auto& previous : seen) {
                if (previous == source) {
                    reject(PlanErrorCode::DuplicateSource, "duplicate task source");
                }
            }
            seen.push_back(source);
        }
        if (task.sources.size() < spec->min_sources ||
            task.sources.size() > spec->max_sources) {
            reject(PlanErrorCode::SourceCount,
                   task.skill + " requires " + std::to_string(spec->min_sources) +
                       ".." + std::to_string(spec->max_sources) + " sources");
        }
    }
    std::vector<int> seen_outputs;
    for (const int output : outputs) {
        if (output >= static_cast<int>(tasks.size())) {
            reject(PlanErrorCode::InvalidOutput, "output references a missing task");
        }
        for (const int previous : seen_outputs) {
            if (previous == output) {
                reject(PlanErrorCode::DuplicateOutput, "duplicate output task id");
            }
        }
        seen_outputs.push_back(output);
    }
}

std::string TaskPlan::to_json(const SkillRegistry& registry) const {
    validate(registry);
    std::ostringstream json;
    json << "{\"status\":\"" << plan_status_name(status) << '"';
    if (status != PlanStatus::Ok) {
        json << ",\"message\":\"" << escape_json(message) << '"';
    }
    json << ",\"tasks\":[";
    for (std::size_t index = 0; index < tasks.size(); ++index) {
        const auto& task = tasks[index];
        if (index > 0) json << ',';
        json << "{\"id\":" << task.id << ",\"skill\":\"" << escape_json(task.skill)
             << "\",\"instruction\":\"" << escape_json(task.instruction)
             << "\",\"sources\":[";
        for (std::size_t source = 0; source < task.sources.size(); ++source) {
            if (source > 0) json << ',';
            if (task.sources[source].is_user()) {
                json << "\"user\"";
            } else {
                json << task.sources[source].task_id;
            }
        }
        json << "]}";
    }
    json << "],\"outputs\":[";
    for (std::size_t index = 0; index < outputs.size(); ++index) {
        if (index > 0) json << ',';
        json << outputs[index];
    }
    json << "]}";
    return json.str();
}

}  // namespace slm
