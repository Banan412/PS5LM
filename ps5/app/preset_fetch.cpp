// Hugging Face model-card preset parser and local sidecar cache for PS5LM.
// SPDX-License-Identifier: GPL-3.0-or-later
#include "preset_fetch.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <regex>
#include <sstream>

namespace ps5lm {

namespace {

std::string to_lower(std::string s) {
    for (char & c : s) {
        c = (char) std::tolower((unsigned char) c);
    }
    return s;
}

std::string trim(const std::string & s) {
    size_t a = 0;
    while (a < s.size() && std::isspace((unsigned char) s[a])) {
        ++a;
    }
    size_t b = s.size();
    while (b > a && std::isspace((unsigned char) s[b - 1])) {
        --b;
    }
    return s.substr(a, b - a);
}

std::string fmt_num(double v) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.4f", v);
    std::string s(buf);
    while (s.size() > 1 && s.find('.') != std::string::npos && s.back() == '0') {
        s.pop_back();
    }
    if (!s.empty() && s.back() == '.') {
        s.push_back('0');
    }
    return s;
}

std::string file_name_of(const std::string & path) {
    const size_t pos = path.find_last_of("/\\");
    return pos == std::string::npos ? path : path.substr(pos + 1);
}

std::string repo_display_label(const std::string & repo_id) {
    if (repo_id.empty()) {
        return {};
    }
    const size_t slash = repo_id.find('/');
    std::string  name  = slash == std::string::npos ? repo_id : repo_id.substr(slash + 1);
    for (const char * suf : { "-GGUF", "-gguf", "-i1-GGUF", "-IMATRIX-MAX-MTP-GGUF" }) {
        const size_t len = std::strlen(suf);
        if (name.size() > len && name.compare(name.size() - len, len, suf) == 0) {
            name.erase(name.size() - len);
            break;
        }
    }
    if (name.size() > 48) {
        name.resize(45);
        name += "...";
    }
    return name;
}

std::string strip_html_to_text(const std::string & raw) {
    std::string out;
    out.reserve(raw.size());
    bool in_tag = false;
    std::string tag;
    for (size_t i = 0; i < raw.size(); ++i) {
        char c = raw[i];
        if (c == '<') {
            in_tag = true;
            tag.clear();
            continue;
        }
        if (in_tag) {
            if (c == '>') {
                in_tag = false;
                const std::string lt = to_lower(tag);
                if (lt.rfind("br", 0) == 0 || lt.rfind("/p", 0) == 0 || lt.rfind("/tr", 0) == 0 ||
                    lt.rfind("/li", 0) == 0 || lt.rfind("/h", 0) == 0 || lt.rfind("/div", 0) == 0) {
                    out.push_back('\n');
                } else {
                    out.push_back(' ');
                }
            } else if (tag.size() < 24) {
                tag.push_back(c);
            }
            continue;
        }
        if (c == '&') {
            if (raw.compare(i, 6, "&nbsp;") == 0) {
                out.push_back(' ');
                i += 5;
                continue;
            }
            if (raw.compare(i, 5, "&amp;") == 0) {
                out.push_back('&');
                i += 4;
                continue;
            }
            if (raw.compare(i, 4, "&lt;") == 0) {
                out.push_back('<');
                i += 3;
                continue;
            }
            if (raw.compare(i, 4, "&gt;") == 0) {
                out.push_back('>');
                i += 3;
                continue;
            }
            if (raw.compare(i, 6, "&quot;") == 0) {
                out.push_back('"');
                i += 5;
                continue;
            }
        }
        if (c != '\r') {
            out.push_back(c);
        }
    }
    return out;
}

// Removes bracketed side notes like "[1 to 1.1 suggested]" or "`" markers while preserving values.
std::string strip_bracket_notes(const std::string & s) {
    std::string out;
    out.reserve(s.size());
    int depth = 0;
    for (char c : s) {
        if (c == '[') {
            ++depth;
            continue;
        }
        if (c == ']') {
            if (depth > 0) {
                --depth;
            }
            continue;
        }
        if (depth == 0 && c != '`') {
            out.push_back(c);
        }
    }
    return out;
}

bool parse_double_token(const std::string & s, double * val) {
    if (s.empty()) {
        return false;
    }
    char * end = nullptr;
    const double v = std::strtod(s.c_str(), &end);
    if (end == s.c_str() || !std::isfinite(v)) {
        return false;
    }
    *val = v;
    return true;
}

struct SamplingMatch {
    bool   has_temp             = false;
    double temp                 = 0.0;
    bool   has_top_p            = false;
    double top_p                = 0.0;
    bool   has_top_k            = false;
    int    top_k                = 0;
    bool   has_min_p            = false;
    double min_p                = 0.0;
    bool   has_repeat_penalty   = false;
    double repeat_penalty       = 1.0;
    bool   has_presence_penalty = false;
    double presence_penalty     = 0.0;

    int key_count() const {
        return (int) has_temp + (int) has_top_p + (int) has_top_k + (int) has_min_p +
               (int) has_repeat_penalty + (int) has_presence_penalty;
    }
};

SamplingMatch extract_sampling_from_text(const std::string & raw_line) {
    SamplingMatch m;
    const std::string cleaned = strip_bracket_notes(raw_line);
    // Match "<key><sep><num1> [to|- <num2>]" where <num> allows leading dot like ".95".
    static const std::regex re(
        R"(\b(temperature|temp|top[_\s-]?p|topp|top[_\s-]?k|topk|min[_\s-]?p|minp|rep(?:etition|eat)?[_\s-]?pen(?:alty)?|reppen|presence[_\s-]?pen(?:alty)?)\b\s*(?:=|:|\s)\s*([0-9]*\.?[0-9]+)(?:\s*(?:to|-)\s*([0-9]*\.?[0-9]+))?)",
        std::regex::icase);

    std::sregex_iterator it(cleaned.begin(), cleaned.end(), re);
    std::sregex_iterator end;
    for (; it != end; ++it) {
        const std::string key = to_lower((*it)[1].str());
        double v1 = 0.0, v2 = 0.0;
        if (!parse_double_token((*it)[2].str(), &v1)) {
            continue;
        }
        const bool has_v2 = (*it)[3].matched && parse_double_token((*it)[3].str(), &v2);

        if (key == "temp" || key == "temperature") {
            // Range like "Temp .5 to 1": prefer upper bound <= 1.0 (matches general chat guidance).
            const double v = (has_v2 && v2 > v1 && v2 <= 1.0) ? v2 : v1;
            if (v >= 0.0 && v <= 2.0 && !m.has_temp) {
                m.has_temp = true;
                m.temp     = v;
            }
        } else if (key.find("top") == 0 && key.back() == 'p') {
            if (v1 >= 0.0 && v1 <= 1.0 && !m.has_top_p) {
                m.has_top_p = true;
                m.top_p     = v1;
            }
        } else if (key.find("top") == 0 && key.back() == 'k') {
            const int iv = (int) std::llround(v1);
            if (iv >= 0 && iv <= 250 && !m.has_top_k) {
                m.has_top_k = true;
                m.top_k     = iv;
            }
        } else if (key.find("min") == 0) {
            if (v1 >= 0.0 && v1 <= 1.0 && !m.has_min_p) {
                m.has_min_p = true;
                m.min_p     = v1;
            }
        } else if (key.find("rep") == 0) {
            if (v1 >= 0.5 && v1 <= 2.0 && !m.has_repeat_penalty) {
                m.has_repeat_penalty = true;
                m.repeat_penalty     = v1;
            }
        } else if (key.find("pres") == 0) {
            if (v1 >= 0.0 && v1 <= 2.5 && !m.has_presence_penalty) {
                m.has_presence_penalty = true;
                m.presence_penalty     = v1;
            }
        }
    }
    return m;
}

int score_sampling_line(const std::string & line_lower, const SamplingMatch & m) {
    const int k = m.key_count();
    if (k < 2 && !(k == 1 && (line_lower.find("suggested") != std::string::npos ||
                              line_lower.find("setting") != std::string::npos))) {
        return -1000;
    }
    int score = k * 10;
    if (line_lower.find("thinking") != std::string::npos &&
        line_lower.find("non-thinking") == std::string::npos &&
        line_lower.find("instruct") == std::string::npos) {
        score += 18;
    }
    if (line_lower.find("general tasks") != std::string::npos ||
        line_lower.find("text tasks") != std::string::npos ||
        line_lower.find("suggested settings") != std::string::npos ||
        line_lower.find("tester settings") != std::string::npos ||
        line_lower.find("settings:") != std::string::npos) {
        score += 14;
    }
    if (line_lower.find("non-thinking") != std::string::npos ||
        line_lower.find("instruct") != std::string::npos ||
        line_lower.find("vl tasks") != std::string::npos ||
        line_lower.find("precise coding") != std::string::npos) {
        score -= 16;
    }
    if (line_lower.find("example") != std::string::npos ||
        line_lower.find("smoothing") != std::string::npos ||
        line_lower.find("experiment") != std::string::npos) {
        score -= 12;
    }
    return score;
}

void push_unique_mode(std::vector<std::string> & modes, const std::string & mode) {
    if (!mode.empty() && std::find(modes.begin(), modes.end(), mode) == modes.end()) {
        modes.push_back(mode);
    }
}

void extract_thinking_modes(const std::string & text, const std::string & jinja, FetchedPreset & out) {
    const std::string combined = to_lower(text + "\n" + jinja);
    if (combined.find("enable_thinking") != std::string::npos ||
        combined.find("reasoning_effort") != std::string::npos ||
        combined.find("{reason:") != std::string::npos ||
        combined.find("reasoning modes") != std::string::npos) {
        push_unique_mode(out.thinking_modes, "off");
        push_unique_mode(out.thinking_modes, "low");
        push_unique_mode(out.thinking_modes, "medium");
        push_unique_mode(out.thinking_modes, "high");
    }
    if (combined.find("xhigh") != std::string::npos) {
        push_unique_mode(out.thinking_modes, "xhigh");
        if (combined.find("default_effort = 'xhigh'") != std::string::npos ||
            combined.find("default reasoning mode is \"xhigh\"") != std::string::npos ||
            combined.find("default is qwen's \"xhigh\"") != std::string::npos ||
            combined.find("\"reasoning\" is set at \"xhigh\"") != std::string::npos) {
            out.default_reasoning_effort = "xhigh";
        }
    }
    for (const char * special : {
             "einstein", "spoon", "deeptree", "socrates", "logic", "omni", "hyper", "ultra",
             "xultra", "007", "coder", "xomni", "xdeeptree", "xhyper", "xsocrates", "xlogic",
             "xeinstein", "xspoon" }) {
        const std::string sp(special);
        if (combined.find("'" + sp + "'") != std::string::npos ||
            combined.find("\"" + sp + "\"") != std::string::npos ||
            combined.find("`" + sp + "`") != std::string::npos ||
            combined.find("{reason:" + sp) != std::string::npos) {
            push_unique_mode(out.thinking_modes, special);
        }
    }
}

std::string expand_token_abbrev(const std::string & tok) {
    const std::string low = to_lower(tok);
    if (low == "uncnr" || low == "uncn" || low == "uncens") {
        return "Uncensored";
    }
    if (low == "brilli" || low == "brill") {
        return "Brilliance";
    }
    if (low == "inst") {
        return "Instruct";
    }
    if (low == "think") {
        return "Thinking";
    }
    return tok;
}

}  // namespace

std::string clean_gguf_search_stem(const std::string & file_name) {
    std::string stem = file_name_of(file_name);
    const std::string low = to_lower(stem);
    if (low.size() > 5 && low.compare(low.size() - 5, 5, ".gguf") == 0) {
        stem.resize(stem.size() - 5);
    }
    // Strip split suffix "-00001-of-00004"
    static const std::regex split_re(R"(-[0-9]{5}-of-[0-9]{5}$)", std::regex::icase);
    stem = std::regex_replace(stem, split_re, "");
    // Strip trailing quant tag (with or without UD-/i1-) like .Q8_0, -Q4_K_M, -UD-IQ2_XXS
    static const std::regex quant_re(
        R"([._-](?:(?:UD|i1|IQ|BF|FP)[._-]?)?(?:(?:I?Q\d+(?:_[KMS0-9]+)*)|(?:F(?:16|32))|(?:BF16)|(?:PTQ\d+_\d+)|(?:PQ\d+_\d+))$)",
        std::regex::icase);
    stem = std::regex_replace(stem, quant_re, "");
    return trim(stem);
}

std::vector<std::string> build_hf_search_queries(const std::string & file_name) {
    std::vector<std::string> out;
    auto push_unique = [&](const std::string & q) {
        const std::string t = trim(q);
        if (!t.empty() && std::find(out.begin(), out.end(), t) == out.end()) {
            out.push_back(t);
        }
    };

    const std::string stem = clean_gguf_search_stem(file_name);
    if (stem.empty()) {
        return out;
    }
    push_unique(stem);

    // Split hyphenated parts and build an abbreviation-expanded query + shorter prefixes.
    std::vector<std::string> parts;
    std::vector<std::string> expanded;
    {
        std::string cur;
        for (char c : stem) {
            if (c == '-' || c == '_') {
                if (!cur.empty()) {
                    parts.push_back(cur);
                    expanded.push_back(expand_token_abbrev(cur));
                    cur.clear();
                }
            } else {
                cur.push_back(c);
            }
        }
        if (!cur.empty()) {
            parts.push_back(cur);
            expanded.push_back(expand_token_abbrev(cur));
        }
    }

    auto join_prefix = [](const std::vector<std::string> & toks, size_t n, char sep) {
        std::string s;
        for (size_t i = 0; i < std::min(n, toks.size()); ++i) {
            if (i > 0) {
                s.push_back(sep);
            }
            s += toks[i];
        }
        return s;
    };

    if (expanded != parts) {
        push_unique(join_prefix(expanded, expanded.size(), '-'));
    }
    if (expanded.size() > 6) {
        push_unique(join_prefix(expanded, 6, '-'));
    }
    if (expanded.size() > 5) {
        push_unique(join_prefix(expanded, 5, '-'));
    }
    if (expanded.size() > 4) {
        push_unique(join_prefix(expanded, 4, '-'));
    }
    if (expanded.size() > 3) {
        push_unique(join_prefix(expanded, 3, ' '));
    }
    return out;
}

FetchedPreset parse_hf_model_metadata(const std::string & repo_id, const std::string & gen_config_json,
                                      const std::string & readme_md, const std::string & chat_template_jinja) {
    FetchedPreset out;
    out.checked     = true;
    out.source_repo = repo_id;
    out.label       = repo_display_label(repo_id);

    // 1. Baseline from generation_config.json (if present)
    if (!gen_config_json.empty()) {
        const nlohmann::json j = nlohmann::json::parse(gen_config_json, nullptr, false);
        if (j.is_object()) {
            auto take_f = [&](const char * k, bool & has, double & dst, double lo, double hi) {
                if (j.contains(k) && j[k].is_number()) {
                    const double v = j[k].get<double>();
                    if (std::isfinite(v) && v >= lo && v <= hi) {
                        has = true;
                        dst = v;
                    }
                }
            };
            take_f("temperature", out.has_temp, out.temp, 0.0, 2.0);
            take_f("top_p", out.has_top_p, out.top_p, 0.0, 1.0);
            take_f("min_p", out.has_min_p, out.min_p, 0.0, 1.0);
            take_f("repetition_penalty", out.has_repeat_penalty, out.repeat_penalty, 0.5, 2.0);
            take_f("repeat_penalty", out.has_repeat_penalty, out.repeat_penalty, 0.5, 2.0);
            take_f("presence_penalty", out.has_presence_penalty, out.presence_penalty, 0.0, 2.5);
            if (j.contains("top_k") && j["top_k"].is_number()) {
                const int k = (int) std::llround(j["top_k"].get<double>());
                if (k >= 0 && k <= 250) {
                    out.has_top_k = true;
                    out.top_k     = k;
                }
            }
        }
    }

    // 2. Override / enrich from author's README.md before any sample generation dump.
    const std::string text = strip_html_to_text(readme_md);
    std::vector<std::string> lines;
    {
        std::istringstream iss(text);
        std::string line;
        while (std::getline(iss, line)) {
            const std::string t  = trim(line);
            const std::string tl = to_lower(t);
            if (tl.find("example generations") != std::string::npos ||
                tl.find("how smart are \"they\"") != std::string::npos ||
                tl.find("## model evaluation") != std::string::npos ||
                tl.find("# model evaluation") != std::string::npos) {
                break;  // stop before long benchmark/example outputs
            }
            if (!t.empty()) {
                lines.push_back(t);
            }
        }
    }

    SamplingMatch best;
    int           best_score = -999;
    std::string   best_line;

    auto consider = [&](const std::string & candidate_line) {
        const SamplingMatch m = extract_sampling_from_text(candidate_line);
        const int s = score_sampling_line(to_lower(candidate_line), m);
        if (s > best_score && m.key_count() > 0) {
            best_score = s;
            best       = m;
            best_line  = candidate_line;
        }
    };

    for (size_t i = 0; i < lines.size(); ++i) {
        consider(lines[i]);
        // Also join multi-line setting bullets (e.g. DavidAU's "Model Suggested Settings" block)
        const std::string li = to_lower(lines[i]);
        if (li.find("setting") != std::string::npos || li.find("sampler") != std::string::npos ||
            li.find("parameter") != std::string::npos) {
            std::string block;
            for (size_t j = i; j < std::min(lines.size(), i + 5); ++j) {
                if (j > i) {
                    block += ", ";
                }
                block += lines[j];
            }
            consider(block);
        }
    }

    if (best.key_count() > 0) {
        if (best.has_temp) {
            out.has_temp = true;
            out.temp     = best.temp;
        }
        if (best.has_top_p) {
            out.has_top_p = true;
            out.top_p     = best.top_p;
        }
        if (best.has_top_k) {
            out.has_top_k = true;
            out.top_k     = best.top_k;
        }
        if (best.has_min_p) {
            out.has_min_p = true;
            out.min_p     = best.min_p;
        }
        if (best.has_repeat_penalty) {
            out.has_repeat_penalty = true;
            out.repeat_penalty     = best.repeat_penalty;
        }
        if (best.has_presence_penalty) {
            out.has_presence_penalty = true;
            out.presence_penalty     = best.presence_penalty;
        }
    }

    extract_thinking_modes(text, chat_template_jinja, out);
    out.sync_version = kPresetSyncVersion;
    out.found        = out.has_sampling();
    refresh_preset_note(out);

    return out;
}

void refresh_preset_note(FetchedPreset & preset) {
    if (!preset.found) {
        preset.note.clear();
        return;
    }
    std::string note;
    auto add_note = [&](const std::string & part) {
        if (!note.empty()) {
            note += ", ";
        }
        note += part;
    };
    if (preset.has_temp) add_note("temp " + fmt_num(preset.temp));
    if (preset.has_top_p) add_note("top_p " + fmt_num(preset.top_p));
    if (preset.has_top_k) add_note("top_k " + std::to_string(preset.top_k));
    if (preset.has_min_p) add_note("min_p " + fmt_num(preset.min_p));
    if (preset.has_repeat_penalty) add_note("rep_pen " + fmt_num(preset.repeat_penalty));
    if (preset.has_presence_penalty) add_note("pres_pen " + fmt_num(preset.presence_penalty));
    if (!preset.thinking_modes.empty()) {
        note += " | modes: ";
        constexpr size_t kMaxShown = 8;
        const size_t show_n = std::min(preset.thinking_modes.size(), kMaxShown);
        for (size_t i = 0; i < show_n; ++i) {
            if (i > 0) note += ",";
            note += preset.thinking_modes[i];
        }
        if (preset.thinking_modes.size() > kMaxShown) {
            note += " (+" + std::to_string(preset.thinking_modes.size() - kMaxShown) + " more)";
        }
    }
    preset.note = note;
}

std::string preset_sidecar_path(const std::string & model_path) {
    return model_path.empty() ? std::string{} : model_path + ".preset.json";
}

std::string preset_cache_path(const std::string & model_path, const std::string & cache_dir) {
    if (model_path.empty() || cache_dir.empty()) {
        return {};
    }
    return cache_dir + "/" + file_name_of(model_path) + ".preset.json";
}

static bool load_from_file(const std::string & path, FetchedPreset * out) {
    if (path.empty()) {
        return false;
    }
    std::ifstream in(path);
    if (!in.is_open()) {
        return false;
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    const nlohmann::json j = nlohmann::json::parse(ss.str(), nullptr, false);
    if (!j.is_object()) {
        return false;
    }
    FetchedPreset p;
    p.checked                  = j.value("checked", true);
    p.found                    = j.value("found", false);
    p.sync_version             = j.value("sync_version", 0);
    p.source_repo              = j.value("source_repo", std::string());
    p.base_repo                = j.value("base_repo", std::string());
    p.label                    = j.value("label", std::string());
    p.note                     = j.value("note", std::string());
    p.max_ctx                  = j.value("max_ctx", 0u);
    p.default_reasoning_effort = j.value("default_reasoning_effort", std::string());
    p.fetched_at               = j.value("fetched_at", (int64_t) 0);
    if (j.contains("temp") && j["temp"].is_number()) {
        p.has_temp = true;
        p.temp     = j["temp"].get<double>();
    }
    if (j.contains("top_p") && j["top_p"].is_number()) {
        p.has_top_p = true;
        p.top_p     = j["top_p"].get<double>();
    }
    if (j.contains("top_k") && j["top_k"].is_number()) {
        p.has_top_k = true;
        p.top_k     = j["top_k"].get<int>();
    }
    if (j.contains("min_p") && j["min_p"].is_number()) {
        p.has_min_p = true;
        p.min_p     = j["min_p"].get<double>();
    }
    if (j.contains("repeat_penalty") && j["repeat_penalty"].is_number()) {
        p.has_repeat_penalty = true;
        p.repeat_penalty     = j["repeat_penalty"].get<double>();
    }
    if (j.contains("presence_penalty") && j["presence_penalty"].is_number()) {
        p.has_presence_penalty = true;
        p.presence_penalty     = j["presence_penalty"].get<double>();
    }
    if (j.contains("thinking_modes") && j["thinking_modes"].is_array()) {
        for (const auto & item : j["thinking_modes"]) {
            if (item.is_string()) {
                p.thinking_modes.push_back(item.get<std::string>());
            }
        }
    }
    *out = std::move(p);
    return true;
}

bool load_preset_sidecar(const std::string & model_path, FetchedPreset * out, const std::string & cache_dir) {
    if (!out) {
        return false;
    }
    if (load_from_file(preset_sidecar_path(model_path), out)) {
        return true;
    }
    if (!cache_dir.empty() && load_from_file(preset_cache_path(model_path, cache_dir), out)) {
        return true;
    }
    return false;
}

bool save_preset_sidecar(const std::string & model_path, const FetchedPreset & preset, const std::string & cache_dir) {
    nlohmann::json j = {
        { "checked", preset.checked },
        { "found", preset.found },
        { "sync_version", preset.sync_version > 0 ? preset.sync_version : kPresetSyncVersion },
        { "source_repo", preset.source_repo },
        { "base_repo", preset.base_repo },
        { "label", preset.label },
        { "note", preset.note },
        { "max_ctx", preset.max_ctx },
        { "default_reasoning_effort", preset.default_reasoning_effort },
        { "thinking_modes", preset.thinking_modes },
        { "fetched_at", preset.fetched_at },
    };
    if (preset.has_temp) j["temp"] = preset.temp;
    if (preset.has_top_p) j["top_p"] = preset.top_p;
    if (preset.has_top_k) j["top_k"] = preset.top_k;
    if (preset.has_min_p) j["min_p"] = preset.min_p;
    if (preset.has_repeat_penalty) j["repeat_penalty"] = preset.repeat_penalty;
    if (preset.has_presence_penalty) j["presence_penalty"] = preset.presence_penalty;

    const std::string body = j.dump(2) + "\n";
    auto try_write = [&](const std::string & p) -> bool {
        if (p.empty()) {
            return false;
        }
        std::error_code ec;
        std::filesystem::create_directories(std::filesystem::path(p).parent_path(), ec);
        std::ofstream out(p);
        if (!out.is_open()) {
            return false;
        }
        out << body;
        return (bool) out;
    };

    if (try_write(preset_sidecar_path(model_path))) {
        return true;
    }
    return try_write(preset_cache_path(model_path, cache_dir));
}

std::vector<std::string> preset_to_llama_args(const FetchedPreset & preset) {
    std::vector<std::string> args;
    if (preset.has_temp) {
        args.insert(args.end(), { "--temp", fmt_num(preset.temp) });
    }
    if (preset.has_top_p) {
        args.insert(args.end(), { "--top-p", fmt_num(preset.top_p) });
    }
    if (preset.has_top_k) {
        args.insert(args.end(), { "--top-k", std::to_string(preset.top_k) });
    }
    if (preset.has_min_p) {
        args.insert(args.end(), { "--min-p", fmt_num(preset.min_p) });
    }
    if (preset.has_repeat_penalty) {
        args.insert(args.end(), { "--repeat-penalty", fmt_num(preset.repeat_penalty) });
    }
    if (preset.has_presence_penalty) {
        args.insert(args.end(), { "--presence-penalty", fmt_num(preset.presence_penalty) });
    }
    return args;
}

}  // namespace ps5lm
