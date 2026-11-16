// Background Hugging Face preset synchronization for installed PS5LM models.
// SPDX-License-Identifier: GPL-3.0-or-later
#include "preset_sync.hpp"

#include "fetch.hpp"
#include "log.hpp"
#include "preset_fetch.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <ctime>

namespace ps5lm {

namespace {

constexpr const char * kHub = "https://huggingface.co";

std::string to_lower(std::string s) {
    for (char & c : s) {
        c = (char) std::tolower((unsigned char) c);
    }
    return s;
}

std::string file_name_of(const std::string & path) {
    const size_t pos = path.find_last_of("/\\");
    return pos == std::string::npos ? path : path.substr(pos + 1);
}

std::string url_encode(const std::string & s) {
    std::string out;
    char        hex[4];
    for (unsigned char c : s) {
        if (std::isalnum(c) || c == '-' || c == '_' || c == '.' || c == '~') {
            out += (char) c;
        } else {
            std::snprintf(hex, sizeof(hex), "%%%02X", c);
            out += hex;
        }
    }
    return out;
}

std::string fetch_text(const std::string & url, size_t max_bytes, int tmpl) {
    std::string body;
    const FetchResult r = https_get(
        url, "",
        [&](const char * p, size_t n) {
            const size_t take = std::min(n, max_bytes - body.size());
            body.append(p, take);
            return body.size() < max_bytes;
        },
        tmpl, 64u << 10);
    if (r.status < 200 || r.status >= 300) {
        return {};
    }
    return body;
}

std::string first_base_model(const nlohmann::json & meta) {
    if (!meta.is_object()) {
        return {};
    }
    const auto cd_it = meta.find("cardData");
    if (cd_it == meta.end() || !cd_it->is_object()) {
        return {};
    }
    const auto bm_it = cd_it->find("base_model");
    if (bm_it == cd_it->end()) {
        return {};
    }
    if (bm_it->is_string()) {
        return bm_it->get<std::string>();
    }
    if (bm_it->is_array() && !bm_it->empty() && (*bm_it)[0].is_string()) {
        return (*bm_it)[0].get<std::string>();
    }
    return {};
}

bool has_sibling(const nlohmann::json & meta, const std::string & fname) {
    if (!meta.is_object() || !meta.contains("siblings") || !meta["siblings"].is_array()) {
        return true;  // optimistic fallback if API metadata wasn't available
    }
    for (const auto & s : meta["siblings"]) {
        if (s.is_object() && s.value("rfilename", std::string()) == fname) {
            return true;
        }
    }
    return false;
}

FetchedPreset resolve_repo_chain(const std::string & repo_id, int tmpl, int depth = 0) {
    FetchedPreset out;
    if (repo_id.empty() || repo_id.find('/') == std::string::npos) {
        return out;
    }

    const std::string api_body = fetch_text(std::string(kHub) + "/api/models/" + repo_id, 256u << 10, tmpl);
    const nlohmann::json meta  = api_body.empty() ? nlohmann::json{} : nlohmann::json::parse(api_body, nullptr, false);

    std::string gen_cfg;
    if (has_sibling(meta, "generation_config.json")) {
        gen_cfg = fetch_text(std::string(kHub) + "/" + repo_id + "/raw/main/generation_config.json", 64u << 10, tmpl);
    }
    std::string readme = fetch_text(std::string(kHub) + "/" + repo_id + "/raw/main/README.md", 384u << 10, tmpl);
    std::string jinja;
    if (has_sibling(meta, "chat_template.jinja")) {
        jinja = fetch_text(std::string(kHub) + "/" + repo_id + "/raw/main/chat_template.jinja", 192u << 10, tmpl);
    }

    out = parse_hf_model_metadata(repo_id, gen_cfg, readme, jinja);
    const std::string base = first_base_model(meta);
    out.base_repo          = base;

    const bool need_parent = !out.found || !out.has_min_p || !out.has_repeat_penalty ||
                             !out.has_presence_penalty || out.thinking_modes.empty() ||
                             out.default_reasoning_effort.empty();
    if (need_parent && !base.empty() && base != repo_id && depth < 2) {
        FetchedPreset parent = resolve_repo_chain(base, tmpl, depth + 1);
        if (!out.found && parent.found) {
            parent.source_repo = repo_id;
            parent.base_repo   = base;
            out                = std::move(parent);
        } else if (parent.found || !parent.thinking_modes.empty()) {
            if (!out.has_temp && parent.has_temp) {
                out.has_temp = true;
                out.temp     = parent.temp;
            }
            if (!out.has_top_p && parent.has_top_p) {
                out.has_top_p = true;
                out.top_p     = parent.top_p;
            }
            if (!out.has_top_k && parent.has_top_k) {
                out.has_top_k = true;
                out.top_k     = parent.top_k;
            }
            if (!out.has_min_p && parent.has_min_p) {
                out.has_min_p = true;
                out.min_p     = parent.min_p;
            }
            if (!out.has_repeat_penalty && parent.has_repeat_penalty) {
                out.has_repeat_penalty = true;
                out.repeat_penalty     = parent.repeat_penalty;
            }
            if (!out.has_presence_penalty && parent.has_presence_penalty) {
                out.has_presence_penalty = true;
                out.presence_penalty     = parent.presence_penalty;
            }
            for (const auto & mode : parent.thinking_modes) {
                if (std::find(out.thinking_modes.begin(), out.thinking_modes.end(), mode) ==
                    out.thinking_modes.end()) {
                    out.thinking_modes.push_back(mode);
                }
            }
            if (out.default_reasoning_effort.empty() && !parent.default_reasoning_effort.empty()) {
                out.default_reasoning_effort = parent.default_reasoning_effort;
            }
            out.found = out.has_sampling();
            refresh_preset_note(out);
        }
    }

    out.checked      = true;
    out.sync_version = kPresetSyncVersion;
    out.fetched_at   = (int64_t) std::time(nullptr);
    return out;
}

std::string find_repo_for_gguf(const std::string & model_path, int tmpl) {
    const std::string fname   = file_name_of(model_path);
    const auto        queries = build_hf_search_queries(fname);
    if (queries.empty()) {
        return {};
    }
    const std::string fname_low = to_lower(fname);
    const std::string stem_low  = to_lower(queries.front());

    for (const auto & q : queries) {
        const std::string search_url =
            std::string(kHub) + "/api/models?filter=gguf&sort=downloads&direction=-1&limit=6&search=" + url_encode(q);
        const std::string body = fetch_text(search_url, 256u << 10, tmpl);
        if (body.empty()) {
            continue;
        }
        const nlohmann::json arr = nlohmann::json::parse(body, nullptr, false);
        if (!arr.is_array() || arr.empty()) {
            continue;
        }

        std::string fallback;
        for (const auto & entry : arr) {
            if (!entry.is_object()) {
                continue;
            }
            const std::string id = entry.value("id", std::string());
            if (id.empty()) {
                continue;
            }
            const std::string id_low = to_lower(id);
            if (fallback.empty() && (id_low.find(stem_low) != std::string::npos ||
                                     id_low.find(to_lower(q)) != std::string::npos)) {
                fallback = id;
            }
            // Check siblings via repo API to confirm exact GGUF match when possible.
            const std::string meta_body = fetch_text(std::string(kHub) + "/api/models/" + id, 256u << 10, tmpl);
            const nlohmann::json meta   = nlohmann::json::parse(meta_body, nullptr, false);
            if (meta.is_object() && meta.contains("siblings") && meta["siblings"].is_array()) {
                for (const auto & s : meta["siblings"]) {
                    if (!s.is_object()) {
                        continue;
                    }
                    const std::string rf = to_lower(s.value("rfilename", std::string()));
                    if (rf == fname_low) {
                        return id;
                    }
                }
            }
        }
        if (!fallback.empty()) {
            return fallback;
        }
        if (arr[0].is_object()) {
            const std::string top = arr[0].value("id", std::string());
            if (!top.empty()) {
                return top;
            }
        }
    }
    return {};
}

}  // namespace

bool sync_preset_for_repo(const std::string & repo_id, const std::string & model_path, const std::string & cache_dir) {
    const int tmpl = https_template();
    if (tmpl < 0 || repo_id.empty() || model_path.empty()) {
        return false;
    }
    FetchedPreset p = resolve_repo_chain(repo_id, tmpl);
    if (!save_preset_sidecar(model_path, p, cache_dir)) {
        return false;
    }
    ps5lm::log_line("preset %s: %s (%s)\n", file_name_of(model_path).c_str(), p.found ? "found" : "checked",
                    p.source_repo.empty() ? "-" : p.source_repo.c_str());
    return p.found;
}

void sync_presets_for_models(const std::vector<std::string> & model_paths, const std::string & cache_dir,
                             const std::function<void()> & on_updated) {
    const int tmpl = https_template();
    if (tmpl < 0) {
        return;
    }
    bool any_found = false;
    for (const auto & path : model_paths) {
        FetchedPreset existing;
        if (load_preset_sidecar(path, &existing, cache_dir) && existing.checked &&
            existing.sync_version >= kPresetSyncVersion) {
            continue;
        }
        const std::string repo = find_repo_for_gguf(path, tmpl);
        FetchedPreset p;
        if (!repo.empty()) {
            p = resolve_repo_chain(repo, tmpl);
        } else {
            p.checked      = true;
            p.sync_version = kPresetSyncVersion;
            p.fetched_at   = (int64_t) std::time(nullptr);
        }
        if (save_preset_sidecar(path, p, cache_dir) && p.found) {
            any_found = true;
            ps5lm::log_line("auto-preset %s <- %s (%s)\n", file_name_of(path).c_str(), p.source_repo.c_str(),
                            p.note.c_str());
        }
    }
    if (any_found && on_updated) {
        on_updated();
    }
}

}  // namespace ps5lm
