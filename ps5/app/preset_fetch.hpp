// Hugging Face model-card preset parser and local sidecar cache for PS5LM.
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace ps5lm {

constexpr int kPresetSyncVersion = 2;

struct FetchedPreset {
    bool        checked      = false;
    bool        found        = false;
    int         sync_version = 0;
    std::string source_repo;
    std::string base_repo;
    std::string label;
    std::string note;

    bool   has_temp             = false;
    double temp                 = 1.0;
    bool   has_top_p            = false;
    double top_p                = 0.95;
    bool   has_top_k            = false;
    int    top_k                = 20;
    bool   has_min_p            = false;
    double min_p                = 0.0;
    bool   has_repeat_penalty   = false;
    double repeat_penalty       = 1.0;
    bool   has_presence_penalty = false;
    double presence_penalty     = 0.0;

    uint32_t                 max_ctx = 0;
    std::string              default_reasoning_effort;
    std::vector<std::string> thinking_modes;
    int64_t                  fetched_at = 0;

    bool has_sampling() const {
        return has_temp || has_top_p || has_top_k || has_min_p || has_repeat_penalty || has_presence_penalty;
    }
};

// Strips split suffixes, .gguf extension, and quant tags (Q4_K_M, IQ4_XS, etc.)
// so Hugging Face search finds the model repo from a local GGUF file name.
std::string clean_gguf_search_stem(const std::string & file_name);

// Builds progressive Hugging Face search queries (exact stem, expanded common
// filename abbreviations like Uncnr->Uncensored / Brilli->Brilliance, and shorter prefixes).
std::vector<std::string> build_hf_search_queries(const std::string & file_name);

// Parses generation_config.json, README.md, and optional chat_template.jinja
// to extract author-recommended sampling parameters and supported thinking modes.
FetchedPreset parse_hf_model_metadata(const std::string & repo_id, const std::string & gen_config_json,
                                      const std::string & readme_md, const std::string & chat_template_jinja = {});

// Sidecar path next to the GGUF (<model_path>.preset.json) and optional fallback
// directory (<cache_dir>/<file_name>.preset.json).
std::string preset_sidecar_path(const std::string & model_path);
std::string preset_cache_path(const std::string & model_path, const std::string & cache_dir);

bool load_preset_sidecar(const std::string & model_path, FetchedPreset * out, const std::string & cache_dir = {});
bool save_preset_sidecar(const std::string & model_path, const FetchedPreset & preset,
                         const std::string & cache_dir = {});

// Rebuilds the human-readable summary note from extracted sampling/modes.
void refresh_preset_note(FetchedPreset & preset);

// Converts extracted sampling fields to llama-server CLI flags.
std::vector<std::string> preset_to_llama_args(const FetchedPreset & preset);

}  // namespace ps5lm
