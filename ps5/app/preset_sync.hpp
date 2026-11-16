// Background Hugging Face preset synchronization for installed PS5LM models.
// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once

#include <functional>
#include <string>
#include <vector>

namespace ps5lm {

// Fetches model card & generation_config.json for `repo_id` (following base_model
// if the GGUF repo omits sampling settings) and saves `<model_path>.preset.json`.
bool sync_preset_for_repo(const std::string & repo_id, const std::string & model_path,
                          const std::string & cache_dir = {});

// Scans installed model paths lacking a `.preset.json` sidecar, resolves each via
// Hugging Face search, and caches its extracted preset. Runs synchronously; invoke
// from a background thread. Invokes `on_updated` if any preset was saved.
void sync_presets_for_models(const std::vector<std::string> & model_paths, const std::string & cache_dir,
                             const std::function<void()> & on_updated = {});

}  // namespace ps5lm
