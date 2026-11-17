// Host check for model_plan.cpp:
//   c++ -std=c++17 -O1 -I third_party/llama.cpp/vendor ps5/app/model_plan_test.cpp ps5/app/model_plan.cpp ps5/app/preset_fetch.cpp -o build/model_plan_test
//   build/model_plan_test          synthetic headers only
//   build/model_plan_test models   also the real GGUF files in models/
// Uses the budget the console reported (11.38 GiB free) and the two
// configurations measured there.
// SPDX-License-Identifier: GPL-3.0-or-later
#include "model_plan.hpp"
#include "preset_fetch.hpp"

#include <cmath>

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <fstream>
#include <unistd.h>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace {

// A GGUF header with string and uint32 metadata, no tensors.
struct Gguf {
    std::string kvs;
    uint64_t    n = 0;
    template <class T> static std::string raw(T v) { return std::string((const char *) &v, sizeof(v)); }
    static std::string str(const std::string & v) { return raw<uint64_t>(v.size()) + v; }
    Gguf & kv(const std::string & k, const std::string & v) { kvs += str(k) + raw<uint32_t>(8) + str(v); n++; return *this; }
    Gguf & kv(const std::string & k, uint32_t v) { kvs += str(k) + raw<uint32_t>(4) + raw(v); n++; return *this; }
    Gguf & kv(const std::string & k, const std::vector<bool> & v) {  // a bool array
        kvs += str(k) + raw<uint32_t>(9) + raw<uint32_t>(7) + raw<uint64_t>(v.size());
        for (bool b : v) {
            kvs += (char) b;
        }
        n++;
        return *this;
    }
    // magic, version 3, tensor count, metadata count, metadata
    std::string bytes() const { return "GGUF" + raw<uint32_t>(3) + raw<uint64_t>(0) + raw(n) + kvs; }
};

bool has_args(const ps5lm::Plan & p, const char * flag, const char * value) {
    auto it = std::find(p.args.begin(), p.args.end(), flag);
    return it != p.args.end() && it + 1 != p.args.end() && *(it + 1) == value;
}

}  // namespace

int main(int argc, char ** argv) {
    const double budget = 11.38;

    // A decision model (Clef is Qwen 3.5 9B with a decision head): found from
    // "<arch>.decision.type", a wide ubatch, a 4k cap and no chat sampling.
    Gguf clef;
    clef.kv("general.architecture", "qwen35").kv("general.name", "Clef Flash Qwen3.5 9B")
        .kv("qwen35.block_count", 32u).kv("qwen35.attention.head_count", 16u)
        .kv("qwen35.attention.head_count_kv", 4u).kv("qwen35.embedding_length", 4096u)
        .kv("qwen35.context_length", 262144u).kv("qwen35.decision.type", "clef");
    const ps5lm::ModelInfo d = ps5lm::read_model_header("clef-flash-Q4_K_M.gguf", clef.bytes(), 5ull << 30);
    assert(d.ok && d.decision == "clef");
    assert(!ps5lm::find_preset(d));
    const ps5lm::Plan dp = ps5lm::plan_model(d, budget);
    assert(dp.fits && dp.ctx == 4096 && has_args(dp, "-ub", "2048") && has_args(dp, "-b", "2048"));
    assert(dp.why.find("decision model (clef)") == 0);

    // The same model without the decision key is a chat model with the narrow batch.
    Gguf chat;
    chat.kv("general.architecture", "qwen35").kv("general.name", "Qwen3.5 9B")
        .kv("qwen35.block_count", 32u).kv("qwen35.attention.head_count", 16u)
        .kv("qwen35.attention.head_count_kv", 4u).kv("qwen35.embedding_length", 4096u)
        .kv("qwen35.context_length", 262144u);
    const ps5lm::ModelInfo c = ps5lm::read_model_header("Qwen3.5-9B-Q4_K_M.gguf", chat.bytes(), 5ull << 30);
    assert(c.ok && c.decision.empty() && ps5lm::find_preset(c));
    const ps5lm::Plan cp = ps5lm::plan_model(c, budget);
    assert(cp.fits && cp.ctx > 4096 && has_args(cp, "-ub", "64"));

    // A larger prompt batch costs compute buffer, counted by the feed-forward width.
    Gguf wide_ff = chat;
    wide_ff.kv("qwen35.feed_forward_length", 12288u);
    const ps5lm::ModelInfo w = ps5lm::read_model_header("Qwen3.5-9B-Q4_K_M.gguf", wide_ff.bytes(), 5ull << 30);
    const ps5lm::Plan w64 = ps5lm::plan_model(w, budget), w512 = ps5lm::plan_model(w, budget, 65536, nullptr, 512);
    assert(w.n_ff == 12288 && has_args(w64, "-ub", "64") && has_args(w512, "-ub", "512") && has_args(w512, "-b", "512"));
    assert(w512.reserve_gib - w64.reserve_gib > 0.18 && w512.reserve_gib - w64.reserve_gib < 0.21);

    // Bonsai 2: upstream's Q2_0 loads with its preset (64k at most, PrismML's
    // sampling); PrismML's own PQ2_0 and PTQ1_0 say what to do instead.
    const auto bonsai = [](uint32_t ftype) {
        Gguf g;
        g.kv("general.architecture", "qwen35").kv("general.name", "Ternary-Bonsai-2-27B")
            .kv("qwen35.block_count", 64u).kv("qwen35.attention.head_count", 24u)
            .kv("qwen35.attention.head_count_kv", 4u).kv("qwen35.embedding_length", 5120u)
            .kv("qwen35.attention.key_length", 256u).kv("qwen35.attention.value_length", 256u)
            .kv("qwen35.full_attention_interval", 4u).kv("qwen35.context_length", 262144u)
            .kv("prism.hadamard.version", 1u).kv("general.file_type", ftype);
        return g.bytes();
    };
    const ps5lm::ModelInfo b = ps5lm::read_model_header("Ternary-Bonsai-2-27B-Q2_0.gguf", bonsai(41), 7626008928ull);
    const ps5lm::Preset * bp = ps5lm::find_preset(b);
    assert(b.ok && bp && std::string(bp->label) == "Bonsai 2");
    const ps5lm::Plan bpl = ps5lm::plan_model(b, 12.0, 131072);
    assert(bpl.fits && bpl.ctx == 65536 && has_args(bpl, "--min-p", "0.05"));
    const ps5lm::ModelInfo pq = ps5lm::read_model_header("Ternary-Bonsai-2-27B-PQ2_0.gguf", bonsai(141), 7206168928ull);
    assert(!pq.ok && pq.error.find("bonsai-pq2-to-q2.py") != std::string::npos);
    const ps5lm::ModelInfo pt = ps5lm::read_model_header("Ternary-Bonsai-2-27B-PTQ1_0.gguf", bonsai(143), 5946648928ull);
    assert(!pt.ok && pt.error.find("PTQ1_0") != std::string::npos);

    // Sliding-window attention (#57): Spark-X2.6 4B's header. 27 of its 36
    // layers keep a 512-token window, so 256k fits; counted as full attention
    // it stopped at 128k.
    std::vector<bool> swa;
    for (int i = 0; i < 36; i++) {
        swa.push_back(i % 4 != 3);
    }
    Gguf spark;
    spark.kv("general.architecture", "spark2_5").kv("spark2_5.block_count", 36u)
        .kv("spark2_5.attention.head_count", 16u).kv("spark2_5.attention.head_count_kv", 4u)
        .kv("spark2_5.embedding_length", 2560u).kv("spark2_5.attention.key_length", 256u)
        .kv("spark2_5.attention.value_length", 256u).kv("spark2_5.context_length", 1048576u)
        .kv("spark2_5.attention.sliding_window", 512u).kv("spark2_5.attention.sliding_window_pattern", swa);
    const ps5lm::ModelInfo s = ps5lm::read_model_header("Spark-X2.6-4B-Q8_0.gguf", spark.bytes(), 4375021824ull);
    assert(s.ok && s.n_swa_layers == 27 && s.swa_window == 512);
    assert(kv_bytes_per_token(s, "q8_0") == 9 * 4 * 512 * 34.0 / 32);
    const ps5lm::Plan sp = ps5lm::plan_model(s, budget, 262144);
    assert(sp.fits && sp.ctx == 262144 && sp.kv_type == "q8_0");
    ps5lm::ModelInfo full = s;
    full.n_swa_layers = 0;
    assert(ps5lm::plan_model(full, budget, 262144).ctx == 131072);

    // A Qwen projector beside its model takes images in pieces: a narrow
    // ubatch, a 1280 batch and image cap, and the headroom kept. Another
    // projector type keeps the wide ubatch.
    {
        char dir[] = "/tmp/model_plan_test.XXXXXX";
        assert(mkdtemp(dir));
        const std::string at = dir;
        const auto write = [&](const std::string & name, const Gguf & g) {
            std::ofstream(at + "/" + name, std::ios::binary) << g.bytes();
        };
        Gguf projector;
        projector.kv("general.architecture", "clip").kv("clip.projector_type", "qwen3vl_merger");
        write("Qwen3.8-27B-UD-IQ2_S.gguf", chat);
        write("mmproj-Qwen3.8-27B-F16.gguf", projector);
        const ps5lm::ModelInfo m = ps5lm::read_model_info(at + "/Qwen3.8-27B-UD-IQ2_S.gguf");
        assert(m.mmproj == at + "/mmproj-Qwen3.8-27B-F16.gguf" && m.projector_type == "qwen3vl_merger");
        const ps5lm::Plan vp = ps5lm::plan_model(m, budget, 8192);
        assert(has_args(vp, "-ub", "256") && has_args(vp, "-b", "1280") && has_args(vp, "--image-max-tokens", "1280"));
        assert(vp.reserve_gib >= 0.75 + 0.25 + 0.5);
        assert(!ps5lm::causal_projector("gemma3") && ps5lm::causal_projector("qwen2vl_merger"));
        ps5lm::ModelInfo g = m;
        g.projector_type = "gemma3";
        const ps5lm::Plan gp = ps5lm::plan_model(g, budget, 8192);
        assert(has_args(gp, "-ub", "2048") && has_args(gp, "-b", "2048"));
        assert(std::find(gp.args.begin(), gp.args.end(), "--image-max-tokens") == gp.args.end());
        unlink((at + "/Qwen3.8-27B-UD-IQ2_S.gguf").c_str());
        unlink((at + "/mmproj-Qwen3.8-27B-F16.gguf").c_str());
        rmdir(dir);
    }

    // Headroom-based RAM prompt cache (-cram) & checkpoints (-ctxcp):
    // spare headroom enables RAM caching (capped at <= 512 MiB); tight headroom switches automatically to disk caching (--cache-disk).
    const ps5lm::Plan roomy = ps5lm::plan_model(c, budget, 65536);
    assert(roomy.fits && has_args(roomy, "-cram", "256") && has_args(roomy, "-ctxcp", "4"));
    assert(has_args(roomy, "--cache-disk", "/data/PS5LM/cache/prompts"));
    ps5lm::ModelInfo small2b = c;
    small2b.file_bytes = 2ull << 30;
    const ps5lm::Plan p_small2b = ps5lm::plan_model(small2b, budget, 65536);
    assert(p_small2b.fits && has_args(p_small2b, "-cram", "512") && has_args(p_small2b, "-ctxcp", "4"));
    const ps5lm::Plan tight = ps5lm::plan_model(c, c.file_bytes / (1024.0 * 1024.0 * 1024.0) + 1.1, 4096);
    assert(tight.fits && has_args(tight, "-cram", "0") && has_args(tight, "-ctxcp", "1"));
    assert(has_args(tight, "--cache-disk", "/data/PS5LM/cache/prompts") && has_args(tight, "--cache-disk-mib", "4096"));

    // Disk cache off leaves out --cache-disk and preserves strict fixed kv_type without expansion.
    const ps5lm::Plan no_disk = ps5lm::plan_model(c, budget, 65536, "q8_0", ps5lm::kPromptUbatch, nullptr, "off");
    assert(no_disk.fits && std::find(no_disk.args.begin(), no_disk.args.end(), "--cache-disk") == no_disk.args.end());

    // Near-full VRAM with fixed q8_0 honors q8_0 when >= 8k fits in "auto" mode, and expands
    // via q4_0 when q8_0 cannot reach 8k or when disk_cache == "on".
    const double tight_budget = 6.05;
    const ps5lm::Plan strict_q8 = ps5lm::plan_model(c, tight_budget, 65536, "q8_0", ps5lm::kPromptUbatch, nullptr, "off");
    const ps5lm::Plan auto_exp  = ps5lm::plan_model(c, tight_budget, 65536, "q8_0", ps5lm::kPromptUbatch, nullptr, "auto", 4096);
    assert(auto_exp.fits && (!strict_q8.fits || auto_exp.ctx > strict_q8.ctx) && auto_exp.kv_type == "q4_0");
    assert(has_args(auto_exp, "-cram", "0") && has_args(auto_exp, "--cache-disk", "/data/PS5LM/cache/prompts"));
    const ps5lm::Plan keep_q8_auto = ps5lm::plan_model(c, 6.55, 65536, "q8_0", ps5lm::kPromptUbatch, nullptr, "auto", 4096);
    const ps5lm::Plan max_ctx_on   = ps5lm::plan_model(c, 6.55, 65536, "q8_0", ps5lm::kPromptUbatch, nullptr, "on", 4096);
    assert(keep_q8_auto.fits && keep_q8_auto.ctx == 8192 && keep_q8_auto.kv_type == "q8_0");
    assert(max_ctx_on.fits && max_ctx_on.ctx == 16384 && max_ctx_on.kv_type == "q4_0");

    // 18B MoE (~10.59 GiB) on 11.56 GiB free pool: "auto"/"on" fit >= 8k via tight disk packing;
    // "off" keeps standard RAM buffers (-ub 64, -ctxcp 1) without --cache-disk and at shorter context.
    ps5lm::ModelInfo moe18 = c;
    moe18.arch       = "qwen35moe";
    moe18.file_name  = "qwen3.5-18b-a3b-reap-coding-heretic-v0.i1-Q4_K_M.gguf";
    moe18.file_bytes = 11375418272ull;
    moe18.n_expert   = 256;
    moe18.full_attn_every = 4;
    moe18.state_bytes = 64ull << 20;
    const ps5lm::Plan p18_auto = ps5lm::plan_model(moe18, 11.56, 65536, "q8_0", ps5lm::kPromptUbatch, nullptr, "auto", 4096);
    const ps5lm::Plan p18_on   = ps5lm::plan_model(moe18, 11.56, 65536, "q8_0", ps5lm::kPromptUbatch, nullptr, "on", 4096);
    const ps5lm::Plan p18_off  = ps5lm::plan_model(moe18, 11.56, 65536, "q8_0", ps5lm::kPromptUbatch, nullptr, "off", 4096,
                                                   "/data/PS5LM/cache/prompts", false);
    assert(p18_auto.fits && p18_auto.ctx >= 8192 && p18_auto.kv_type == "q8_0" && (11.56 - p18_auto.total_gib) >= 0.15 &&
           has_args(p18_auto, "-cram", "0") &&
           has_args(p18_auto, "--cache-disk", "/data/PS5LM/cache/prompts"));
    assert(p18_on.fits && p18_on.ctx >= 8192 && p18_on.kv_type == "q8_0" && (11.56 - p18_on.total_gib) >= 0.15 &&
           has_args(p18_on, "-cram", "0") &&
           has_args(p18_on, "--cache-disk", "/data/PS5LM/cache/prompts"));
    assert(p18_on.why.find(" + disk") != std::string::npos && p18_on.why.find("(expanded)") == std::string::npos);
    assert(p18_off.fits && p18_off.ctx <= p18_auto.ctx && p18_off.kv_type == "q8_0" && !p18_off.disable_introspect &&
           has_args(p18_off, "-ub", "64") &&
           std::find(p18_off.args.begin(), p18_off.args.end(), "--cache-disk") == p18_off.args.end() &&
           p18_off.why.find("(expanded)") == std::string::npos && p18_off.why.find("+ disk") == std::string::npos);
    assert(has_args(p18_auto, "-b", "128") && has_args(p18_on, "-b", "128") && has_args(p18_off, "-b", "128"));

    // Verify state_gib is counted exactly once (not doubled in base_reserve + try_fit).
    {
        ps5lm::ModelInfo no_state = moe18;
        no_state.file_bytes  = 8ull << 30;
        no_state.state_bytes = 0;
        ps5lm::ModelInfo with_state = no_state;
        with_state.state_bytes = 256ull << 20;  // 0.25 GiB
        const ps5lm::Plan p_ns = ps5lm::plan_model(no_state, 11.56, 8192);
        const ps5lm::Plan p_ws = ps5lm::plan_model(with_state, 11.56, 8192);
        assert(p_ns.fits && p_ws.fits && p_ns.ctx == p_ws.ctx && p_ns.kv_type == p_ws.kv_type);
        assert(std::fabs((p_ws.total_gib - p_ns.total_gib) - 0.25) < 1e-6);
    }

    // Introspect adds +0.25 GiB to base_reserve, and tight_plan enforces >= 0.15 GiB
    // safety margin below budget_gib while capping logical batch -b to 128.
    {
        const ps5lm::Plan cp_intro = ps5lm::plan_model(c, budget, 65536, nullptr, ps5lm::kPromptUbatch, nullptr,
                                                        "auto", 4096, "/data/PS5LM/cache/prompts", true);
        assert(cp_intro.fits && std::fabs((cp_intro.reserve_gib - cp.reserve_gib) - 0.25) < 1e-6);

        ps5lm::ModelInfo near_full = moe18;
        near_full.file_bytes = (uint64_t) (10.58 * 1073741824.0);  // 10.58 GiB weights on 11.54 GiB pool
        near_full.state_bytes = 160ull << 20;
        const ps5lm::Plan safe_p = ps5lm::plan_model(near_full, 11.54, 65536, "auto", ps5lm::kPromptUbatch, nullptr,
                                                      "auto", 4096, "/data/PS5LM/cache/prompts", false);
        assert(safe_p.fits && safe_p.ctx <= 16384 && (11.54 - safe_p.total_gib) >= 0.15);
        if (safe_p.ctx >= 16384) {
            assert((11.54 - safe_p.total_gib) >= 0.32);
        }
        assert(has_args(safe_p, "-ub", "32") && has_args(safe_p, "-b", "128"));

        // An 18B-style tight fit cannot promote 16k without preserving >= 0.32 GiB planner headroom.
        ps5lm::ModelInfo near_16k = moe18;
        near_16k.file_bytes = (uint64_t) (10.55 * 1073741824.0);
        near_16k.state_bytes = 64ull << 20;
        const ps5lm::Plan p16 = ps5lm::plan_model(near_16k, 11.54, 65536, "q8_0", ps5lm::kPromptUbatch, nullptr,
                                                  "auto", 8192, "/data/PS5LM/cache/prompts", false);
        assert(p16.fits && (p16.ctx < 16384 || (11.54 - p16.total_gib) >= 0.32));

        // When introspect already reaches >= 32k, keep introspect enabled (reserve +0.25 GiB,
        // context steps down, both preserve >= 0.15 GiB). When introspect caps a model at 8k,
        // auto-disable introspect to unlock >= 16k / 32k.
        near_full.file_bytes = (uint64_t) (10.00 * 1073741824.0);
        near_full.state_bytes = 64ull << 20;
        const ps5lm::Plan safe_no_intro = ps5lm::plan_model(near_full, 11.54, 65536, "auto", ps5lm::kPromptUbatch, nullptr,
                                                             "auto", 4096, "/data/PS5LM/cache/prompts", false);
        const ps5lm::Plan safe_intro    = ps5lm::plan_model(near_full, 11.54, 65536, "auto", ps5lm::kPromptUbatch, nullptr,
                                                             "auto", 4096, "/data/PS5LM/cache/prompts", true);
        assert(safe_no_intro.fits && safe_intro.fits);
        assert(!safe_intro.disable_introspect);
        assert(safe_intro.reserve_gib > safe_no_intro.reserve_gib + 0.24);
        assert((11.54 - safe_no_intro.total_gib) >= 0.15 && (11.54 - safe_intro.total_gib) >= 0.15);
        assert(safe_intro.ctx < safe_no_intro.ctx);

        // With 256 MiB flexibleMemorySize (~11.75 GiB free Direct Memory), real 18B-A3B (40 layers,
        // 2 KV heads, 10 KV layers) clearly distinguishes all 3 modes: "off" < "auto" (16k q8_0 + disk)
        // < "on" (32k q4_0 expanded + disk).
        ps5lm::ModelInfo real18 = moe18;
        real18.n_layer = 40;
        real18.n_head_kv = 2;
        real18.state_bytes = 65863680ull;  // ~62.81 MiB
        const ps5lm::Plan p18_32k_auto = ps5lm::plan_model(real18, 11.7475, 65536, nullptr, ps5lm::kPromptUbatch, nullptr,
                                                             "auto", 2048, "/data/PS5LM/cache/prompts", true);
        const ps5lm::Plan p18_16k_q8   = ps5lm::plan_model(real18, 11.7475, 65536, "q8_0", ps5lm::kPromptUbatch, nullptr,
                                                             "auto", 2048, "/data/PS5LM/cache/prompts", true);
        const ps5lm::Plan p18_8k_off   = ps5lm::plan_model(real18, 11.7475, 65536, "q8_0", ps5lm::kPromptUbatch, nullptr,
                                                             "off", 2048, "/data/PS5LM/cache/prompts", true);
        const ps5lm::Plan p18_32k_on   = ps5lm::plan_model(real18, 11.7475, 65536, "q8_0", ps5lm::kPromptUbatch, nullptr,
                                                             "on", 2048, "/data/PS5LM/cache/prompts", true);
        assert(p18_32k_auto.fits && p18_32k_auto.disable_introspect && p18_32k_auto.ctx == 32768 && p18_32k_auto.kv_type == "q4_0");
        assert(p18_16k_q8.fits && p18_16k_q8.disable_introspect && p18_16k_q8.ctx == 16384 && p18_16k_q8.kv_type == "q8_0");
        assert(p18_8k_off.fits && !p18_8k_off.disable_introspect && p18_8k_off.ctx == 8192 && p18_8k_off.kv_type == "q8_0");
        assert(p18_8k_off.why.find("+ disk") == std::string::npos);
        assert(p18_32k_on.fits && p18_32k_on.disable_introspect && p18_32k_on.ctx == 32768 && p18_32k_on.kv_type == "q4_0");
        assert(p18_32k_on.why.find(" (expanded) + disk") != std::string::npos);

        // Even with introspect enabled in Settings, 18B MoE fits on 11.56 GiB by reclaiming the unused
        // -ctxcp 0 checkpoint and dropping the 256 MiB introspect buffer when needed.
        ps5lm::ModelInfo moe18_intro = moe18;
        moe18_intro.state_bytes = 160ull << 20;
        const ps5lm::Plan p18_intro = ps5lm::plan_model(moe18_intro, 11.56, 65536, "q8_0", ps5lm::kPromptUbatch, nullptr,
                                                          "auto", 8192, "/data/PS5LM/cache/prompts", true);
        assert(p18_intro.fits && p18_intro.disable_introspect && p18_intro.ctx >= 8192);
        assert((11.56 - p18_intro.total_gib) >= 0.15);
        assert(has_args(p18_intro, "-b", "128") && has_args(p18_intro, "--cache-disk", "/data/PS5LM/cache/prompts"));
    }

    // Built-in fine-tune presets for the user's installed Hugging Face models:
    {
        ps5lm::ModelInfo m2b = c;
        m2b.file_name = "Qwen3.5-2B-Claude-4.6-OS-Auto-Variable-HERETIC-UNCENSORED-THINKING.Q8_0.gguf";
        const ps5lm::Plan p2b = ps5lm::plan_model(m2b, budget);
        assert(has_args(p2b, "--temp", "1.0") && has_args(p2b, "--presence-penalty", "1.5") &&
               has_args(p2b, "--repeat-penalty", "1.0"));

        ps5lm::ModelInfo m9b = c;
        m9b.file_name = "Qwen3.5-9B-The-Defiant-Fable-Uncensored-Heretic-NEO-IMATRIX-MAX-MTP-Q4_K_M.gguf";
        const ps5lm::Plan p9b = ps5lm::plan_model(m9b, budget);
        assert(has_args(p9b, "--temp", "1.0") && has_args(p9b, "--repeat-penalty", "1.0") &&
               has_args(p9b, "--presence-penalty", "1.5"));

        ps5lm::ModelInfo m12b = c;
        m12b.file_name = "Qwen3.6-12B-IQ-Ultra-Heretic-Uncensored-Thinking-V2-Hightop-Q4_K_S.gguf";
        const ps5lm::Plan p12b = ps5lm::plan_model(m12b, budget);
        assert(has_args(p12b, "--temp", "1.0") && has_args(p12b, "--min-p", "0.05") &&
               has_args(p12b, "--repeat-penalty", "1.05"));

        ps5lm::ModelInfo m27b = c;
        m27b.file_name = "Qwen3.8-27B-Ultimate-Heretic-Uncensored-Turbo-Brilliance-50x-AGI-NEO-Q4_K_M.gguf";
        const ps5lm::Plan p27b = ps5lm::plan_model(m27b, budget);
        assert(has_args(p27b, "--temp", "1.0") && has_args(p27b, "--top-k", "64") &&
               has_args(p27b, "--min-p", "0.05") && has_args(p27b, "--repeat-penalty", "1.0"));
    }

    // Hugging Face README + generation_config.json parser and sidecar override:
    {
        assert(ps5lm::clean_gguf_search_stem("Qwen3.5-2B-Claude-4.6.Q8_0.gguf") == "Qwen3.5-2B-Claude-4.6");
        assert(ps5lm::clean_gguf_search_stem("Model-00001-of-00004.gguf") == "Model");

        const ps5lm::FetchedPreset p_hightop = ps5lm::parse_hf_model_metadata(
            "DavidAU/Qwen3.6-12B-IQ-Ultra-Heretic-Uncensored-Thinking-V2-Hightop",
            R"({"temperature": 1.0, "top_k": 20, "top_p": 0.95})",
            "<B>SETTINGS:</B>\n- Temp 1, topp .95, minp .05, rep pen 1.05 [1 to 1.1 suggested]\n");
        assert(p_hightop.found && p_hightop.has_temp && std::fabs(p_hightop.temp - 1.0) < 1e-4);
        assert(p_hightop.has_top_k && p_hightop.top_k == 20);
        assert(p_hightop.has_min_p && std::fabs(p_hightop.min_p - 0.05) < 1e-4);
        assert(p_hightop.has_repeat_penalty && std::fabs(p_hightop.repeat_penalty - 1.05) < 1e-4);

        const ps5lm::FetchedPreset p_27b = ps5lm::parse_hf_model_metadata(
            "DavidAU/Qwen3.8-27B-Ultimate-Heretic-Uncensored-Turbo-Brilliance-50x-AGI-NEO-GGUF",
            "{}",
            "<b>Model Suggested Settings [from testing]:</b>\n"
            "- Temp .5 to 1 \n"
            "- Rep pen 1 [off]\n"
            "- TopK 64, min p: 0.05, topp: 0.95 (tester settings)\n"
            "The default reasoning mode is \"xhigh\". {REASON:deeptree} {REASON:einstein}\n",
            "{%- set default_effort = 'xhigh' -%}");
        assert(p_27b.found && std::fabs(p_27b.temp - 1.0) < 1e-4 && p_27b.top_k == 64);
        assert(std::fabs(p_27b.min_p - 0.05) < 1e-4 && std::fabs(p_27b.repeat_penalty - 1.0) < 1e-4);
        assert(p_27b.default_reasoning_effort == "xhigh");
        assert(std::find(p_27b.thinking_modes.begin(), p_27b.thinking_modes.end(), "einstein") != p_27b.thinking_modes.end());

        const auto q_9b = ps5lm::build_hf_search_queries(
            "Qwen3.5-9B-The-Defiant-Fable-Uncnr-Heretic-NEO-MAX-Q6_K.gguf");
        assert(q_9b.size() >= 3);
        assert(std::find(q_9b.begin(), q_9b.end(), "Qwen3.5-9B-The-Defiant-Fable-Uncensored-Heretic-NEO-MAX") !=
               q_9b.end());
        assert(std::find(q_9b.begin(), q_9b.end(), "Qwen3.5-9B-The-Defiant-Fable") != q_9b.end());

        const auto q_27b = ps5lm::build_hf_search_queries(
            "Qwen3.8-27B-Ultimate-Heretic-Uncn-Turbo-Brilli-50x-AGI-NEO-MTP-IQ2_M.gguf");
        assert(std::find(q_27b.begin(), q_27b.end(),
                         "Qwen3.8-27B-Ultimate-Heretic-Uncensored-Turbo-Brilliance-50x-AGI-NEO-MTP") !=
               q_27b.end());
        assert(std::find(q_27b.begin(), q_27b.end(), "Qwen3.8-27B-Ultimate-Heretic") != q_27b.end());

        const ps5lm::FetchedPreset p_table = ps5lm::parse_hf_model_metadata(
            "DavidAU/Qwen3.8-27B-Ultimate-Heretic-Uncensored-Turbo-Brilliance-50x-AGI-NEO-GGUF",
            "{}",
            "| **Reasoning Modes** | `deeptree`, `socrates`, `007`, `coder` |\n"
            "| **X-Combined Modes** | `xeinstein`, `xspoon` |\n"
            "- Temp .5 to 1 \n- Rep pen 1 [off]\n- TopK 64, min p: 0.05, topp: 0.95 (tester settings)\n");
        assert(std::find(p_table.thinking_modes.begin(), p_table.thinking_modes.end(), "007") !=
               p_table.thinking_modes.end());
        assert(std::find(p_table.thinking_modes.begin(), p_table.thinking_modes.end(), "xeinstein") !=
               p_table.thinking_modes.end());
        assert(p_table.sync_version == ps5lm::kPresetSyncVersion);

        char dir[] = "/tmp/preset_sidecar_test.XXXXXX";
        assert(mkdtemp(dir));
        const std::string model_p = std::string(dir) + "/CustomModel-Q4_K_M.gguf";
        assert(ps5lm::save_preset_sidecar(model_p, p_hightop));
        ps5lm::ModelInfo custom = c;
        custom.path             = model_p;
        custom.file_name        = "CustomModel-Q4_K_M.gguf";
        const ps5lm::Plan custom_plan = ps5lm::plan_model(custom, budget);
        assert(has_args(custom_plan, "--repeat-penalty", "1.05") && has_args(custom_plan, "--min-p", "0.05"));
        unlink((model_p + ".preset.json").c_str());
        rmdir(dir);
    }

    // Nothing fits in too little memory.
    ps5lm::ModelInfo big;
    big.ok = true; big.file_bytes = 20ull << 30; big.n_layer = 32; big.n_head_kv = 8;
    big.key_length = big.value_length = 128;
    assert(!ps5lm::plan_model(big, budget).fits);
    // "auto" / "" kv_force normalizes to best-fit even when disk cache is off.
    const ps5lm::Plan auto_kv_off = ps5lm::plan_model(c, budget, 65536, "auto", ps5lm::kPromptUbatch, nullptr, "off");
    assert(auto_kv_off.fits && auto_kv_off.ctx == cp.ctx && auto_kv_off.kv_type == cp.kv_type);
    // Invalid disk_cache_mode falls back to auto; disk_cache_mib == 0 omits --cache-disk.
    const ps5lm::Plan fb = ps5lm::plan_model(c, budget, 65536, "auto", ps5lm::kPromptUbatch, nullptr, "bogus", 2048);
    assert(fb.fits && fb.kv_type == cp.kv_type && has_args(fb, "--cache-disk", "/data/PS5LM/cache/prompts") &&
           has_args(fb, "--cache-disk-mib", "2048"));
    const ps5lm::Plan zero_mib = ps5lm::plan_model(c, budget, 65536, "auto", ps5lm::kPromptUbatch, nullptr, "on", 0);
    assert(zero_mib.fits && std::find(zero_mib.args.begin(), zero_mib.args.end(), "--cache-disk") == zero_mib.args.end());
    // Large or small disk cache limits strictly honor Longest context (ctx_cap) and keep spillover enabled across 2..32 GB.
    const ps5lm::Plan big_disk = ps5lm::plan_model(c, budget, 65536, "auto", ps5lm::kPromptUbatch, nullptr, "auto", 32768);
    assert(big_disk.fits && big_disk.ctx == 65536 && has_args(big_disk, "--cache-disk-mib", "32768"));
    const ps5lm::Plan cap_128k = ps5lm::plan_model(c, budget, 131072, "auto", ps5lm::kPromptUbatch, nullptr, "auto", 4096);
    assert(cap_128k.fits && cap_128k.ctx == 131072 && has_args(cap_128k, "--cache-disk-mib", "4096"));
    // When p.ctx < ctx_cap even with >0.5 GiB initial headroom, tight compute compaction unlocks longer context.
    ps5lm::ModelInfo iq2_27b;
    iq2_27b.ok = true; iq2_27b.file_bytes = 10243816864ull; iq2_27b.n_layer = 64; iq2_27b.full_attn_every = 4;
    iq2_27b.n_head_kv = 4; iq2_27b.key_length = iq2_27b.value_length = 256; iq2_27b.state_bytes = 160ull << 20;
    const ps5lm::Plan p27_q8 = ps5lm::plan_model(iq2_27b, 11.74, 65536, "q8_0", ps5lm::kPromptUbatch, nullptr, "auto", 2048, "/data/PS5LM/cache/prompts", true);
    assert(p27_q8.fits && p27_q8.ctx == 32768 && p27_q8.kv_type == "q8_0" && p27_q8.disable_introspect);
    const ps5lm::Plan p27_auto = ps5lm::plan_model(iq2_27b, 11.74, 65536, "auto", ps5lm::kPromptUbatch, nullptr, "auto", 2048, "/data/PS5LM/cache/prompts", true);
    assert(p27_auto.fits && p27_auto.ctx == 32768 && p27_auto.kv_type == "q4_0");

    if (argc < 2) {
        std::puts("model_plan_test: ok (synthetic only)");
        return 0;
    }
    bool saw_q2k = false, saw_iq2xxs = false;
    for (const auto & path : ps5lm::find_models({ argv[1] })) {
        const ps5lm::ModelInfo m = ps5lm::read_model_info(path);
        assert(m.ok);
        const ps5lm::Plan p = ps5lm::plan_model(m, budget);
        const ps5lm::Preset * pre = ps5lm::find_preset(m);
        std::printf("%-34s %s L%u kvL%u kvh%u k%u v%u state %.0f MiB | %s | %s\n", m.file_name.c_str(),
                    m.arch.c_str(), m.n_layer, m.n_kv_layers(), m.n_head_kv, m.key_length, m.value_length,
                    m.state_bytes / 1048576.0, p.why.c_str(), pre ? pre->label : "-");
        if (m.arch == "qwen35") {
            // Qwen 3.8 27B: 16 of 64 layers keep a KV cache, 4 heads of 256.
            assert(m.n_layer == 64 && m.n_kv_layers() == 16 && m.n_head_kv == 4 && m.key_length == 256);
            assert(kv_bytes_per_token(m, "f16") == 65536.0);
            assert(m.state_bytes > 140.0 * 1048576 && m.state_bytes < 170.0 * 1048576);
            assert(pre && std::string(pre->label) == "Qwen 3.8");
        }
        if (m.file_name.find("UD-Q2_K_XL") != std::string::npos) {
            // Ran on the console: 64k with a q4_0 cache.
            assert(p.fits && p.ctx == 65536 && p.kv_type == "q4_0");
            saw_q2k = true;
        }
        if (m.file_name.find("UD-IQ2_XXS") != std::string::npos) {
            // 4k at f16 ran; 64k needs at most q8_0 at this size.
            assert(p.fits && p.ctx == 65536 && p.kv_type == "q8_0");
            saw_iq2xxs = true;
        }
    }
    assert(saw_q2k && saw_iq2xxs);
    std::puts("model_plan_test: ok");
}
