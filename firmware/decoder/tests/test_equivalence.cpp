// Fixture-equivalence runner: streams every fixture's raw.csv sample by sample through a
// fresh Decoder for both configs and compares against the frozen Python outputs
// (filtered_<cfg>.csv and expected.json, flattened to points_<cfg>.csv by
// firmware/tools/flatten_fixtures.py). Prints a summary table and writes it as Markdown.
//
// Gates (PLAN.md phases 1-3):
//   double: filtered abs err <= 1e-12 V; scores / all_correlations / margin <= 1e-9;
//           winners and every decision 100 %.
//   float:  filtered abs err <= 1e-3 * signal RMS; scores / all_correlations / margin
//           < 1e-3; winner and ungated decision 100 %; test-only gated decisions 100 %
//           except points with |score - T| < 1e-3 or |margin - M| < 1e-3 (counted, listed).
//   both:   degenerate status exactly where Python raised DegenerateSignalError; no NaN.
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "synapsys/config_v1.hpp"
#include "synapsys/decoder.hpp"

using namespace synapsys;

namespace {

constexpr bool kIsDouble = sizeof(Real) == sizeof(double);
constexpr double kFilterAbsTol = 1e-12;   // double build, volts
constexpr double kFilterRelTol = 1e-3;    // float build, x signal RMS
constexpr double kScoreTol = kIsDouble ? 1e-9 : 1e-3;
constexpr double kNearThreshold = 1e-3;

// Input path: push_f64() (float64 volts, DC removed in double -- the gated path) or push()
// with the samples first rounded to Real (informational for the float build: the filter
// gate is reported but not enforced, because rounding a large electrode offset to float32
// before DC removal is an input-representation limit, not a decoder error).
bool g_real_input = false;

using Rows = std::vector<std::vector<double>>;

// Minimal CSV reader: skips the header line, parses numbers with strtod (no exceptions).
bool read_csv(const std::string& path, Rows* rows) {
  FILE* f = std::fopen(path.c_str(), "r");
  if (!f) return false;
  std::string line;
  char buf[8192];
  bool header = true;
  while (std::fgets(buf, sizeof buf, f)) {
    line.assign(buf);
    while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) line.pop_back();
    if (line.empty()) continue;
    if (header) {
      header = false;
      continue;
    }
    std::vector<double> row;
    const char* p = line.c_str();
    while (*p) {
      char* end = nullptr;
      row.push_back(std::strtod(p, &end));
      if (end == p) {
        std::fclose(f);
        return false;
      }
      p = end;
      if (*p == ',') ++p;
    }
    rows->push_back(row);
  }
  std::fclose(f);
  return true;
}

struct ExpectedPoint {
  int window_end = 0;
  bool error = false;
  std::array<double, kTargets> scores{};
  int winner = kReject;
  double confidence = 0, margin = 0;
  int decision_ungated = kReject;
  std::array<int, kTargets> ncorr{};
  std::array<std::array<double, 4>, kTargets> corr{};
  std::vector<std::array<double, 3>> gated;  // T, M, decision
};

std::vector<ExpectedPoint> parse_points(const Rows& rows) {
  std::vector<ExpectedPoint> pts;
  for (const auto& r : rows) {
    ExpectedPoint p;
    size_t k = 0;
    p.window_end = static_cast<int>(r[k++]);
    p.error = r[k++] != 0.0;
    for (int t = 0; t < kTargets; ++t) p.scores[t] = r[k++];
    p.winner = static_cast<int>(r[k++]);
    p.confidence = r[k++];
    p.margin = r[k++];
    p.decision_ungated = static_cast<int>(r[k++]);
    for (int t = 0; t < kTargets; ++t) {
      p.ncorr[t] = static_cast<int>(r[k++]);
      for (int j = 0; j < 4; ++j) p.corr[t][j] = r[k++];
    }
    while (k + 2 < r.size()) {
      p.gated.push_back({r[k], r[k + 1], r[k + 2]});
      k += 3;
    }
    pts.push_back(p);
  }
  return pts;
}

struct Summary {
  std::string fixture, cfg;
  double max_filter_err = 0, signal_rms = 0, filter_limit = 0;
  double max_score_err = 0, max_corr_err = 0, max_margin_err = 0;
  int points = 0, degenerate_expected = 0, degenerate_match = 0;
  int winner_ok = 0, winner_total = 0;
  int ungated_ok = 0, ungated_total = 0;
  int gated_ok = 0, gated_total = 0;
  int near_threshold = 0, near_threshold_mismatch = 0;
  int hard_failures = 0;
  std::vector<std::string> notes;
};

void note(Summary* s, const char* fmt, double a = 0, double b = 0, double c = 0, double d = 0) {
  char buf[256];
  std::snprintf(buf, sizeof buf, fmt, a, b, c, d);
  s->notes.emplace_back(buf);
}

Summary run_one(const std::string& fixtures, const std::string& generated,
                const std::string& fixture, const char* cfg_name, const Config& cfg) {
  Summary s;
  s.fixture = fixture;
  s.cfg = cfg_name;
  Rows raw, filt, pts_rows;
  const std::string fdir = fixtures + "/" + fixture;
  if (!read_csv(fdir + "/raw.csv", &raw) ||
      !read_csv(fdir + "/filtered_" + cfg_name + ".csv", &filt) ||
      !read_csv(generated + "/" + fixture + "/points_" + cfg_name + ".csv", &pts_rows) ||
      raw.size() != filt.size() || raw.empty()) {
    ++s.hard_failures;
    s.notes.emplace_back("could not read fixture files");
    return s;
  }
  const auto expected = parse_points(pts_rows);
  s.points = static_cast<int>(expected.size());

  double sumsq = 0;
  for (const auto& r : filt)
    for (double v : r) sumsq += v * v;
  s.signal_rms = std::sqrt(sumsq / (filt.size() * kChannels));
  s.filter_limit = kIsDouble ? kFilterAbsTol : kFilterRelTol * s.signal_rms;

  auto dec = std::make_unique<Decoder>(cfg);
  size_t next = 0;  // next expected point
  for (size_t i = 0; i < raw.size(); ++i) {
    WindowResult res;
    bool due;
    if (g_real_input) {
      Real x[kChannels];
      for (int ch = 0; ch < kChannels; ++ch) x[ch] = static_cast<Real>(raw[i][ch]);
      due = dec->push(x, &res);
    } else {
      double x[kChannels];
      for (int ch = 0; ch < kChannels; ++ch) x[ch] = raw[i][ch];
      due = dec->push_f64(x, &res);
    }
    for (int ch = 0; ch < kChannels; ++ch) {
      const double got = static_cast<double>(dec->last_filtered()[ch]);
      const double err = std::fabs(got - filt[i][ch]);
      if (!(err <= s.max_filter_err) || std::isnan(got)) s.max_filter_err = std::isnan(got) ? INFINITY : err;
    }
    const int seen = static_cast<int>(i) + 1;
    const bool expected_here = next < expected.size() && expected[next].window_end == seen;
    if (due && !expected_here) {
      ++s.hard_failures;
      note(&s, "unexpected decision point at %g", seen);
      continue;
    }
    if (!expected_here) continue;
    if (!due) res = dec->evaluate();  // off-schedule point (Wang benchmark window, 785)
    const ExpectedPoint& e = expected[next++];
    if (res.window_end_sample != static_cast<uint32_t>(seen)) ++s.hard_failures;

    for (int t = 0; t < kTargets; ++t) {
      if (!std::isfinite(static_cast<double>(res.scores[t]))) {
        ++s.hard_failures;
        note(&s, "NaN/Inf score at %g", seen);
      }
    }
    if (e.error) {
      ++s.degenerate_expected;
      if (res.status == Status::kDegenerate && res.decision == kReject) {
        ++s.degenerate_match;
      } else {
        ++s.hard_failures;
        note(&s, "expected DegenerateSignalError at %g, got status ok", seen);
      }
      continue;
    }
    if (res.status != Status::kOk) {
      ++s.hard_failures;
      note(&s, "unexpected degenerate status at %g", seen);
      continue;
    }
    for (int t = 0; t < kTargets; ++t) {
      s.max_score_err =
          std::fmax(s.max_score_err, std::fabs(static_cast<double>(res.scores[t]) - e.scores[t]));
      if (res.n_correlations[t] != e.ncorr[t]) {
        ++s.hard_failures;
        note(&s, "n_correlations %g vs %g at %g", res.n_correlations[t], e.ncorr[t], seen);
      }
      for (int j = 0; j < e.ncorr[t] && j < kMaxCorr; ++j) {
        s.max_corr_err = std::fmax(
            s.max_corr_err, std::fabs(static_cast<double>(res.all_correlations[t][j]) - e.corr[t][j]));
      }
    }
    s.max_margin_err =
        std::fmax(s.max_margin_err, std::fabs(static_cast<double>(res.margin) - e.margin));
    ++s.winner_total;
    if (res.winner == e.winner) {
      ++s.winner_ok;
    } else {
      note(&s, "winner %g vs python %g at %g", res.winner, e.winner, seen);
    }
    ++s.ungated_total;
    if (res.decision == e.decision_ungated && !res.gated) ++s.ungated_ok;

    for (const auto& g : e.gated) {
      Config gc = cfg;
      gc.has_confidence_threshold = true;
      gc.confidence_threshold = g[0];
      gc.has_margin_threshold = true;
      gc.margin_threshold = g[1];
      WindowResult gr;
      decide(res.scores, gc, &gr);
      const bool near = std::fabs(e.confidence - g[0]) < kNearThreshold ||
                        std::fabs(e.margin - g[1]) < kNearThreshold;
      const bool match = gr.decision == static_cast<int>(g[2]);
      ++s.gated_total;
      if (match) ++s.gated_ok;
      if (near) {
        ++s.near_threshold;
        if (!match) ++s.near_threshold_mismatch;
        char buf[256];
        std::snprintf(buf, sizeof buf,
                      "near-threshold point end=%d T=%g M=%g (python score %.6f margin %.6f; "
                      "C++ %.6f / %.6f): %s",
                      seen, g[0], g[1], e.confidence, e.margin,
                      static_cast<double>(res.confidence), static_cast<double>(res.margin),
                      match ? "agrees" : "DIFFERS");
        s.notes.emplace_back(buf);
      }
    }
  }
  if (next != expected.size()) {
    ++s.hard_failures;
    note(&s, "%g expected points never reached", static_cast<double>(expected.size() - next));
  }
  return s;
}

bool passes(const Summary& s, std::string* why) {
  std::string w;
  if (s.hard_failures) w += "hard failures; ";
  // Filter gate: enforced except in the float build's Real-input (informational) mode.
  const bool enforce_filter = !(g_real_input && !kIsDouble);
  if (enforce_filter && !(s.max_filter_err <= s.filter_limit)) w += "filter; ";
  if (kIsDouble) {
    if (!(s.max_score_err <= kScoreTol)) w += "score; ";
    if (!(s.max_corr_err <= kScoreTol)) w += "all_correlations; ";
    if (!(s.max_margin_err <= kScoreTol)) w += "margin; ";
  } else {
    if (!(s.max_score_err < kScoreTol)) w += "score; ";
    if (!(s.max_corr_err < kScoreTol)) w += "all_correlations; ";
    if (!(s.max_margin_err < kScoreTol)) w += "margin; ";
  }
  if (s.winner_ok != s.winner_total) w += "winner; ";
  if (s.ungated_ok != s.ungated_total) w += "ungated decision; ";
  if (s.degenerate_match != s.degenerate_expected) w += "degenerate; ";
  const int gated_bad = s.gated_total - s.gated_ok;
  if (kIsDouble ? gated_bad != 0 : gated_bad != s.near_threshold_mismatch) w += "gated decision; ";
  *why = w;
  return w.empty();
}

}  // namespace

int main(int argc, char** argv) {
  std::string fixtures, generated, report, rev = "unknown";
  for (int i = 1; i + 1 < argc; i += 2) {
    if (!std::strcmp(argv[i], "--fixtures")) fixtures = argv[i + 1];
    else if (!std::strcmp(argv[i], "--generated")) generated = argv[i + 1];
    else if (!std::strcmp(argv[i], "--report")) report = argv[i + 1];
    else if (!std::strcmp(argv[i], "--rev")) rev = argv[i + 1];
    else if (!std::strcmp(argv[i], "--input")) g_real_input = !std::strcmp(argv[i + 1], "real");
  }
  if (fixtures.empty() || generated.empty()) {
    std::fprintf(stderr, "usage: %s --fixtures DIR --generated DIR [--report FILE]\n", argv[0]);
    return 2;
  }
  std::vector<std::string> ids;
  {
    FILE* f = std::fopen((generated + "/index.txt").c_str(), "r");
    if (!f) {
      std::fprintf(stderr, "missing %s/index.txt (run flatten_fixtures.py)\n", generated.c_str());
      return 2;
    }
    char buf[512];
    while (std::fgets(buf, sizeof buf, f)) {
      std::string s(buf);
      while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) s.pop_back();
      if (!s.empty()) ids.push_back(s);
    }
    std::fclose(f);
  }

  const char* cfg_names[2] = {"synapsys_v1", "benchmark_v1"};
  const Config cfgs[2] = {config_synapsys_v1(), config_benchmark_v1()};
  std::vector<Summary> all;
  for (const auto& id : ids)
    for (int c = 0; c < 2; ++c) all.push_back(run_one(fixtures, generated, id, cfg_names[c], cfgs[c]));

  const char* prec = kIsDouble ? "double" : "float";
  std::string md;
  char line[1024];
  const char* mode = g_real_input ? "push(): samples rounded to Real before DC removal"
                                  : "push_f64(): float64 volts, DC removed in double";
  std::snprintf(line, sizeof line,
                "# Fixture equivalence: C++ decoder (%s) vs Python reference v1\n\n"
                "Generated by `test_equivalence_%s` (input path: %s%s); source rev `%s`.\n"
                "Fixtures: %zu (synthetic + local Wang), 2 configs each. Gates: filter %s; "
                "score / all_correlations / margin %s %g; winner and ungated decision 100 %%; "
                "gated test-only decisions 100 %%%s.\n\n",
                prec, prec, mode,
                (g_real_input && !kIsDouble) ? "; filter column informational, not gated" : "",
                rev.c_str(), ids.size(),
                kIsDouble ? "<= 1e-12 V" : "<= 1e-3 x signal RMS", kIsDouble ? "<=" : "<",
                kScoreTol, kIsDouble ? "" : " except near-threshold points (listed)");
  md += line;
  md += "| fixture | config | pts | max filter err (V) | filter limit (V) | max score err | "
        "max corr err | max margin err | winner | ungated | gated | near-thr (differ) | "
        "degenerate | result |\n";
  md += "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n";
  int failed = 0, total_near = 0, total_near_diff = 0;
  double w_filter_rel = 0, w_score = 0, w_corr = 0, w_margin = 0, w_filter = 0;
  for (const auto& s : all) {
    std::string why;
    const bool ok = passes(s, &why);
    if (!ok) ++failed;
    total_near += s.near_threshold;
    total_near_diff += s.near_threshold_mismatch;
    w_filter = std::fmax(w_filter, s.max_filter_err);
    if (s.signal_rms > 0) w_filter_rel = std::fmax(w_filter_rel, s.max_filter_err / s.signal_rms);
    w_score = std::fmax(w_score, s.max_score_err);
    w_corr = std::fmax(w_corr, s.max_corr_err);
    w_margin = std::fmax(w_margin, s.max_margin_err);
    std::snprintf(line, sizeof line,
                  "| %s | %s | %d | %.3g | %.3g | %.3g | %.3g | %.3g | %d/%d | %d/%d | %d/%d | "
                  "%d (%d) | %d/%d | %s%s |\n",
                  s.fixture.c_str(), s.cfg.c_str(), s.points, s.max_filter_err, s.filter_limit,
                  s.max_score_err, s.max_corr_err, s.max_margin_err, s.winner_ok, s.winner_total,
                  s.ungated_ok, s.ungated_total, s.gated_ok, s.gated_total, s.near_threshold,
                  s.near_threshold_mismatch, s.degenerate_match, s.degenerate_expected,
                  ok ? "PASS" : "FAIL: ", why.c_str());
    md += line;
  }
  std::snprintf(line, sizeof line,
                "\n**Worst case** over all fixtures: filter err %.3g V (%.3g x signal RMS), "
                "score err %.3g, all_correlations err %.3g, margin err %.3g. "
                "Near-threshold gated decisions: %d (of which differing from Python: %d). "
                "Result: **%s** (%d / %zu fixture x config failing).\n",
                w_filter, w_filter_rel, w_score, w_corr, w_margin, total_near, total_near_diff,
                failed ? "FAIL" : "PASS", failed, all.size());
  md += line;
  md += "\n## Notes (near-threshold points and any failures)\n\n";
  bool any_note = false;
  for (const auto& s : all) {
    for (const auto& n : s.notes) {
      md += "- " + s.fixture + " / " + s.cfg + ": " + n + "\n";
      any_note = true;
    }
  }
  if (!any_note) md += "- none\n";

  std::fputs(md.c_str(), stdout);
  if (!report.empty()) {
    FILE* f = std::fopen(report.c_str(), "w");
    if (f) {
      std::fputs(md.c_str(), f);
      std::fclose(f);
    } else {
      std::fprintf(stderr, "could not write %s\n", report.c_str());
    }
  }
  return failed == 0 && !all.empty() ? 0 : 1;
}
