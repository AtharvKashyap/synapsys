// Replays fixtures/rearm/sequences.csv (generated from the Python reference) through the
// C++ Rearm and requires identical emits and states at every step.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "synapsys/rearm.hpp"

using synapsys::kReject;

static int label_index(const std::string& s) {
  if (s == "YES") return 0;
  if (s == "NO") return 1;
  if (s == "HELP") return 2;
  if (s == "REJECT" || s.empty()) return kReject;
  std::fprintf(stderr, "unknown label '%s'\n", s.c_str());
  std::exit(2);
}

int main(int argc, char** argv) {
  const char* path = argc > 1 ? argv[1] : "fixtures/rearm/sequences.csv";
  FILE* f = std::fopen(path, "r");
  if (!f) {
    std::fprintf(stderr, "cannot open %s\n", path);
    return 2;
  }
  char line[512];
  if (!std::fgets(line, sizeof line, f)) return 2;  // header
  std::string current;
  synapsys::Rearm rearm(synapsys::RearmConfig{});
  long rows = 0, emits = 0, mismatches = 0, sequences = 0;
  while (std::fgets(line, sizeof line, f)) {
    line[std::strcspn(line, "\r\n")] = 0;
    std::string fields[9];
    int n = 0;
    for (char* p = line; n < 9; ++n) {
      char* c = std::strchr(p, ',');
      fields[n] = c ? std::string(p, c - p) : std::string(p);
      if (!c) { ++n; break; }
      p = c + 1;
    }
    if (n != 9) {
      std::fprintf(stderr, "bad row: %s\n", line);
      return 2;
    }
    if (fields[0] != current) {
      current = fields[0];
      synapsys::RearmConfig cfg;
      cfg.confirm_count = std::atoi(fields[1].c_str());
      cfg.refractory_samples = static_cast<uint32_t>(std::strtoul(fields[2].c_str(), nullptr, 10));
      cfg.release_count = std::atoi(fields[3].c_str());
      rearm = synapsys::Rearm(cfg);
      ++sequences;
    }
    const uint32_t t = static_cast<uint32_t>(std::strtoul(fields[5].c_str(), nullptr, 10));
    const int got = rearm.update(label_index(fields[6]), t);
    const int want = label_index(fields[7]);
    ++rows;
    emits += want != kReject;
    if (got != want || fields[8] != rearm.state_name()) {
      if (++mismatches <= 10) {
        std::fprintf(stderr, "MISMATCH %s step %s: emit got %d want %d, state got %s want %s\n",
                     fields[0].c_str(), fields[4].c_str(), got, want, rearm.state_name(),
                     fields[8].c_str());
      }
    }
  }
  std::fclose(f);
  std::printf("rearm: %ld sequences, %ld steps, %ld expected emits, %ld mismatches\n", sequences,
              rows, emits, mismatches);
  return (mismatches == 0 && rows > 0) ? 0 : 1;
}
