// Regression test for DKD threshold-mode selection (top_k <= 0).
//
// Contract: eligible candidates are NMS peaks outside the border with score > scores_th. If there are more than
// n_limit of them, exactly n_limit are kept and none is weaker than an omitted one. Checked on both the sub-pixel
// and the integer branch, on CPU, with isolated single-pixel peaks so every output maps to one known peak.
#include "dkd.hpp"

#include <torch/torch.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <exception>
#include <map>
#include <utility>
#include <vector>

namespace {
constexpr int kSize = 64;
constexpr int kSpacing = 8;
constexpr float kTh = 0.2f;

struct Peak { int y, x; float score; };

// m eligible peaks in raster order, one peak exactly at the threshold, and one high-scoring peak inside the
// 2-px border. Scores are distinct and spatially nonmonotone unless `given` supplies them (for tied scores).
std::pair<torch::Tensor, std::vector<Peak>> makeMap(int m, const std::vector<float>& given) {
  auto map = torch::zeros({1, 1, kSize, kSize});
  auto a = map.accessor<float, 4>();
  std::vector<Peak> eligible;
  int k = 0;
  for (int y = 4; y < kSize - 4 && k <= m; y += kSpacing)
    for (int x = 4; x < kSize - 4 && k <= m; x += kSpacing, ++k) {
      const float s = k < m && !given.empty() ? given[k]
                      : k < m ? 0.3f + 0.6f * static_cast<float>((k * 7) % std::max(m, 1)) / std::max(m, 1) +
                                  0.0001f * static_cast<float>(k)
                            : kTh;
      a[0][0][y][x] = s;
      if (k < m) eligible.push_back({y, x, s});
    }
  a[0][0][1][20] = 0.99f;
  return {map, eligible};
}

int checkImpl(int m, int n, bool subPixel, const std::vector<float>& given) {
  auto [map, eligible] = makeMap(m, given);
  DKD dkd(2, -1, kTh, n);
  auto [kpts, disp, scores] = std::move(dkd).forward(map, subPixel);
  const auto k = kpts[0];
  std::map<std::pair<int, int>, float> byPos;
  for (const auto& p : eligible) byPos[{p.y, p.x}] = p.score;
  std::vector<float> kept;
  for (int64_t i = 0; i < k.size(0); ++i) {
    const int x = static_cast<int>(std::lround((k[i][0].item<float>() + 1) / 2 * (kSize - 1)));
    const int y = static_cast<int>(std::lround((k[i][1].item<float>() + 1) / 2 * (kSize - 1)));
    const auto f = byPos.find({y, x});
    if (f == byPos.end()) { std::printf("FAIL m=%d n=%d sub_pixel=%d: row %lld is not an eligible peak\n", m, n, subPixel, (long long)i); return 1; }
    kept.push_back(f->second);
    byPos.erase(f);
  }
  const int want = std::min(m, n);
  float weakestKept = kept.empty() ? 1e9f : *std::min_element(kept.begin(), kept.end());
  float strongestOmitted = -1e9f;
  for (const auto& [pos, s] : byPos) strongestOmitted = std::max(strongestOmitted, s);
  const bool ok = static_cast<int>(kept.size()) == want && (byPos.empty() || weakestKept >= strongestOmitted) &&
                  disp[0].size(0) == k.size(0) && scores[0].size(0) == k.size(0);
  std::printf("%s m=%d n=%d sub_pixel=%d returned=%zu\n", ok ? "ok  " : "FAIL", m, n, subPixel, kept.size());
  return ok ? 0 : 1;
}
int check(int m, int n, bool subPixel, const std::vector<float>& given = {}) {
  try {
    return checkImpl(m, n, subPixel, given);
  } catch (const std::exception& error) {
    std::printf("FAIL m=%d n=%d sub_pixel=%d: %s\n", m, n, subPixel, error.what());
    return 1;
  }
}
}  // namespace

int main() {
  int failures = 0;
  for (const bool subPixel : {false, true})
    for (const auto& [m, n] : std::vector<std::pair<int, int>>{{0, 4}, {2, 4}, {4, 4}, {5, 4}, {6, 4}, {11, 4}, {40, 13}})
      failures += check(m, n, subPixel);
  // Tied scores: at the cutoff (any one of the three 0.6 peaks may fill the last slot) and all equal.
  const std::vector<float> cutoffTie{0.9f, 0.8f, 0.7f, 0.6f, 0.6f, 0.6f, 0.5f};
  const std::vector<float> allEqual(6, 0.5f);
  for (const bool subPixel : {false, true}) {
    failures += check(static_cast<int>(cutoffTie.size()), 4, subPixel, cutoffTie);
    failures += check(static_cast<int>(allEqual.size()), 4, subPixel, allEqual);
  }
  std::printf("%d failure(s)\n", failures);
  return failures == 0 ? 0 : 1;
}
