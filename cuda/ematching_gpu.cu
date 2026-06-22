// "E-matching is a relational join" benchmark — six matchers, CPU and GPU.
//
// Pattern  f(α, g(α))  →  Q(root, α) ← R_f(root, α, x), R_g(x, α).
//
//   cpu_backtracking  Θ(N²)  prover-style nested loop (baseline)            [CPU]
//   cpu_hash_join     Θ(N)   std::unordered_set index on (x, α), probe      [CPU]
//   cpu_lftj          Θ(NlogN) Leapfrog Triejoin (worst-case-optimal)       [CPU]
//   gpu_nested_loop   Θ(N²)  one thread per f-row, scan R_g                 [GPU]
//   gpu_build+probe   Θ(N)   open-addressing hash table on (x, α)           [GPU]
//   gpu_lftj          Θ(NlogN) Leapfrog Triejoin, parallelized over α       [GPU]
//
// METHODOLOGY — each variant is benchmarked END-TO-END and SELF-CONTAINED, with
// NO reuse between variants:
//   * The input relations are generated in RANDOM (unsorted) order, so no method
//     free-rides on sortedness.
//   * Every variant's timed region starts from those raw arrays and includes ALL
//     the work that method needs: build its own hash table / sort its own trie;
//     each GPU variant does its OWN host→device upload + kernel + device→host
//     download. Nothing (no sorted arrays, no hash table, no upload) is shared.
//   * Timing is wall-clock (std::chrono), min of a few reps. Device-buffer
//     allocation is outside the timed region (infrastructure, not algorithm).
//   * gpu_lftj needs trie-sorted input; since we forbid reuse, it sorts on the
//     HOST inside its own timed region (a pure-GPU engine would sort on-device
//     with thrust/cub — noted; the host sort is what we can verify here).
//
// Correctness: the answer is known in closed form to be exactly
// {(I_F, k) : k=1..N}. Every variant's full output set is verified against it and
// the program aborts on any mismatch. (CPU matchers + the gpu_lftj logic were
// validated against this truth with a host compiler, incl. on shuffled input;
// the CUDA execution paths self-verify on first run on real hardware.)
//
// Build & run:
//   nvcc -O3 -std=c++14 -arch=native -o ematching_gpu cuda/ematching_gpu.cu
//   ./ematching_gpu                # default sweep
//   ./ematching_gpu 2097152 32768  # max-N for O(N) matchers, cap for O(N²)

#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

// Result e-class ids for f / g. Set far above any feasible N (~10^9) so the α
// ids (1..N) never numerically collide with them even for huge sweeps — cosmetic
// only (matching is positional), but keeps large-N data unambiguous. Both < i64
// max (~9.2e18). Kernels index with int, so keep N <= ~2^30.
static const int64_t I_F = 4000000000000000000LL;
static const int64_t I_G = 5000000000000000000LL;
static const int64_t EMPTY = -1;

static const int CPU_BT_REPS = 3;
static const int CPU_REPS = 5;
static const int GPU_REPS = 3;

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _e = (call);                                                \
        if (_e != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,       \
                    cudaGetErrorString(_e));                                    \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

// ---- device helpers --------------------------------------------------------

__device__ __forceinline__ uint64_t hash2(int64_t kx, int64_t ka) {
    uint64_t h = (uint64_t)kx * 0x9E3779B97F4A7C15ULL;
    h ^= (uint64_t)ka + 0x9E3779B97F4A7C15ULL + (h << 6) + (h >> 2);
    h ^= h >> 33; h *= 0xff51afd7ed558ccdULL; h ^= h >> 33;
    return h;
}
__device__ __forceinline__ int dev_lb(const int64_t* a, int lo, int hi, int64_t v) {
    while (lo < hi) { int m = (lo + hi) >> 1; if (a[m] < v) lo = m + 1; else hi = m; }
    return lo;
}
__device__ __forceinline__ int dev_ub(const int64_t* a, int lo, int hi, int64_t v) {
    while (lo < hi) { int m = (lo + hi) >> 1; if (a[m] <= v) lo = m + 1; else hi = m; }
    return lo;
}

// ---- GPU kernels -----------------------------------------------------------

__global__ void gpu_nested_loop(const int64_t* f_id, const int64_t* f_a1,
                                const int64_t* f_a2, int n_f, const int64_t* g_id,
                                const int64_t* g_a1, int n_g, int64_t* out_root,
                                int64_t* out_alpha, int* out_count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_f) return;
    int64_t root = f_id[i], fa = f_a1[i], fx = f_a2[i];
    for (int j = 0; j < n_g; ++j)
        if (fx == g_id[j] && fa == g_a1[j]) {
            int idx = atomicAdd(out_count, 1);
            out_root[idx] = root; out_alpha[idx] = fa;
        }
}

__global__ void gpu_build(const int64_t* g_id, const int64_t* g_a1, int n_g,
                          int64_t* keys_x, int64_t* keys_a, uint64_t cap) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_g) return;
    int64_t kx = g_id[i], ka = g_a1[i];
    uint64_t slot = hash2(kx, ka) & (cap - 1);
    const unsigned long long empty_u = (unsigned long long)EMPTY;
    while (true) {
        unsigned long long prev = atomicCAS((unsigned long long*)&keys_x[slot],
                                            empty_u, (unsigned long long)kx);
        if (prev == empty_u) { keys_a[slot] = ka; return; }
        slot = (slot + 1) & (cap - 1);
    }
}

__global__ void gpu_probe(const int64_t* f_id, const int64_t* f_a1,
                          const int64_t* f_a2, int n_f, const int64_t* keys_x,
                          const int64_t* keys_a, uint64_t cap, int64_t* out_root,
                          int64_t* out_alpha, int* out_count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_f) return;
    int64_t kx = f_a2[i], ka = f_a1[i];
    uint64_t slot = hash2(kx, ka) & (cap - 1);
    while (keys_x[slot] != EMPTY) {
        if (keys_x[slot] == kx && keys_a[slot] == ka) {
            int idx = atomicAdd(out_count, 1);
            out_root[idx] = f_id[i]; out_alpha[idx] = ka;
            return;
        }
        slot = (slot + 1) & (cap - 1);
    }
}

// Leapfrog Triejoin on trie-sorted input (fa=α, fx=x, fr=root; ga=α, gx=x),
// parallelized over α (one thread per α-run). Leapfrog-intersect x via galloping
// binary search, then enumerate root.
__global__ void gpu_lftj(const int64_t* fa, const int64_t* fx, const int64_t* fr,
                         int n_f, const int64_t* ga, const int64_t* gx, int n_g,
                         int64_t* out_root, int64_t* out_alpha, int* out_count) {
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n_f) return;
    int64_t alpha = fa[t];
    if (t > 0 && fa[t - 1] == alpha) return;
    int gLo = dev_lb(ga, 0, n_g, alpha), gHi = dev_ub(ga, gLo, n_g, alpha);
    if (gLo >= gHi) return;
    int fHi = dev_ub(fa, t, n_f, alpha);
    int p = t, q = gLo;
    while (p < fHi && q < gHi) {
        int64_t xf = fx[p], xg = gx[q];
        if (xf < xg) { p = dev_lb(fx, p + 1, fHi, xg); continue; }
        if (xg < xf) { q = dev_lb(gx, q + 1, gHi, xf); continue; }
        int64_t xv = xf;
        int fxHi = dev_ub(fx, p, fHi, xv);
        for (int r = p; r < fxHi; ++r)
            if (r == p || fr[r - 1] != fr[r]) {
                int idx = atomicAdd(out_count, 1);
                out_root[idx] = fr[r]; out_alpha[idx] = alpha;
            }
        p = fxHi; q = dev_ub(gx, q, gHi, xv);
    }
}

// ---- CPU matchers ----------------------------------------------------------

static int cpu_backtracking(const std::vector<int64_t>& f_id, const std::vector<int64_t>& f_a1,
                            const std::vector<int64_t>& f_a2, const std::vector<int64_t>& g_id,
                            const std::vector<int64_t>& g_a1, std::vector<int64_t>& out_root,
                            std::vector<int64_t>& out_alpha) {
    int n_f = (int)f_id.size(), n_g = (int)g_id.size(), count = 0;
    for (int i = 0; i < n_f; ++i) {
        int64_t root = f_id[i], fa = f_a1[i], fx = f_a2[i];
        for (int j = 0; j < n_g; ++j)
            if (fx == g_id[j] && fa == g_a1[j]) { out_root[count] = root; out_alpha[count] = fa; ++count; }
    }
    return count;
}

struct PairHash {
    size_t operator()(const std::pair<int64_t, int64_t>& p) const {
        uint64_t h = (uint64_t)p.first * 0x9E3779B97F4A7C15ULL;
        h ^= (uint64_t)p.second + 0x9E3779B97F4A7C15ULL + (h << 6) + (h >> 2);
        return (size_t)h;
    }
};

static int cpu_hash_join(const std::vector<int64_t>& f_id, const std::vector<int64_t>& f_a1,
                         const std::vector<int64_t>& f_a2, const std::vector<int64_t>& g_id,
                         const std::vector<int64_t>& g_a1, std::vector<int64_t>& out_root,
                         std::vector<int64_t>& out_alpha) {
    int n_f = (int)f_id.size(), n_g = (int)g_id.size();
    std::unordered_set<std::pair<int64_t, int64_t>, PairHash> index;
    index.reserve(n_g * 2);
    for (int j = 0; j < n_g; ++j) index.insert({g_id[j], g_a1[j]});
    int count = 0;
    for (int i = 0; i < n_f; ++i)
        if (index.count({f_a2[i], f_a1[i]})) { out_root[count] = f_id[i]; out_alpha[count] = f_a1[i]; ++count; }
    return count;
}

static int host_lb(const int64_t* a, int lo, int hi, int64_t v) {
    while (lo < hi) { int m = (lo + hi) >> 1; if (a[m] < v) lo = m + 1; else hi = m; }
    return lo;
}
static int host_ub(const int64_t* a, int lo, int hi, int64_t v) {
    while (lo < hi) { int m = (lo + hi) >> 1; if (a[m] <= v) lo = m + 1; else hi = m; }
    return lo;
}

// Sort R_f into trie order (α,x,root) and R_g into (α,x); fill column arrays.
static void trie_sort(const std::vector<int64_t>& f_id, const std::vector<int64_t>& f_a1,
                      const std::vector<int64_t>& f_a2, const std::vector<int64_t>& g_id,
                      const std::vector<int64_t>& g_a1, std::vector<int64_t>& fa,
                      std::vector<int64_t>& fx, std::vector<int64_t>& fr,
                      std::vector<int64_t>& ga, std::vector<int64_t>& gx) {
    int n = (int)f_id.size();
    std::vector<std::array<int64_t, 3>> tf(n);
    std::vector<std::array<int64_t, 2>> tg(n);
    for (int i = 0; i < n; ++i) { tf[i] = {f_a1[i], f_a2[i], f_id[i]}; tg[i] = {g_a1[i], g_id[i]}; }
    std::sort(tf.begin(), tf.end());
    std::sort(tg.begin(), tg.end());
    for (int i = 0; i < n; ++i) {
        fa[i] = tf[i][0]; fx[i] = tf[i][1]; fr[i] = tf[i][2];
        ga[i] = tg[i][0]; gx[i] = tg[i][1];
    }
}

// CPU Leapfrog Triejoin (sorts internally — its own end-to-end cost).
static int cpu_lftj(const std::vector<int64_t>& f_id, const std::vector<int64_t>& f_a1,
                    const std::vector<int64_t>& f_a2, const std::vector<int64_t>& g_id,
                    const std::vector<int64_t>& g_a1, std::vector<int64_t>& out_root,
                    std::vector<int64_t>& out_alpha) {
    int n = (int)f_id.size();
    std::vector<int64_t> fa(n), fx(n), fr(n), ga(n), gx(n);
    trie_sort(f_id, f_a1, f_a2, g_id, g_a1, fa, fx, fr, ga, gx);
    int count = 0, i = 0, j = 0;
    while (i < n && j < n) {
        int64_t av = fa[i], ag = ga[j];
        if (av < ag) { i = host_lb(fa.data(), i + 1, n, ag); continue; }
        if (ag < av) { j = host_lb(ga.data(), j + 1, n, av); continue; }
        int64_t alpha = av;
        int faHi = host_ub(fa.data(), i, n, alpha), gaHi = host_ub(ga.data(), j, n, alpha);
        int p = i, q = j;
        while (p < faHi && q < gaHi) {
            int64_t xf = fx[p], xg = gx[q];
            if (xf < xg) { p = host_lb(fx.data(), p + 1, faHi, xg); continue; }
            if (xg < xf) { q = host_lb(gx.data(), q + 1, gaHi, xf); continue; }
            int64_t xv = xf;
            int fxHi = host_ub(fx.data(), p, faHi, xv);
            for (int r = p; r < fxHi; ++r)
                if (r == p || fr[r - 1] != fr[r]) { out_root[count] = fr[r]; out_alpha[count] = alpha; ++count; }
            p = fxHi; q = host_ub(gx.data(), q, gaHi, xv);
        }
        i = faHi; j = host_ub(ga.data(), j, n, alpha);
    }
    return count;
}

// ---- timing + verification -------------------------------------------------

// Wall-clock, min of `reps`. GPU lambdas must end with a synchronizing copy.
template <class F>
static double time_wall(F fn, int reps) {
    double best = 1e300;
    for (int r = 0; r < reps; ++r) {
        auto t0 = std::chrono::steady_clock::now();
        fn();
        auto t1 = std::chrono::steady_clock::now();
        double s = std::chrono::duration<double>(t1 - t0).count();
        if (s < best) best = s;
    }
    return best;
}

static bool verify(const std::vector<int64_t>& root, const std::vector<int64_t>& alpha,
                   int count, int n, const char* who) {
    if (count != n) { fprintf(stderr, "VERIFY FAIL (%s): count=%d expected=%d\n", who, count, n); return false; }
    std::vector<char> seen(n + 1, 0);
    for (int i = 0; i < count; ++i) {
        int64_t a = alpha[i];
        if (root[i] != I_F || a < 1 || a > n || seen[a]) {
            fprintf(stderr, "VERIFY FAIL (%s): bad/dup (%lld,%lld)\n", who, (long long)root[i], (long long)a);
            return false;
        }
        seen[a] = 1;
    }
    return true;
}

// Own device buffers for one GPU variant (alloc here = NOT timed; upload IS timed).
struct GpuBuf {
    int64_t *f_id, *f_a1, *f_a2, *g_id, *g_a1, *out_root, *out_alpha;
    int* count;
    int n;
    explicit GpuBuf(int n_) : n(n_) {
        size_t fb = (size_t)n * sizeof(int64_t);
        CUDA_CHECK(cudaMalloc(&f_id, fb));  CUDA_CHECK(cudaMalloc(&f_a1, fb));
        CUDA_CHECK(cudaMalloc(&f_a2, fb));  CUDA_CHECK(cudaMalloc(&g_id, fb));
        CUDA_CHECK(cudaMalloc(&g_a1, fb));  CUDA_CHECK(cudaMalloc(&out_root, fb));
        CUDA_CHECK(cudaMalloc(&out_alpha, fb)); CUDA_CHECK(cudaMalloc(&count, sizeof(int)));
    }
    ~GpuBuf() {
        cudaFree(f_id); cudaFree(f_a1); cudaFree(f_a2); cudaFree(g_id); cudaFree(g_a1);
        cudaFree(out_root); cudaFree(out_alpha); cudaFree(count);
    }
};

// ---- one problem size ------------------------------------------------------

struct Result {
    int n, matches;
    double cpu_bt, cpu_hj, cpu_lftj;     // -1 = skipped
    double gpu_nested, gpu_hj, gpu_lftj; // -1 = skipped
};

static Result run_one_n(int n, int nested_cap, std::mt19937& rng) {
    const int TPB = 256;
    auto blocks = [&](int m) { return (m + TPB - 1) / TPB; };
    size_t fb = (size_t)n * sizeof(int64_t);

    // Raw relations in RANDOM order (no method may free-ride on sortedness).
    std::vector<int64_t> f_id(n), f_a1(n), f_a2(n), g_id(n), g_a1(n);
    for (int k = 1; k <= n; ++k) { f_id[k-1]=I_F; f_a1[k-1]=k; f_a2[k-1]=I_G; g_id[k-1]=I_G; g_a1[k-1]=k; }
    std::vector<int> pf(n), pg(n);
    for (int i = 0; i < n; ++i) pf[i] = pg[i] = i;
    std::shuffle(pf.begin(), pf.end(), rng);
    std::shuffle(pg.begin(), pg.end(), rng);
    { std::vector<int64_t> a(n),b(n),c(n),d(n),e(n);
      for (int i=0;i<n;++i){ a[i]=f_id[pf[i]]; b[i]=f_a1[pf[i]]; c[i]=f_a2[pf[i]]; d[i]=g_id[pg[i]]; e[i]=g_a1[pg[i]]; }
      f_id.swap(a); f_a1.swap(b); f_a2.swap(c); g_id.swap(d); g_a1.swap(e); }

    std::vector<int64_t> h_root(n), h_alpha(n);
    int cnt = 0, h_count = 0;
    Result r; r.n = n;

    // ---- CPU: each builds its own index inside its own timed region --------
    r.cpu_hj = time_wall([&]{ cnt = cpu_hash_join(f_id,f_a1,f_a2,g_id,g_a1,h_root,h_alpha); }, CPU_REPS);
    if (!verify(h_root,h_alpha,cnt,n,"cpu hash join")) exit(2);
    r.matches = cnt;

    r.cpu_lftj = time_wall([&]{ cnt = cpu_lftj(f_id,f_a1,f_a2,g_id,g_a1,h_root,h_alpha); }, CPU_REPS);
    if (!verify(h_root,h_alpha,cnt,n,"cpu lftj")) exit(2);

    if (n <= nested_cap) {
        r.cpu_bt = time_wall([&]{ cnt = cpu_backtracking(f_id,f_a1,f_a2,g_id,g_a1,h_root,h_alpha); }, CPU_BT_REPS);
        if (!verify(h_root,h_alpha,cnt,n,"cpu backtracking")) exit(2);
    } else r.cpu_bt = -1.0;

    // ---- GPU hash join: own buffers, own H2D + build + probe + D2H ---------
    {
        GpuBuf b(n);
        uint64_t cap = 1; while (cap < (uint64_t)2*(uint64_t)n) cap <<= 1;
        int64_t *kx, *ka; CUDA_CHECK(cudaMalloc(&kx, cap*sizeof(int64_t))); CUDA_CHECK(cudaMalloc(&ka, cap*sizeof(int64_t)));
        r.gpu_hj = time_wall([&]{
            CUDA_CHECK(cudaMemcpy(b.f_id,f_id.data(),fb,cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(b.f_a1,f_a1.data(),fb,cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(b.f_a2,f_a2.data(),fb,cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(b.g_id,g_id.data(),fb,cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(b.g_a1,g_a1.data(),fb,cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemset(kx, 0xFF, cap*sizeof(int64_t)));
            CUDA_CHECK(cudaMemset(b.count, 0, sizeof(int)));
            gpu_build<<<blocks(n),TPB>>>(b.g_id,b.g_a1,n,kx,ka,cap);
            gpu_probe<<<blocks(n),TPB>>>(b.f_id,b.f_a1,b.f_a2,n,kx,ka,cap,b.out_root,b.out_alpha,b.count);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(&h_count,b.count,sizeof(int),cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_root.data(),b.out_root,(size_t)h_count*sizeof(int64_t),cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_alpha.data(),b.out_alpha,(size_t)h_count*sizeof(int64_t),cudaMemcpyDeviceToHost));
        }, GPU_REPS);
        cudaFree(kx); cudaFree(ka);
        if (!verify(h_root,h_alpha,h_count,n,"gpu hash join")) exit(2);
    }

    // ---- GPU LFTJ: own host sort + own H2D + kernel + D2H ------------------
    {
        GpuBuf b(n);
        std::vector<int64_t> fa(n), fx(n), fr(n), ga(n), gx(n);
        r.gpu_lftj = time_wall([&]{
            trie_sort(f_id,f_a1,f_a2,g_id,g_a1, fa,fx,fr,ga,gx);     // own sort (host)
            CUDA_CHECK(cudaMemcpy(b.f_a1,fa.data(),fb,cudaMemcpyHostToDevice)); // fa=α
            CUDA_CHECK(cudaMemcpy(b.f_a2,fx.data(),fb,cudaMemcpyHostToDevice)); // fx=x
            CUDA_CHECK(cudaMemcpy(b.f_id,fr.data(),fb,cudaMemcpyHostToDevice)); // fr=root
            CUDA_CHECK(cudaMemcpy(b.g_a1,ga.data(),fb,cudaMemcpyHostToDevice)); // ga=α
            CUDA_CHECK(cudaMemcpy(b.g_id,gx.data(),fb,cudaMemcpyHostToDevice)); // gx=x
            CUDA_CHECK(cudaMemset(b.count, 0, sizeof(int)));
            gpu_lftj<<<blocks(n),TPB>>>(b.f_a1,b.f_a2,b.f_id,n,b.g_a1,b.g_id,n,b.out_root,b.out_alpha,b.count);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(&h_count,b.count,sizeof(int),cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_root.data(),b.out_root,(size_t)h_count*sizeof(int64_t),cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_alpha.data(),b.out_alpha,(size_t)h_count*sizeof(int64_t),cudaMemcpyDeviceToHost));
        }, GPU_REPS);
        if (!verify(h_root,h_alpha,h_count,n,"gpu lftj")) exit(2);
    }

    // ---- GPU nested loop (capped): own H2D + kernel + D2H ------------------
    if (n <= nested_cap) {
        GpuBuf b(n);
        r.gpu_nested = time_wall([&]{
            CUDA_CHECK(cudaMemcpy(b.f_id,f_id.data(),fb,cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(b.f_a1,f_a1.data(),fb,cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(b.f_a2,f_a2.data(),fb,cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(b.g_id,g_id.data(),fb,cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(b.g_a1,g_a1.data(),fb,cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemset(b.count, 0, sizeof(int)));
            gpu_nested_loop<<<blocks(n),TPB>>>(b.f_id,b.f_a1,b.f_a2,n,b.g_id,b.g_a1,n,b.out_root,b.out_alpha,b.count);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(&h_count,b.count,sizeof(int),cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_root.data(),b.out_root,(size_t)h_count*sizeof(int64_t),cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_alpha.data(),b.out_alpha,(size_t)h_count*sizeof(int64_t),cudaMemcpyDeviceToHost));
        }, GPU_REPS);
        if (!verify(h_root,h_alpha,h_count,n,"gpu nested loop")) exit(2);
    } else r.gpu_nested = -1.0;

    return r;
}

static std::string fmt(double s) {
    char b[24];
    if (s < 0) snprintf(b, sizeof(b), "%11s", "skip"); else snprintf(b, sizeof(b), "%11.6f", s);
    return std::string(b);
}
static std::string csvf(double s) { if (s < 0) return std::string(); char b[24]; snprintf(b, sizeof(b), "%.9f", s); return std::string(b); }

int main(int argc, char** argv) {
    int max_n = (argc > 1) ? atoi(argv[1]) : (1 << 21);
    int nested_cap = (argc > 2) ? atoi(argv[2]) : (1 << 15);

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s (cc %d.%d, %d SMs)\n", prop.name, prop.major, prop.minor, prop.multiProcessorCount);
    printf("sweep N=1024..%d  (O(N^2) capped at %d)  |  END-TO-END per variant, no reuse, unsorted input\n\n", max_n, nested_cap);

    FILE* csv = fopen("results_gpu.csv", "w");
    if (!csv) { fprintf(stderr, "cannot open results_gpu.csv\n"); return 1; }
    fprintf(csv, "n,matches,cpu_backtracking_secs,cpu_hashjoin_secs,cpu_lftj_secs,gpu_nested_secs,gpu_hashjoin_secs,gpu_lftj_secs\n");

    printf("%8s %8s %11s %11s %11s %11s %11s %11s\n", "N","matches","cpu_bt","cpu_hj","cpu_lftj","gpu_nest","gpu_hj","gpu_lftj");
    printf("------------------------------------------------------------------------------------------------\n");

    std::mt19937 rng(20260616u); // fixed seed: reproducible shuffles
    for (int n = 1024; n <= max_n; n <<= 1) {
        Result r = run_one_n(n, nested_cap, rng);
        printf("%8d %8d %s %s %s %s %s %s\n", r.n, r.matches, fmt(r.cpu_bt).c_str(), fmt(r.cpu_hj).c_str(),
               fmt(r.cpu_lftj).c_str(), fmt(r.gpu_nested).c_str(), fmt(r.gpu_hj).c_str(), fmt(r.gpu_lftj).c_str());
        fprintf(csv, "%d,%d,%s,%.9f,%.9f,%s,%.9f,%.9f\n", r.n, r.matches, csvf(r.cpu_bt).c_str(), r.cpu_hj,
                r.cpu_lftj, csvf(r.gpu_nested).c_str(), r.gpu_hj, r.gpu_lftj);
        fflush(csv);
    }
    fclose(csv);
    printf("\nwrote results_gpu.csv  (all times END-TO-END wall-clock: each variant from raw\n");
    printf("unsorted arrays to result, building/sorting/uploading its own data — no reuse).\n");
    printf("correctness: every variant verified == {(I_F,k):k=1..N}; mismatch aborts.\n");
    return 0;
}
