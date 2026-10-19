// Small dense linear algebra for the CCA (internal header; templated on the scalar type so
// the host unit tests can exercise it in double regardless of the library precision).
//
//  * householder_qr / form_q: LAPACK dgeqr2 / dorg2r on an m x p matrix stored as p column
//    pointers (each column contiguous, length m). In place; no allocation.
//  * jacobi_svd: one-sided (Hestenes) Jacobi SVD of a small dense matrix. Returns singular
//    values sorted descending and the corresponding left singular vectors.
#pragma once

#include <array>
#include <cmath>
#include <limits>

namespace synapsys {
namespace detail {

template <class T>
inline T sign_of(T x) {
  return x < T(0) ? T(-1) : T(1);
}

// Householder QR in place (dgeqr2). On return the upper triangle of the first p rows holds
// R, the entries below the diagonal of column j hold the Householder vector v_j (with an
// implicit v_j[j] = 1), and tau[j] the scalar factor: H_j = I - tau_j v_j v_j^T,
// Q = H_0 H_1 ... H_{p-1}. A zero sub-column gives tau = 0 (H = I), so exactly zero
// (dead) columns are handled without division by zero.
template <class T>
void householder_qr(T* const* col, int m, int p, T* tau) {
  for (int j = 0; j < p; ++j) {
    T* a = col[j];
    T xnorm2 = 0;
    for (int i = j + 1; i < m; ++i) xnorm2 += a[i] * a[i];
    const T alpha = a[j];
    if (xnorm2 == T(0)) {
      tau[j] = 0;
      continue;
    }
    const T beta = -sign_of(alpha) * std::sqrt(alpha * alpha + xnorm2);
    tau[j] = (beta - alpha) / beta;
    const T scale = T(1) / (alpha - beta);
    for (int i = j + 1; i < m; ++i) a[i] *= scale;
    a[j] = beta;
    // Apply H_j to the trailing columns: c -= tau * v * (v^T c).
    for (int k = j + 1; k < p; ++k) {
      T* c = col[k];
      T w = c[j];
      for (int i = j + 1; i < m; ++i) w += a[i] * c[i];
      w *= tau[j];
      c[j] -= w;
      for (int i = j + 1; i < m; ++i) c[i] -= w * a[i];
    }
  }
}

// Overwrite the factored matrix with the first p columns of Q (dorg2r). Must be called
// after the caller has copied R out of the upper triangle.
template <class T>
void form_q(T* const* col, int m, int p, const T* tau) {
  for (int j = p - 1; j >= 0; --j) {
    T* a = col[j];
    if (j < p - 1) {
      // Apply H_j to columns j+1..p-1 (rows j..m-1), using v_j with v_j[j] = 1.
      a[j] = 1;
      for (int k = j + 1; k < p; ++k) {
        T* c = col[k];
        T w = 0;
        for (int i = j; i < m; ++i) w += a[i] * c[i];
        w *= tau[j];
        for (int i = j; i < m; ++i) c[i] -= w * a[i];
      }
    }
    for (int i = j + 1; i < m; ++i) a[i] *= -tau[j];
    a[j] = T(1) - tau[j];
    for (int i = 0; i < j; ++i) a[i] = 0;
  }
}

// Small row-major matrix with compile-time capacity N x N.
template <class T, int N>
using Small = std::array<std::array<T, N>, N>;

// One-sided Jacobi SVD of the m x c matrix a (rows < m, cols < c used; m, c <= N).
// On return: s[0..c-1] = singular values sorted descending (column norms of A V);
// u[:, k] = corresponding unit left singular vector (zero column if s[k] == 0).
// Returns the number of sweeps (max_sweeps if not converged; never NaN).
template <class T, int N>
int jacobi_svd(Small<T, N> a, int m, int c, std::array<T, N>* s, Small<T, N>* u,
               int max_sweeps = 60) {
  const T eps = std::numeric_limits<T>::epsilon();
  int sweep = 0;
  for (; sweep < max_sweeps; ++sweep) {
    bool rotated = false;
    for (int j = 0; j < c - 1; ++j) {
      for (int k = j + 1; k < c; ++k) {
        T alpha = 0, beta = 0, gamma = 0;
        for (int i = 0; i < m; ++i) {
          alpha += a[i][j] * a[i][j];
          beta += a[i][k] * a[i][k];
          gamma += a[i][j] * a[i][k];
        }
        if (gamma == T(0) || std::fabs(gamma) <= eps * std::sqrt(alpha * beta)) continue;
        rotated = true;
        const T zeta = (beta - alpha) / (T(2) * gamma);
        const T t = sign_of(zeta) / (std::fabs(zeta) + std::sqrt(T(1) + zeta * zeta));
        const T cs = T(1) / std::sqrt(T(1) + t * t);
        const T sn = cs * t;
        for (int i = 0; i < m; ++i) {
          const T x = a[i][j];
          const T y = a[i][k];
          a[i][j] = cs * x - sn * y;
          a[i][k] = sn * x + cs * y;
        }
      }
    }
    if (!rotated) break;
  }
  // Column norms -> singular values; normalise columns -> left singular vectors.
  std::array<T, N> norm{};
  std::array<int, N> order{};
  for (int k = 0; k < c; ++k) {
    T sum = 0;
    for (int i = 0; i < m; ++i) sum += a[i][k] * a[i][k];
    norm[k] = std::sqrt(sum);
    order[k] = k;
  }
  // Insertion sort, descending (stable: equal values keep column order).
  for (int k = 1; k < c; ++k) {
    const int v = order[k];
    int q = k - 1;
    while (q >= 0 && norm[order[q]] < norm[v]) {
      order[q + 1] = order[q];
      --q;
    }
    order[q + 1] = v;
  }
  for (int k = 0; k < c; ++k) {
    const int src = order[k];
    (*s)[k] = norm[src];
    const T inv = norm[src] > T(0) ? T(1) / norm[src] : T(0);
    for (int i = 0; i < m; ++i) (*u)[i][k] = a[i][src] * inv;
  }
  return sweep;
}

}  // namespace detail
}  // namespace synapsys
