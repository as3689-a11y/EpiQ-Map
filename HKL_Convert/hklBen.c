/*
 * hklBen.c -- HKL transformation and 3D histogramming for RSM reconstruction.
 *
 * Functions (called from hklBen.py via ctypes):
 *
 *   hklhist  -- NEW fused kernel: rotates detector q-vectors into HKL and
 *               histograms them in a single parallel pass. This is the fast
 *               path used by autoRSM.py. Per-pixel work stays in registers;
 *               no intermediate HR/KR/LR arrays are written.
 *
 *   hist2    -- 3D histogram with binary-search bin lookup (legacy path,
 *               kept for compatibility with existing notebooks).
 *   hist     -- 3D histogram with linear-search bin lookup (legacy).
 *   histarb  -- 2D histogram on an arbitrary plane (free slicing).
 *   benhkl   -- q -> HKL rotation only (legacy).
 *   calchkl  -- HKL from polar/azimuthal detector angles (legacy).
 *
 * Build:  make            (gcc -O3 -march=native -fopenmp -shared -fPIC)
 * Thread count is controlled with the OMP_NUM_THREADS environment variable.
 */

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <unistd.h>
#include <sys/time.h>
#include <stdarg.h>
#include <omp.h>

/* ---------------- small linear-algebra helpers ---------------- */

double dotf        (double *a, double *b, int size);
void   cross3f     (double *a, double *b, double *v);
void   trans3f     (double A[3][3], double M[3][3]);
void   MatrixMultf (double A[3][3], double B[3][3], double C[3][3]);
void   MVMult3f    (double *vin, double B[3][3], double *vout);
double determ3f    (double A[3][3]);
void   inverse3f   (double A[3][3], double invA[3][3]);
double getclock    (void);

/* ---------------- public API ---------------- */

void hklhist(double *q, double *M, float *counts,
             double *Hbin, double *Kbin, double *Lbin,
             float *vol, float *norm,
             int N, int hn, int kn, int ln, float mon);

void calchkl(double *P, double *A, double eta, double mu, double chi,
             double phi, double WL, double *U,
             double *HR, double *KR, double *LR, int N);

void benhkl(double *qmag, double *tth, double *gam,
            double *thetaMat, double *chiMat, double *phiMat,
            double WL, double *UB_inv_T,
            double *HR, double *KR, double *LR, int N);

void hist(double *Hbin, double *Kbin, double *Lbin,
          double *HR, double *KR, double *LR,
          float *counts, float *vol, float *norm, float *errors,
          int N, int hn, int kn, int ln, float mon);

void hist2(double *Hbin, double *Kbin, double *Lbin,
           double *HR, double *KR, double *LR,
           float *counts, float *vol, float *norm, float *errors,
           int N, int hn, int kn, int ln, float mon);

void histarb(double *Q1bin, double *Q2bin, float *phat,
             double *HR, double *KR, double *LR,
             float *vec1, float *vec2, float prangemin, float prangemax,
             float *counts, float *vol, float *norm,
             int N, int lenQ1, int lenQ2, float mon);

/* ---------------------------------------------------------------
 * Binary search: index of the bin containing x in a sorted grid.
 * Returns i such that bin[i] <= x < bin[i+1].
 * Caller must guarantee bin[0] < x < bin[len-1].
 * --------------------------------------------------------------- */
static inline int bin_index(const double *bin, int len, double x)
{
    int left = 0, right = len - 1;
    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (bin[mid] > x) right = mid - 1;
        else              left = mid + 1;
    }
    return left - 1;
}

/* ---------------------------------------------------------------
 * hklhist -- fused rotation + histogram (fast path).
 *
 *   q       : detector q-vectors, contiguous layout (3, N):
 *             q[0..N-1] = qx, q[N..2N-1] = qy, q[2N..3N-1] = qz.
 *             Frame-independent; compute once per dataset.
 *   M       : row-major 3x3 rotation, hkl = M . q
 *             (in Python: Uinv.T @ PHI @ CHI @ ETA).
 *   counts  : per-pixel intensities (float32). Negative = masked, skipped.
 *   H/K/Lbin: sorted bin-edge grids.
 *   vol     : accumulated intensity, length hn*kn*ln.
 *   norm    : accumulated hit count, length hn*kn*ln.
 *   mon     : monitor normalization (counts divided by this).
 * --------------------------------------------------------------- */
void hklhist(double *q, double *M, float *counts,
             double *Hbin, double *Kbin, double *Lbin,
             float *vol, float *norm,
             int N, int hn, int kn, int ln, float mon)
{
    const double m00 = M[0], m01 = M[1], m02 = M[2];
    const double m10 = M[3], m11 = M[4], m12 = M[5];
    const double m20 = M[6], m21 = M[7], m22 = M[8];
    const double *qx = q;
    const double *qy = q + (long)N;
    const double *qz = q + 2L * (long)N;
    const double hlo = Hbin[0], hhi = Hbin[hn - 1];
    const double klo = Kbin[0], khi = Kbin[kn - 1];
    const double llo = Lbin[0], lhi = Lbin[ln - 1];
    int n;

    #pragma omp parallel for schedule(static)
    for (n = 0; n < N; n++) {
        if (counts[n] < 0.0f) continue;           /* masked pixel */

        const double x = qx[n], y = qy[n], z = qz[n];

        const double h = m00 * x + m01 * y + m02 * z;
        if (h <= hlo || h >= hhi) continue;
        const double k = m10 * x + m11 * y + m12 * z;
        if (k <= klo || k >= khi) continue;
        const double l = m20 * x + m21 * y + m22 * z;
        if (l <= llo || l >= lhi) continue;

        const int hin = bin_index(Hbin, hn, h);
        const int kin = bin_index(Kbin, kn, k);
        const int lin = bin_index(Lbin, ln, l);

        const long index = (long)hin * kn * ln + (long)kin * ln + lin;

        #pragma omp atomic
        vol[index] += counts[n] / mon;
        #pragma omp atomic
        norm[index] += 1.0f;
    }
}

/* ---------------------------------------------------------------
 * Legacy functions below, unchanged in behavior.
 * (hist/hist2 no longer take a useless errors[index]+=0 atomic;
 *  the errors array is still accepted for API compatibility.)
 * --------------------------------------------------------------- */

void histarb(double *Q1bin, double *Q2bin, float *phat,
             double *HR, double *KR, double *LR,
             float *vec1, float *vec2, float prangemin, float prangemax,
             float *counts, float *vol, float *norm,
             int N, int lenQ1, int lenQ2, float mon)
{
    int i, n, Q1index, Q2index;
    double v1conv, v2conv, pconv;

    #pragma omp parallel for private(i, n, Q1index, Q2index, v1conv, v2conv, pconv)
    for (n = 0; n < N; n++) {
        if (counts[n] >= 0.0f) {
            Q1index = 0;
            Q2index = 0;
            pconv = HR[n] * phat[0] + KR[n] * phat[1] + LR[n] * phat[2];
            if (pconv > prangemin && pconv < prangemax) {
                v1conv = HR[n] * vec1[0] + KR[n] * vec1[1] + LR[n] * vec1[2];
                if (v1conv > Q1bin[0] && v1conv < Q1bin[lenQ1 - 1]) {
                    v2conv = HR[n] * vec2[0] + KR[n] * vec2[1] + LR[n] * vec2[2];
                    if (v2conv > Q2bin[0] && v2conv < Q2bin[lenQ2 - 1]) {
                        for (i = 1; i < lenQ1; i++) {
                            if (Q1bin[i] > v1conv) { Q1index = i - 1; break; }
                        }
                        for (i = 1; i < lenQ2; i++) {
                            if (Q2bin[i] > v2conv) { Q2index = i - 1; break; }
                        }
                        #pragma omp atomic
                        vol[Q1index * lenQ2 + Q2index] += counts[n] / mon;
                        #pragma omp atomic
                        norm[Q1index * lenQ2 + Q2index] += 1.0f;
                    }
                }
            }
        }
    }
}

void hist(double *Hbin, double *Kbin, double *Lbin,
          double *HR, double *KR, double *LR,
          float *counts, float *vol, float *norm, float *errors,
          int N, int hn, int kn, int ln, float mon)
{
    (void)errors;
    int i, hin, kin, lin, n, flag, flag2;
    long index;

    #pragma omp parallel for private(n, i, flag, flag2, hin, kin, lin, index)
    for (n = 0; n < N; n++) {
        if (counts[n] >= 0.0f) {
            hin = kin = lin = 0;
            flag2 = 0;

            if (HR[n] > Hbin[0] && HR[n] < Hbin[hn - 1]) {
                flag = 0;
                for (i = 1; i < hn; i++) {
                    if (flag == 0 && Hbin[i] > HR[n]) { flag = 1; hin = i - 1; break; }
                }
            } else flag2 = 1;

            if (KR[n] > Kbin[0] && KR[n] < Kbin[kn - 1] && flag2 == 0) {
                flag = 0;
                for (i = 1; i < kn; i++) {
                    if (flag == 0 && Kbin[i] > KR[n]) { flag = 1; kin = i - 1; break; }
                }
            } else flag2 = 1;

            if (LR[n] > Lbin[0] && LR[n] < Lbin[ln - 1] && flag2 == 0) {
                flag = 0;
                for (i = 1; i < ln; i++) {
                    if (flag == 0 && Lbin[i] > LR[n]) { flag = 1; lin = i - 1; break; }
                }
            } else flag2 = 1;

            if (flag2 == 0) {
                index = (long)hin * kn * ln + (long)kin * ln + lin;
                #pragma omp atomic
                vol[index] += counts[n] / mon;
                #pragma omp atomic
                norm[index] += 1.0f;
            }
        }
    }
}

void hist2(double *Hbin, double *Kbin, double *Lbin,
           double *HR, double *KR, double *LR,
           float *counts, float *vol, float *norm, float *errors,
           int N, int hn, int kn, int ln, float mon)
{
    (void)errors;
    int hin, kin, lin, n, flag2;
    long index;

    #pragma omp parallel for private(n, flag2, hin, kin, lin, index)
    for (n = 0; n < N; n++) {
        if (counts[n] >= 0.0f) {
            hin = kin = lin = 0;
            flag2 = 0;

            if (HR[n] > Hbin[0] && HR[n] < Hbin[hn - 1])
                hin = bin_index(Hbin, hn, HR[n]);
            else flag2 = 1;

            if (flag2 == 0 && KR[n] > Kbin[0] && KR[n] < Kbin[kn - 1])
                kin = bin_index(Kbin, kn, KR[n]);
            else flag2 = 1;

            if (flag2 == 0 && LR[n] > Lbin[0] && LR[n] < Lbin[ln - 1])
                lin = bin_index(Lbin, ln, LR[n]);
            else flag2 = 1;

            if (flag2 == 0) {
                index = (long)hin * kn * ln + (long)kin * ln + lin;
                #pragma omp atomic
                vol[index] += counts[n] / mon;
                #pragma omp atomic
                norm[index] += 1.0f;
            }
        }
    }
}

void benhkl(double *qmag, double *tth, double *gam,
            double *thetaMat, double *chiMat, double *phiMat,
            double WL, double *UB_inv_T,
            double *HR, double *KR, double *LR, int N)
{
    (void)WL;
    int i;
    const double pi = 3.141592653589793;
    double UBinvT[3][3], PHI[3][3], CHI[3][3], ETA[3][3];
    double q[3], hklvector[3];
    double T1[3][3], T2[3][3], M[3][3];

    for (i = 0; i < 9; i++) {
        UBinvT[i / 3][i % 3] = UB_inv_T[i];
        PHI[i / 3][i % 3]    = phiMat[i];
        CHI[i / 3][i % 3]    = chiMat[i];
        ETA[i / 3][i % 3]    = thetaMat[i];
    }

    MatrixMultf(UBinvT, PHI, T1);
    MatrixMultf(T1, CHI, T2);
    MatrixMultf(T2, ETA, M);

    #pragma omp parallel for private(i, q, hklvector)
    for (i = 0; i < N; i++) {
        q[2] = -qmag[i] * sin(pi / 2 - tth[i] / 2) * cos(-gam[i]);
        q[1] = -qmag[i] * cos(pi / 2 - tth[i] / 2);
        q[0] =  qmag[i] * sin(pi / 2 - tth[i] / 2) * sin(-gam[i]);

        hklvector[0] = dotf(q, M[0], 3);
        hklvector[1] = dotf(q, M[1], 3);
        hklvector[2] = dotf(q, M[2], 3);

        HR[i] = hklvector[0];
        KR[i] = hklvector[1];
        LR[i] = hklvector[2];
    }
}

void calchkl(double *P, double *A, double eta, double mu, double chi,
             double phi, double WL, double *U,
             double *HR, double *KR, double *LR, int N)
{
    int i;
    double Umatrix[3][3], Uinv[3][3];
    double OMEGA[3][3], PHI[3][3], CHI[3][3], ETA[3][3], MU[3][3];
    double OMEGAINV[3][3], PHIINV[3][3], CHIINV[3][3], ETAINV[3][3], MUINV[3][3];
    double M[3][3], T1[3][3], T2[3][3], T3[3][3], T4[3][3];
    double omega, q[3], hklvector[3], ki;

    (void)ETA; (void)ETAINV;

    for (i = 0; i < 9; i++) Umatrix[i / 3][i % 3] = U[i];

    PHI[0][0] = cos(phi);  PHI[0][1] = -sin(phi); PHI[0][2] = 0.0;
    PHI[1][0] = sin(phi);  PHI[1][1] =  cos(phi); PHI[1][2] = 0.0;
    PHI[2][0] = 0.0;       PHI[2][1] = 0.0;       PHI[2][2] = 1.0;

    CHI[0][0] = cos(chi);  CHI[0][1] = 0.0; CHI[0][2] = -sin(chi);
    CHI[1][0] = 0.0;       CHI[1][1] = 1.0; CHI[1][2] = 0.0;
    CHI[2][0] = sin(chi);  CHI[2][1] = 0.0; CHI[2][2] =  cos(chi);

    ETA[0][0] = cos(eta);  ETA[0][1] = -sin(eta); ETA[0][2] = 0.0;
    ETA[1][0] = sin(eta);  ETA[1][1] =  cos(eta); ETA[1][2] = 0.0;
    ETA[2][0] = 0.0;       ETA[2][1] = 0.0;       ETA[2][2] = 1.0;

    MU[0][0] = 1.0; MU[0][1] = 0.0;     MU[0][2] = 0.0;
    MU[1][0] = 0.0; MU[1][1] = cos(mu); MU[1][2] = -sin(mu);
    MU[2][0] = 0.0; MU[2][1] = sin(mu); MU[2][2] =  cos(mu);

    inverse3f(Umatrix, Uinv);
    inverse3f(CHI, CHIINV);
    inverse3f(PHI, PHIINV);
    inverse3f(ETA, ETAINV);
    inverse3f(MU, MUINV);

    ki = 1.0 / WL;

    MatrixMultf(MUINV, ETAINV, T4);
    MatrixMultf(T4, CHIINV, T3);
    MatrixMultf(T3, PHIINV, T2);
    MatrixMultf(T2, Uinv, T1);

    #pragma omp parallel for private(i, q, omega, OMEGA, OMEGAINV, T2, M, hklvector)
    for (i = 0; i < N; i++) {
        q[0] = ki * sqrt(1.0 - 2 * cos(A[i]) * cos(P[i]) + cos(A[i]) * cos(A[i]));
        q[1] = 0.0;
        q[2] = sin(A[i]) * ki;
        if (P[i] < 0.0) q[0] = -q[0];

        omega = -atan((1.0 - cos(A[i]) * cos(P[i])) / (cos(A[i]) * sin(P[i])));
        OMEGA[0][0] = cos(omega); OMEGA[0][1] = -sin(omega); OMEGA[0][2] = 0.0;
        OMEGA[1][0] = sin(omega); OMEGA[1][1] =  cos(omega); OMEGA[1][2] = 0.0;
        OMEGA[2][0] = 0.0;        OMEGA[2][1] = 0.0;         OMEGA[2][2] = 1.0;

        inverse3f(OMEGA, OMEGAINV);
        MatrixMultf(OMEGAINV, T1, T2);
        trans3f(T2, M);

        hklvector[0] = dotf(q, M[0], 3);
        hklvector[1] = dotf(q, M[1], 3);
        hklvector[2] = dotf(q, M[2], 3);

        HR[i] = hklvector[0];
        KR[i] = hklvector[1];
        LR[i] = hklvector[2];
    }
}

/* ---------------- helpers ---------------- */

double dotf(double *a, double *b, int size)
{
    int i;
    double sum = 0.0;
    for (i = 0; i < size; i++) sum += a[i] * b[i];
    return sum;
}

void cross3f(double *a, double *b, double *v)
{
    v[0] = a[1] * b[2] - a[2] * b[1];
    v[1] = a[2] * b[0] - a[0] * b[2];
    v[2] = a[0] * b[1] - a[1] * b[0];
}

void trans3f(double A[3][3], double M[3][3])
{
    int i, j;
    for (i = 0; i < 3; i++)
        for (j = 0; j < 3; j++)
            M[i][j] = A[j][i];
}

void MatrixMultf(double A[3][3], double B[3][3], double C[3][3])
{
    int i, j;
    double M[3][3];
    trans3f(B, M);
    for (i = 0; i < 3; i++)
        for (j = 0; j < 3; j++)
            C[i][j] = dotf(A[i], M[j], 3);
}

void MVMult3f(double *vin, double B[3][3], double *vout)
{
    int i;
    double M[3][3];
    trans3f(B, M);
    for (i = 0; i < 3; i++) vout[i] = dotf(vin, M[i], 3);
}

double determ3f(double A[3][3])
{
    double retval = 0.0;
    retval += A[0][0] * A[1][1] * A[2][2];
    retval += A[0][1] * A[1][2] * A[2][0];
    retval += A[0][2] * A[1][0] * A[2][1];
    retval -= A[2][0] * A[1][1] * A[0][2];
    retval -= A[2][1] * A[1][2] * A[0][0];
    retval -= A[2][2] * A[1][0] * A[0][1];
    return retval;
}

void inverse3f(double A[3][3], double invA[3][3])
{
    double detA = determ3f(A);
    double M[3][3], r[3][3];
    int i, j;

    trans3f(A, M);
    cross3f(M[1], M[2], r[0]);
    cross3f(M[2], M[0], r[1]);
    cross3f(M[0], M[1], r[2]);

    for (i = 0; i < 3; i++)
        for (j = 0; j < 3; j++)
            invA[i][j] = r[i][j] / detA;
}

double getclock(void)
{
    struct timezone tzp;
    struct timeval tp;
    gettimeofday(&tp, &tzp);
    return tp.tv_sec + tp.tv_usec * 1.0e-6;
}
