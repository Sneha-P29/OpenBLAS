#include "common.h"
#include <altivec.h>

typedef __vector unsigned char  vec_t;
typedef FLOAT v4sf_t __attribute__ ((vector_size (16)));
typedef FLOAT v2sf_t __attribute__ ((vector_size (8)));

/*
 * FP16 xvf16ger2pp needs 4×2 FP16 input — same MERGE_HIGH/MERGE_LOW as sbgemm.
 */
#define MERGE_HIGH(x, y) (vec_t) vec_mergeh ((vector short)x, (vector short)y)
#define MERGE_LOW(x, y)  (vec_t) vec_mergel ((vector short)x, (vector short)y)

/* FP16 MMA outer-product accumulate */
#define MMA __builtin_mma_xvf16ger2pp

/* Decode a stored hfloat16 (uint16_t) to float using IEEE 754 FP16.
 * Used in scalar residue paths where MMA is not invoked.                */
static inline float hf16_to_f32(hfloat16 h)
{
    unsigned int x = (unsigned int)h;
    unsigned int s = (x >> 15) & 1u;
    unsigned int e = (x >> 10) & 0x1fu;
    unsigned int f =  x        & 0x3ffu;
    float v;
    if (e == 0)       v = ldexpf((float)f, -24);
    else if (e == 31) v = (f == 0) ? (float)(1.0/0.0) : (float)(0.0/0.0);
    else              v = ldexpf((float)(1024u + f), (int)e - 25);
    return s ? -v : v;
}
#define HF(x) hf16_to_f32(x)

/* -----------------------------------------------------------------------
 * SAVE_ACC macros: TRMMKERNEL sets C, GEMM accumulates into C.
 * ----------------------------------------------------------------------- */
#if defined(TRMMKERNEL)
#define SAVE_ACC(ACC, J)  \
	  __builtin_mma_disassemble_acc ((void *)result, ACC); \
	  rowC = (v4sf_t *) &CO[0* ldc+J]; \
          rowC[0] = result[0] * alpha; \
          rowC = (v4sf_t *) &CO[1*ldc+J]; \
          rowC[0] = result[1] * alpha; \
          rowC = (v4sf_t *) &CO[2*ldc+J]; \
          rowC[0] = result[2] * alpha; \
          rowC = (v4sf_t *) &CO[3*ldc+J]; \
          rowC[0] = result[3] * alpha;
#define SAVE_ACC1(ACC, J)  \
	  __builtin_mma_disassemble_acc ((void *)result, ACC); \
	  rowC = (v4sf_t *) &CO[4* ldc+J]; \
          rowC[0] = result[0] * alpha; \
          rowC = (v4sf_t *) &CO[5*ldc+J]; \
          rowC[0] = result[1] * alpha; \
          rowC = (v4sf_t *) &CO[6*ldc+J]; \
          rowC[0] = result[2] * alpha; \
          rowC = (v4sf_t *) &CO[7*ldc+J]; \
          rowC[0] = result[3] * alpha;
#define  SAVE4x2_ACC(ACC, J)  \
	  __builtin_mma_disassemble_acc ((void *)result, ACC); \
	  rowC = (v2sf_t *) &CO[0* ldc+J]; \
          rowC[0] = result[0] * alpha; \
	  rowC = (v2sf_t *) &CO[1* ldc+J]; \
          rowC[0] = result[2] * alpha; \
	  rowC = (v2sf_t *) &CO[2* ldc+J]; \
          rowC[0] = result[4] * alpha; \
	  rowC = (v2sf_t *) &CO[3* ldc+J]; \
          rowC[0] = result[6] * alpha;
#define  SAVE4x2_ACC1(ACC, J)  \
	  __builtin_mma_disassemble_acc ((void *)result, ACC); \
	  rowC = (v2sf_t *) &CO[4* ldc+J]; \
          rowC[0] = result[0] * alpha; \
	  rowC = (v2sf_t *) &CO[5* ldc+J]; \
          rowC[0] = result[2] * alpha; \
	  rowC = (v2sf_t *) &CO[6* ldc+J]; \
          rowC[0] = result[4] * alpha; \
	  rowC = (v2sf_t *) &CO[7* ldc+J]; \
          rowC[0] = result[6] * alpha;
#define  SAVE4x2_ACC_SCALAR(ACC) {                             \
           __builtin_mma_disassemble_acc ((void *)result, ACC); \
           res[0] = result[0] * alpha;                          \
           res[1] = result[1] * alpha;                          \
           res[2] = result[2] * alpha;                          \
           res[3] = result[3] * alpha;                          \
           CO[0 * ldc] = res[0][0];                             \
           CO[1 * ldc] = res[1][0];                             \
           CO[2 * ldc] = res[2][0];                             \
           CO[3 * ldc] = res[3][0];                             \
 }
#define  SAVE4x2_ACC1_SCALAR(ACC) {                            \
           __builtin_mma_disassemble_acc ((void *)result, ACC); \
           res[0] = result[0] * alpha;                          \
           res[1] = result[1] * alpha;                          \
           res[2] = result[2] * alpha;                          \
           res[3] = result[3] * alpha;                          \
           CO[4 * ldc] = res[0][0];                             \
           CO[5 * ldc] = res[1][0];                             \
           CO[6 * ldc] = res[2][0];                             \
           CO[7 * ldc] = res[3][0];                             \
}
#define  SAVE2x4_ACC(ACC, J)  \
	  __builtin_mma_disassemble_acc ((void *)result, ACC); \
	  rowC = (v4sf_t *) &CO[0* ldc+J]; \
          rowC[0] = result[0] * alpha; \
	  rowC = (v4sf_t *) &CO[1* ldc+J]; \
          rowC[0] = result[1] * alpha;
#else
#define SAVE_ACC(ACC, J)  \
	  __builtin_mma_disassemble_acc ((void *)result, ACC); \
	  rowC = (v4sf_t *) &CO[0* ldc+J]; \
          rowC[0] += result[0] * alpha; \
          rowC = (v4sf_t *) &CO[1*ldc+J]; \
          rowC[0] += result[1] * alpha; \
          rowC = (v4sf_t *) &CO[2*ldc+J]; \
          rowC[0] += result[2] * alpha; \
          rowC = (v4sf_t *) &CO[3*ldc+J]; \
          rowC[0] += result[3] * alpha;
#define SAVE_ACC1(ACC, J)  \
	  __builtin_mma_disassemble_acc ((void *)result, ACC); \
	  rowC = (v4sf_t *) &CO[4* ldc+J]; \
          rowC[0] += result[0] * alpha; \
          rowC = (v4sf_t *) &CO[5*ldc+J]; \
          rowC[0] += result[1] * alpha; \
          rowC = (v4sf_t *) &CO[6*ldc+J]; \
          rowC[0] += result[2] * alpha; \
          rowC = (v4sf_t *) &CO[7*ldc+J]; \
          rowC[0] += result[3] * alpha;
#define  SAVE4x2_ACC(ACC, J)  \
	  __builtin_mma_disassemble_acc ((void *)result, ACC); \
	  rowC = (v2sf_t *) &CO[0* ldc+J]; \
          rowC[0] += result[0] * alpha; \
	  rowC = (v2sf_t *) &CO[1* ldc+J]; \
          rowC[0] += result[2] * alpha; \
	  rowC = (v2sf_t *) &CO[2* ldc+J]; \
          rowC[0] += result[4] * alpha; \
	  rowC = (v2sf_t *) &CO[3* ldc+J]; \
          rowC[0] += result[6] * alpha;
#define  SAVE4x2_ACC1(ACC, J)  \
	  __builtin_mma_disassemble_acc ((void *)result, ACC); \
	  rowC = (v2sf_t *) &CO[4* ldc+J]; \
          rowC[0] += result[0] * alpha; \
	  rowC = (v2sf_t *) &CO[5* ldc+J]; \
          rowC[0] += result[2] * alpha; \
	  rowC = (v2sf_t *) &CO[6* ldc+J]; \
          rowC[0] += result[4] * alpha; \
	  rowC = (v2sf_t *) &CO[7* ldc+J]; \
          rowC[0] += result[6] * alpha;
#define  SAVE4x2_ACC_SCALAR(ACC) {                             \
           __builtin_mma_disassemble_acc ((void *)result, ACC); \
           res[0] = result[0] * alpha;                          \
           res[1] = result[1] * alpha;                          \
           res[2] = result[2] * alpha;                          \
           res[3] = result[3] * alpha;                          \
           CO[0 * ldc] += res[0][0];                            \
           CO[1 * ldc] += res[1][0];                            \
           CO[2 * ldc] += res[2][0];                            \
           CO[3 * ldc] += res[3][0];                            \
 }
#define  SAVE4x2_ACC1_SCALAR(ACC) {                            \
           __builtin_mma_disassemble_acc ((void *)result, ACC); \
           res[0] = result[0] * alpha;                          \
           res[1] = result[1] * alpha;                          \
           res[2] = result[2] * alpha;                          \
           res[3] = result[3] * alpha;                          \
           CO[4 * ldc] += res[0][0];                            \
           CO[5 * ldc] += res[1][0];                            \
           CO[6 * ldc] += res[2][0];                            \
           CO[7 * ldc] += res[3][0];                            \
}
#define  SAVE2x4_ACC(ACC, J)  \
	  __builtin_mma_disassemble_acc ((void *)result, ACC); \
	  rowC = (v4sf_t *) &CO[0* ldc+J]; \
          rowC[0] += result[0] * alpha; \
	  rowC = (v4sf_t *) &CO[1* ldc+J]; \
          rowC[0] += result[1] * alpha;
#endif

#define SET_ACC_ZERO4() \
	  __builtin_mma_xxsetaccz (&acc0); \
	  __builtin_mma_xxsetaccz (&acc1); \
	  __builtin_mma_xxsetaccz (&acc2); \
	  __builtin_mma_xxsetaccz (&acc3);

#define SET_ACC_ZERO8() \
	  __builtin_mma_xxsetaccz (&acc0); \
	  __builtin_mma_xxsetaccz (&acc1); \
	  __builtin_mma_xxsetaccz (&acc2); \
	  __builtin_mma_xxsetaccz (&acc3); \
	  __builtin_mma_xxsetaccz (&acc4); \
	  __builtin_mma_xxsetaccz (&acc5); \
	  __builtin_mma_xxsetaccz (&acc6); \
	  __builtin_mma_xxsetaccz (&acc7);

#define PREFETCH1(x, y) asm volatile ("dcbt %0, %1" : : "b" (x), "r" (y) : "memory");

/* -----------------------------------------------------------------------
 * TRMM offset tracking macros — identical to sgemm_kernel_power10.c
 * ----------------------------------------------------------------------- */
#if (defined(LEFT) && !defined(TRANSA)) || (!defined(LEFT) && defined(TRANSA))
#define REFRESH_TEMP_BK(x, y) \
            temp = k - off;
#elif defined(LEFT)
#define REFRESH_TEMP_BK(x, y) \
            temp = off + x;
#else
#define REFRESH_TEMP_BK(x, y) \
            temp = off + y;
#endif
#if (defined(LEFT) && defined(TRANSA)) || (!defined(LEFT) && !defined(TRANSA))
#define REFRESH_POINTERS(x, y) \
	  BO = B; \
          REFRESH_TEMP_BK(x, y)
#else
#define REFRESH_POINTERS(x, y) \
          AO += off * x; \
          BO = B + off * y; \
          REFRESH_TEMP_BK(x, y)
#endif
#ifdef LEFT
#define REFRESH_OFF(x) \
            off += x;
#else
#define REFRESH_OFF(x)
#endif
#ifdef LEFT
#define UPDATE_TEMP(x, y) \
            temp -= x;
#else
#define UPDATE_TEMP(x, y) \
            temp -= y;
#endif
#if (defined(LEFT) && defined(TRANSA)) || (!defined(LEFT) && !defined(TRANSA))
#define REFRESH_TMP_AFTER_SAVE(x, y) \
            temp = k - off; \
            UPDATE_TEMP(x, y) \
            AO += temp * x; \
            BO += temp * y;
#else
#define REFRESH_TMP_AFTER_SAVE(x, y)
#endif
#define REFRESH_AFTER_SAVE(x,y) \
        REFRESH_TMP_AFTER_SAVE(x, y) \
	REFRESH_OFF(x)

/*************************************************************************************
* SHGEMM / SHTRMM Kernel  —  hfloat16 (FP16) × hfloat16 → float (FP32)
* Identical structure to sbgemm_kernel_power10.c; only MMA macro differs.
* Supports both GEMM (C += α·A·B) and TRMM (C = α·A·B) via TRMMKERNEL flag.
*************************************************************************************/
int
CNAME (BLASLONG m, BLASLONG n, BLASLONG k, FLOAT alpha, IFLOAT * A,
       IFLOAT * B, FLOAT * C, BLASLONG ldc
#ifdef TRMMKERNEL
       , BLASLONG offset
#endif
  )
{
  BLASLONG i1;
#if defined(TRMMKERNEL)
  BLASLONG off, temp;
#endif
#if defined(TRMMKERNEL) && !defined(LEFT)
  off = -offset;
#endif
  v4sf_t valpha = { alpha, alpha, alpha, alpha };
  vector short vzero = { 0, 0, 0, 0, 0, 0, 0, 0 };

  /* Loop for n >= 8. */
  for (i1 = 0; i1 < (n >> 3); i1++)
    {
      BLASLONG j, temp;
      FLOAT *CO;
      IFLOAT *AO;
#if defined(TRMMKERNEL) && defined(LEFT)
      off = offset;
#endif
      CO = C;
      C += ldc << 3;
      AO = A;
      PREFETCH1 (A, 128);
      PREFETCH1 (A, 256);
      /* Loop for m >= 16. */
      for (j = 0; j < (m >> 4); j++)
 {
   IFLOAT *BO;
#if defined(TRMMKERNEL)
   REFRESH_POINTERS (16, 8);
#else
   BO = B;
   temp = k;
#endif
   v4sf_t *rowC;
   v4sf_t result[4];
   __vector_quad acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7;
   SET_ACC_ZERO8 ();
   BLASLONG l = 0;
	  for (l = 0; l < temp / 2; l++)
	    {
	      vec_t *rowA = (vec_t *) & (AO[l << 5]);
	      vec_t *rowB = (vec_t *) & (BO[l << 4]);
	      MMA (&acc0, rowB[0], rowA[0]);
	      MMA (&acc1, rowB[1], rowA[0]);
	      MMA (&acc2, rowB[0], rowA[1]);
	      MMA (&acc3, rowB[1], rowA[1]);
	      MMA (&acc4, rowB[0], rowA[2]);
	      MMA (&acc5, rowB[1], rowA[2]);
	      MMA (&acc6, rowB[0], rowA[3]);
	      MMA (&acc7, rowB[1], rowA[3]);
	    }
	  if (temp % 2 == 1)
	    {
	      if (temp > 1)
	 l = (temp / 2) << 4;
	      vec_t *rowA = (vec_t *) & (AO[l << 1]);
	      vec_t *rowB = (vec_t *) & (BO[l]);
	      vec_t rowB_h = MERGE_HIGH (rowB[0], vzero);
	      vec_t rowB_l = MERGE_LOW (rowB[0], vzero);
	      vec_t rowA_h = MERGE_HIGH (rowA[0], vzero);
	      vec_t rowA_l = MERGE_LOW (rowA[0], vzero);
	      vec_t rowA2_h = MERGE_HIGH (rowA[1], vzero);
	      vec_t rowA2_l = MERGE_LOW (rowA[1], vzero);
	      MMA (&acc0, rowB_h, rowA_h);
	      MMA (&acc1, rowB_l, rowA_h);
	      MMA (&acc2, rowB_h, rowA_l);
	      MMA (&acc3, rowB_l, rowA_l);
	      MMA (&acc4, rowB_h, rowA2_h);
	      MMA (&acc5, rowB_l, rowA2_h);
	      MMA (&acc6, rowB_h, rowA2_l);
	      MMA (&acc7, rowB_l, rowA2_l);
	    }
	  SAVE_ACC (&acc0, 0);
	  SAVE_ACC (&acc2, 4);
	  SAVE_ACC1 (&acc1, 0);
	  SAVE_ACC1 (&acc3, 4);
	  SAVE_ACC (&acc4, 8);
	  SAVE_ACC (&acc6, 12);
	  SAVE_ACC1 (&acc5, 8);
	  SAVE_ACC1 (&acc7, 12);
#if defined(TRMMKERNEL)
	  REFRESH_AFTER_SAVE (16, 8)
#endif
	  CO += 16;
	  AO += (temp << 4);
	  BO += (temp << 3);
	}
      if (m & 8)
	{
	  IFLOAT *BO;
#if defined(TRMMKERNEL)
	  REFRESH_POINTERS (8, 8);
#else
	  BO = B;
	  temp = k;
#endif

	  v4sf_t *rowC;
	  v4sf_t result[4];
	  __vector_quad acc0, acc1, acc2, acc3;
	  SET_ACC_ZERO4 ();
	  BLASLONG l = 0;
	  for (l = 0; l < temp / 2; l++)
	    {
	      vec_t *rowA = (vec_t *) & (AO[l << 4]);
	      vec_t *rowB = (vec_t *) & (BO[l << 4]);
	      MMA (&acc0, rowB[0], rowA[0]);
	      MMA (&acc1, rowB[1], rowA[0]);
	      MMA (&acc2, rowB[0], rowA[1]);
	      MMA (&acc3, rowB[1], rowA[1]);
	    }
	  if (temp % 2 == 1)
	    {
	      if (temp > 1)
		l = (temp / 2) << 4;
	      vec_t *rowA = (vec_t *) & (AO[l]);
	      vec_t *rowB = (vec_t *) & (BO[l]);
	      vec_t rowB_h = MERGE_HIGH (rowB[0], vzero);
	      vec_t rowB_l = MERGE_LOW (rowB[0], vzero);
	      vec_t rowA_h = MERGE_HIGH (rowA[0], vzero);
	      vec_t rowA_l = MERGE_LOW (rowA[0], vzero);
	      MMA (&acc0, rowB_h, rowA_h);
	      MMA (&acc1, rowB_l, rowA_h);
	      MMA (&acc2, rowB_h, rowA_l);
	      MMA (&acc3, rowB_l, rowA_l);
	    }
	  SAVE_ACC (&acc0, 0);
	  SAVE_ACC (&acc2, 4);
	  SAVE_ACC1 (&acc1, 0);
	  SAVE_ACC1 (&acc3, 4);
#if defined(TRMMKERNEL)
	  REFRESH_AFTER_SAVE (8, 8)
#endif
	  CO += 8;
	  AO += (temp << 3);
	  BO += (temp << 3);
	}
      if (m & 4)
	{
	  IFLOAT *BO;
#if defined(TRMMKERNEL)
	  REFRESH_POINTERS (4, 8);
#else
	  BO = B;
	  temp = k;
#endif

	  v4sf_t *rowC;
	  v4sf_t result[4];
	  __vector_quad acc0, acc1;
	  __builtin_mma_xxsetaccz (&acc0);
	  __builtin_mma_xxsetaccz (&acc1);
	  BLASLONG l = 0;
	  for (l = 0; l < temp / 2; l++)
	    {
	      vec_t *rowA = (vec_t *) & (AO[l << 3]);
	      vec_t *rowB = (vec_t *) & (BO[l << 4]);
	      MMA (&acc0, rowB[0], rowA[0]);
	      MMA (&acc1, rowB[1], rowA[0]);
	    }
	  if (temp % 2 == 1)
	    {
	      if (temp > 1)
		l = (temp / 2) << 3;
	      vector short rowA =
		{ AO[l + 0], 0, AO[l + 1], 0, AO[l + 2], 0, AO[l + 3], 0 };
	      vec_t *rowB = (vec_t *) & (BO[l << 1]);
	      MMA (&acc0, MERGE_HIGH (rowB[0], vzero), (vec_t) rowA);
	      MMA (&acc1, MERGE_LOW (rowB[0], vzero), (vec_t) rowA);
	    }
	  SAVE_ACC (&acc0, 0);
	  SAVE_ACC1 (&acc1, 0);
#if defined(TRMMKERNEL)
	  REFRESH_AFTER_SAVE (4, 8)
#endif
	  CO += 4;
	  AO += (temp << 2);
	  BO += (temp << 3);
	}
      if (m & 2)
	{
	  IFLOAT *BO;
#if defined(TRMMKERNEL)
	  REFRESH_POINTERS (2, 8);
#else
	  BO = B;
	  temp = k;
#endif

	  v2sf_t *rowC;
	  v2sf_t result[8];
	  __vector_quad acc0, acc1;
	  __builtin_mma_xxsetaccz (&acc0);
	  __builtin_mma_xxsetaccz (&acc1);
	  BLASLONG l = 0;
	  for (l = 0; l < temp / 2; l++)
	    {
	      vector short rowA =
		{ AO[(l << 2) + 0], AO[(l << 2) + 2], AO[(l << 2) + 1],
		AO[(l << 2) + 3],
		0, 0, 0, 0
	      };
	      vec_t *rowB = (vec_t *) & (BO[l << 4]);
	      MMA (&acc0, rowB[0], (vec_t) rowA);
	      MMA (&acc1, rowB[1], (vec_t) rowA);
	    }
	  if (temp % 2 == 1)
	    {
	      if (temp > 1)
		l = (temp / 2) << 2;
	      vector short rowA = { AO[l + 0], 0, AO[l + 1], 0, 0, 0, 0, 0 };
	      vec_t *rowB = (vec_t *) & (BO[(l << 2)]);
	      MMA (&acc0, MERGE_HIGH (rowB[0], vzero), (vec_t) rowA);
	      MMA (&acc1, MERGE_LOW (rowB[0], vzero), (vec_t) rowA);
	    }
	  SAVE4x2_ACC (&acc0, 0);
	  SAVE4x2_ACC1 (&acc1, 0);
#if defined(TRMMKERNEL)
	  REFRESH_AFTER_SAVE (2, 8)
#endif
	  CO += 2;
	  AO += (temp << 1);
	  BO += (temp << 3);
	}
      if (m & 1)
	{
	  IFLOAT *BO;
#if defined(TRMMKERNEL)
	  REFRESH_POINTERS (1, 8);
#else
	  BO = B;
	  temp = k;
#endif

	  v4sf_t result[4], res[4];
	  __vector_quad acc0, acc1;
	  __builtin_mma_xxsetaccz (&acc0);
	  __builtin_mma_xxsetaccz (&acc1);
	  BLASLONG l = 0;
	  for (l = 0; l < temp / 2; l++)
	    {
	      vector short rowA =
		{ AO[(l << 1) + 0], AO[(l << 1) + 1], 0, 0, 0, 0, 0, 0};
	      vec_t *rowB = (vec_t *) & (BO[l << 4]);
	      MMA (&acc0, rowB[0], (vec_t) rowA);
	      MMA (&acc1, rowB[1], (vec_t) rowA);
	    }
	  if (temp % 2 == 1)
	    {
	      if (temp > 1)
		l = (temp / 2) << 1;
	      vector short rowA = { AO[l], 0, 0, 0, 0, 0, 0, 0 };
	      vec_t *rowB = (vec_t *) & (BO[(l << 3)]);
	      MMA (&acc0, MERGE_HIGH (rowB[0], vzero), (vec_t) rowA);
	      MMA (&acc1, MERGE_LOW (rowB[0], vzero), (vec_t) rowA);
	    }
	  SAVE4x2_ACC_SCALAR  (&acc0);
	  SAVE4x2_ACC1_SCALAR (&acc1);
#if defined(TRMMKERNEL)
	  REFRESH_AFTER_SAVE (1, 8)
#endif
	  CO += 1;
	  AO += temp;
	  BO += (temp << 3);
	}
      B += k << 3;
    }

  if (n & 4)
    {
      BLASLONG j,temp;
      FLOAT *CO;
      IFLOAT *AO;
      CO = C;
      C += ldc << 2;
      AO = A;
      /* Loop for m >= 32. */
      for (j = 0; j < (m >> 5); j++)
	{
	  IFLOAT *BO;
#if defined(TRMMKERNEL)
	  REFRESH_POINTERS (32, 4);
#else
	  BO = B;
	  temp = k;
#endif

	  IFLOAT *A1 = AO + (16 * temp);
	  v4sf_t *rowC;
	  v4sf_t result[4];
	  __vector_quad acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7;
	  SET_ACC_ZERO8 ();
	  BLASLONG l = 0;
	  for (l = 0; l < temp / 2; l++)
	    {
	      vec_t *rowA = (vec_t *) & (AO[l << 5]);
	      vec_t *rowA1 = (vec_t *) & (A1[l << 5]);
	      vec_t *rowB = (vec_t *) & (BO[l << 3]);
	      MMA (&acc0, rowB[0], rowA[0]);
	      MMA (&acc1, rowB[0], rowA[1]);
	      MMA (&acc2, rowB[0], rowA[2]);
	      MMA (&acc3, rowB[0], rowA[3]);
	      MMA (&acc4, rowB[0], rowA1[0]);
	      MMA (&acc5, rowB[0], rowA1[1]);
	      MMA (&acc6, rowB[0], rowA1[2]);
	      MMA (&acc7, rowB[0], rowA1[3]);
	    }
	  if (temp % 2 == 1)
	    {
	      if (temp > 1)
	 l = (temp / 2) << 3;
	      vec_t *rowA = (vec_t *) & (AO[(l << 2)]);
	      vec_t *rowA1 = (vec_t *) & (A1[(l << 2)]);
	      vector short rowB_mrg =
	 { BO[l], 0, BO[l + 1], 0, BO[l + 2], 0, BO[l + 3], 0 };
	      MMA (&acc0, (vec_t)rowB_mrg, MERGE_HIGH (rowA[0], vzero));
	      MMA (&acc1, (vec_t)rowB_mrg, MERGE_LOW (rowA[0], vzero));
	      MMA (&acc2, (vec_t)rowB_mrg, MERGE_HIGH (rowA[1], vzero));
	      MMA (&acc3, (vec_t)rowB_mrg, MERGE_LOW (rowA[1], vzero));
	      MMA (&acc4, (vec_t)rowB_mrg, MERGE_HIGH (rowA1[0], vzero));
	      MMA (&acc5, (vec_t)rowB_mrg, MERGE_LOW (rowA1[0], vzero));
	      MMA (&acc6, (vec_t)rowB_mrg, MERGE_HIGH (rowA1[1], vzero));
	      MMA (&acc7, (vec_t)rowB_mrg, MERGE_LOW (rowA1[1], vzero));
	    }
	  SAVE_ACC (&acc0, 0);
	  SAVE_ACC (&acc1, 4);
	  CO += 8;
	  SAVE_ACC (&acc2, 0);
	  SAVE_ACC (&acc3, 4);
	  CO += 8;
	  SAVE_ACC (&acc4, 0);
	  SAVE_ACC (&acc5, 4);
	  CO += 8;
	  SAVE_ACC (&acc6, 0);
	  SAVE_ACC (&acc7, 4);
	  CO += 8;
#if defined(TRMMKERNEL)
	  REFRESH_AFTER_SAVE (32, 4)
#endif
	  AO += temp << 5;
	  BO += temp << 2;
	}
      if (m & 16)
	{
	  IFLOAT *BO;
#if defined(TRMMKERNEL)
	  REFRESH_POINTERS (16, 4);
#else
	  BO = B;
	  temp = k;
#endif

	  v4sf_t *rowC;
	  v4sf_t result[4];
	  __vector_quad acc0, acc1, acc2, acc3;
	  SET_ACC_ZERO4 ();
	  BLASLONG l = 0;
	  for (l = 0; l < temp / 2; l++)
	    {
	      vec_t *rowA = (vec_t *) & (AO[l << 5]);
	      vec_t *rowB = (vec_t *) & (BO[l << 3]);
	      MMA (&acc0, rowB[0], rowA[0]);
	      MMA (&acc1, rowB[0], rowA[1]);
	      MMA (&acc2, rowB[0], rowA[2]);
	      MMA (&acc3, rowB[0], rowA[3]);
	    }
	  if (temp % 2 == 1)
	    {
	      if (temp > 1)
		l = (temp / 2) << 3;
	      vec_t *rowA = (vec_t *) & (AO[(l << 2)]);
	      vector short rowB_mrg =
		{ BO[l], 0, BO[l + 1], 0, BO[l + 2], 0, BO[l + 3], 0 };
	      MMA (&acc0, (vec_t)rowB_mrg, MERGE_HIGH (rowA[0], vzero));
	      MMA (&acc1, (vec_t)rowB_mrg, MERGE_LOW (rowA[0], vzero));
	      MMA (&acc2, (vec_t)rowB_mrg, MERGE_HIGH (rowA[1], vzero));
	      MMA (&acc3, (vec_t)rowB_mrg, MERGE_LOW (rowA[1], vzero));
	    }
	  SAVE_ACC (&acc0, 0);
	  SAVE_ACC (&acc1, 4);
	  CO += 8;
	  SAVE_ACC (&acc2, 0);
	  SAVE_ACC (&acc3, 4);
	  CO += 8;
#if defined(TRMMKERNEL)
	  REFRESH_AFTER_SAVE (16, 4)
#endif
	  AO += temp << 4;
	  BO += temp << 2;
	}
      if (m & 8)
	{
	  IFLOAT *BO;
#if defined(TRMMKERNEL)
	  REFRESH_POINTERS (8, 4);
#else
	  BO = B;
	  temp = k;
#endif

	  v4sf_t *rowC;
	  v4sf_t result[4];
	  __vector_quad acc0, acc1;
	  __builtin_mma_xxsetaccz (&acc0);
	  __builtin_mma_xxsetaccz (&acc1);
	  BLASLONG l = 0;
	  for (l = 0; l < temp / 2; l++)
	    {
	      vec_t *rowA = (vec_t *) & (AO[l << 4]);
	      vec_t *rowB = (vec_t *) & (BO[l << 3]);
	      MMA (&acc0, rowB[0], rowA[0]);
	      MMA (&acc1, rowB[0], rowA[1]);
	    }
	  if (temp % 2 == 1)
	    {
	      if (temp > 1)
		l = (temp / 2) << 3;
	      vec_t *rowA = (vec_t *) & (AO[l << 1]);
	      vector short rowB_mrg =
		{ BO[l], 0, BO[l + 1], 0, BO[l + 2], 0, BO[l + 3], 0 };
	      MMA (&acc0, (vec_t)rowB_mrg, MERGE_HIGH (rowA[0], vzero));
	      MMA (&acc1, (vec_t)rowB_mrg, MERGE_LOW (rowA[0], vzero));
	    }
	  SAVE_ACC (&acc0, 0);
	  SAVE_ACC (&acc1, 4);
#if defined(TRMMKERNEL)
	  REFRESH_AFTER_SAVE (8, 4)
#endif
	  CO += 8;
	  AO += temp << 3;
	  BO += temp << 2;
	}
      if (m & 4)
	{
	  IFLOAT *BO;
#if defined(TRMMKERNEL)
	  REFRESH_POINTERS (4, 4);
#else
	  BO = B;
	  temp = k;
#endif

	  v4sf_t *rowC;
	  __vector_quad acc0;
	  v4sf_t result[4];
	  BLASLONG l = 0;
	  __builtin_mma_xxsetaccz (&acc0);
	  for (l = 0; l < temp / 2; l++)
	    {
	      vec_t *rowA = (vec_t *) & (AO[l << 3]);
	      vec_t *rowB = (vec_t *) & (BO[l << 3]);
	      MMA (&acc0, rowB[0], rowA[0]);
	    }
	  if (temp % 2 == 1)
	    {
	      if (temp > 1)
		l = (temp / 2) << 3;
	      vector short rowA =
		{ AO[l], 0, AO[l + 1], 0, AO[l + 2], 0, AO[l + 3], 0 };
	      vector short rowB_mrg =
		{ BO[l], 0, BO[l + 1], 0, BO[l + 2], 0, BO[l + 3], 0 };
	      MMA (&acc0, (vec_t)(rowB_mrg), (vec_t) rowA);
	    }
	  SAVE_ACC (&acc0, 0);
#if defined(TRMMKERNEL)
	  REFRESH_AFTER_SAVE (4, 4)
#endif
	  CO += 4;
	  AO += temp << 2;
	  BO += temp << 2;
	}
      if (m & 2)
	{
	  IFLOAT *BO;
#if defined(TRMMKERNEL)
	  REFRESH_POINTERS (2, 4);
#else
	  BO = B;
	  temp = k;
#endif

	  v2sf_t *rowC;
	  v2sf_t result[8];
	  __vector_quad acc0;
	  BLASLONG l = 0;
	  __builtin_mma_xxsetaccz (&acc0);
	  for (l = 0; l < temp / 2; l++)
	    {
	      vector short rowA =
		{ AO[(l << 2) + 0], AO[(l << 2) + 2], AO[(l << 2) + 1],
		AO[(l << 2) + 3],
		0, 0, 0, 0
	      };
	      vec_t *rowB = (vec_t *) & (BO[l << 3]);
	      MMA (&acc0, rowB[0], (vec_t) rowA);
	    }
	  if (temp % 2 == 1)
	    {
	      if (temp > 1)
		l = (temp / 2) << 2;
	      vector short rowA = { AO[l], 0, AO[l + 1], 0, 0, 0, 0, 0 };
	      vector short rowB_mrg =
		{ BO[(l<<1)], 0, BO[(l<<1) + 1], 0, BO[(l<<1) + 2], 0,
		BO[(l<<1) + 3], 0
	      };
	      MMA (&acc0, (vec_t)(rowB_mrg), (vec_t) rowA);
	    }
	  SAVE4x2_ACC (&acc0, 0);
#if defined(TRMMKERNEL)
	  REFRESH_AFTER_SAVE (2, 4)
#endif
	  CO += 2;
	  AO += temp << 1;
	  BO += temp << 2;
	}
      if (m & 1)
	{
	  IFLOAT *BO;
#if defined(TRMMKERNEL)
	  REFRESH_POINTERS (1, 4);
#else
	  BO = B;
	  temp = k;
#endif

	  v4sf_t result[4], res[4];
	  __vector_quad acc0;
	  BLASLONG l = 0;
	  __builtin_mma_xxsetaccz (&acc0);
	  for (l = 0; l < temp / 2; l++)
	    {
	      vector short rowA =
		{ AO[(l << 1) + 0], AO[(l << 1) + 1], 0,
		0, 0, 0, 0
	      };
	      vec_t *rowB = (vec_t *) & (BO[l << 3]);
	      MMA (&acc0, rowB[0], (vec_t) rowA);
	    }
	  if (temp % 2 == 1)
	    {
	      if (temp > 1)
		l = (temp / 2) << 1;
	      vector short rowA = { AO[l], 0, 0, 0, 0, 0, 0, 0 };
	      vector short rowB_mrg =
		{ BO[(l<<2) + 0], 0, BO[(l<<2) + 1], 0, BO[(l <<2) + 2], 0,
		BO[(l<<2) + 3], 0
	      };
	      MMA (&acc0, (vec_t)(rowB_mrg), (vec_t) rowA);
	    }
	  SAVE4x2_ACC_SCALAR (&acc0);
	  AO += temp;
	  BO += temp << 2;
#if defined(TRMMKERNEL)
	  REFRESH_AFTER_SAVE (1, 4)
#endif
	  CO += 1;
	}

      B += k << 2;
    }

  if (n & 2)
    {
      BLASLONG j, temp;
      FLOAT *CO;
      IFLOAT *AO;
      CO = C;
      C += ldc << 1;
      AO = A;
      /* Loop for m >= 32. */
      for (j = 0; j < (m >> 5); j++)
	{
	  IFLOAT *BO;
#if defined(TRMMKERNEL)
	  REFRESH_POINTERS (32, 2);
#else
	  BO = B;
	  temp = k;
#endif

	  v4sf_t *rowC;
	  v4sf_t result[4];
	  IFLOAT *A1 = AO + (16 * temp);
	  __vector_quad acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7;
	  SET_ACC_ZERO8 ();
	  BLASLONG l = 0;
	  for (l = 0; l < temp / 2; l++)
	    {
	      vector short rowB =
	 { BO[(l << 2) + 0], BO[(l << 2) + 2], BO[(l << 2) + 1],
	 BO[(l << 2) + 3],
	 0, 0, 0, 0
	      };
	      vec_t *rowA = (vec_t *) & (AO[l << 5]);
	      vec_t *rowA1 = (vec_t *) & (A1[l << 5]);
	      MMA (&acc0, (vec_t) rowB, rowA[0]);
	      MMA (&acc1, (vec_t) rowB, rowA[1]);
	      MMA (&acc2, (vec_t) rowB, rowA[2]);
	      MMA (&acc3, (vec_t) rowB, rowA[3]);
	      MMA (&acc4, (vec_t) rowB, rowA1[0]);
	      MMA (&acc5, (vec_t) rowB, rowA1[1]);
	      MMA (&acc6, (vec_t) rowB, rowA1[2]);
	      MMA (&acc7, (vec_t) rowB, rowA1[3]);
	    }
	  if (temp % 2 == 1)
	    {
	      if (temp > 1)
	 l = (temp / 2) << 2;
	      vector short rowB = { BO[l + 0], 0, BO[l + 1], 0, 0, 0, 0, 0 };
	      vec_t *rowA = (vec_t *) & (AO[l << 3]);
	      vec_t *rowA1 = (vec_t *) & (A1[l << 3]);
	      MMA (&acc0, (vec_t) rowB, MERGE_HIGH (rowA[0], vzero));
	      MMA (&acc1, (vec_t) rowB, MERGE_LOW (rowA[0], vzero));
	      MMA (&acc2, (vec_t) rowB, MERGE_HIGH (rowA[1], vzero));
	      MMA (&acc3, (vec_t) rowB, MERGE_LOW (rowA[1], vzero));
	      MMA (&acc4, (vec_t) rowB, MERGE_HIGH (rowA1[0], vzero));
	      MMA (&acc5, (vec_t) rowB, MERGE_LOW (rowA1[0], vzero));
	      MMA (&acc6, (vec_t) rowB, MERGE_HIGH (rowA1[1], vzero));
	      MMA (&acc7, (vec_t) rowB, MERGE_LOW (rowA1[1], vzero));
	    }
	  SAVE2x4_ACC (&acc0, 0);
	  SAVE2x4_ACC (&acc1, 4);
	  SAVE2x4_ACC (&acc2, 8);
	  SAVE2x4_ACC (&acc3, 12);
	  CO += 16;
	  SAVE2x4_ACC (&acc4, 0);
	  SAVE2x4_ACC (&acc5, 4);
	  SAVE2x4_ACC (&acc6, 8);
	  SAVE2x4_ACC (&acc7, 12);
	  CO += 16;
#if defined(TRMMKERNEL)
	  REFRESH_AFTER_SAVE (32, 2)
#endif
	  AO += temp << 5;
	  BO += temp << 1;
	}
      if (m & 16)
	{
	  IFLOAT *BO;
#if defined(TRMMKERNEL)
	  REFRESH_POINTERS (16, 2);
#else
	  BO = B;
	  temp = k;
#endif

	  v4sf_t *rowC;
	  v4sf_t result[4];
	  __vector_quad acc0, acc1, acc2, acc3;
	  SET_ACC_ZERO4 ();
	  BLASLONG l = 0;
	  for (l = 0; l < temp / 2; l++)
	    {
	      vector short rowB =
		{ BO[(l << 2) + 0], BO[(l << 2) + 2], BO[(l << 2) + 1],
		BO[(l << 2) + 3],
		0, 0, 0, 0
	      };
	      vec_t *rowA = (vec_t *) & (AO[l << 5]);
	      MMA (&acc0, (vec_t) rowB, rowA[0]);
	      MMA (&acc1, (vec_t) rowB, rowA[1]);
	      MMA (&acc2, (vec_t) rowB, rowA[2]);
	      MMA (&acc3, (vec_t) rowB, rowA[3]);
	    }
	  if (temp % 2 == 1)
	    {
	      if (temp > 1)
		l = (temp / 2) << 2;
	      vector short rowB = { BO[l + 0], 0, BO[l + 1], 0, 0, 0, 0, 0 };
	      vec_t *rowA = (vec_t *) & (AO[l << 3]);
	      MMA (&acc0, (vec_t) rowB, MERGE_HIGH (rowA[0], vzero ));
	      MMA (&acc1, (vec_t) rowB, MERGE_LOW (rowA[0], vzero));
	      MMA (&acc2, (vec_t) rowB, MERGE_HIGH (rowA[1], vzero));
	      MMA (&acc3, (vec_t) rowB, MERGE_LOW (rowA[1], vzero));
	    }
	  SAVE2x4_ACC (&acc0, 0);
	  SAVE2x4_ACC (&acc1, 4);
	  SAVE2x4_ACC (&acc2, 8);
	  SAVE2x4_ACC (&acc3, 12);
#if defined(TRMMKERNEL)
	  REFRESH_AFTER_SAVE (16, 2)
#endif
	  CO += 16;
	  AO += temp << 4;
	  BO += temp << 1;
	}
      if (m & 8)
	{
	  IFLOAT *BO;
#if defined(TRMMKERNEL)
	  REFRESH_POINTERS (8, 2);
#else
	  BO = B;
	  temp = k;
#endif

	  v4sf_t *rowC;
	  v4sf_t result[4];
	  __vector_quad acc0, acc1;
	  __builtin_mma_xxsetaccz (&acc0);
	  __builtin_mma_xxsetaccz (&acc1);
	  BLASLONG l = 0;
	  for (l = 0; l < temp / 2; l++)
	    {
	      vector short rowB =
		{ BO[(l << 2) + 0], BO[(l << 2) + 2], BO[(l << 2) + 1],
		BO[(l << 2) + 3],
		0, 0, 0, 0
	      };
	      vec_t *rowA = (vec_t *) & (AO[l << 4]);
	      MMA (&acc0, (vec_t) rowB, rowA[0]);
	      MMA (&acc1, (vec_t) rowB, rowA[1]);
	    }
	  if (temp % 2 == 1)
	    {
	      if (temp > 1)
		l = (temp / 2) << 2;
	      vector short rowB = { BO[l + 0], 0, BO[l + 1], 0, 0, 0, 0, 0 };
	      vec_t *rowA = (vec_t *) & (AO[(l << 2)]);
	      MMA (&acc0, (vec_t) rowB, MERGE_HIGH (rowA[0], vzero));
	      MMA (&acc1, (vec_t) rowB, MERGE_LOW (rowA[0], vzero));
	    }
	  SAVE2x4_ACC (&acc0, 0);
	  SAVE2x4_ACC (&acc1, 4);
#if defined(TRMMKERNEL)
	  REFRESH_AFTER_SAVE (8, 2)
#endif
	  CO += 8;
	  AO += temp << 3;
	  BO += temp << 1;
	}
      if (m & 4)
	{
	  IFLOAT *BO;
#if defined(TRMMKERNEL)
	  REFRESH_POINTERS (4, 2);
#else
	  BO = B;
	  temp = k;
#endif

	  v4sf_t *rowC;
	  v4sf_t result[4];
	  __vector_quad acc0;
	  __builtin_mma_xxsetaccz (&acc0);
	  BLASLONG l = 0;
	  for (l = 0; l < temp / 2; l++)
	    {
	      vector short rowB =
		{ BO[(l << 2) + 0], BO[(l << 2) + 2], BO[(l << 2) + 1],
		BO[(l << 2) + 3],
		0, 0, 0, 0
	      };
	      vec_t *rowA = (vec_t *) & (AO[l << 3]);
	      MMA (&acc0, (vec_t) rowB, rowA[0]);
	    }
	  if (temp % 2 == 1)
	    {
	      if (temp > 1)
		l = (temp / 2) << 2;
	      vector short rowB = { BO[l + 0], 0, BO[l + 1], 0, 0, 0, 0, 0 };
	      vector short rowA =
	        { AO[(l << 1)], 0, AO[(l << 1) + 1] , 0 , AO[(l<<1) + 2],
	        0, AO[(l << 1) + 3], 0 };
	      MMA (&acc0, (vec_t) rowB, (vec_t)(rowA));
	    }
	  SAVE2x4_ACC (&acc0, 0);
#if defined(TRMMKERNEL)
	  REFRESH_AFTER_SAVE (4, 2)
#endif
	  CO += 4;
	  AO += temp << 2;
	  BO += temp << 1;
	}
      if (m & 2)
	{
	  IFLOAT *BO;
#if defined(TRMMKERNEL)
	  REFRESH_POINTERS (2, 2);
#else
	  BO = B;
	  temp = k;
#endif

	  BLASLONG l = 0;
	  v4sf_t t = { 0, 0, 0, 0 };
	  for (l = 0; l < (temp << 1); l += 2)
	    {
	      v4sf_t rowA =
		{ HF(AO[l]), HF(AO[l]), HF(AO[l + 1]),
		HF(AO[l + 1])
		     };
		     v4sf_t rowB =
		{ HF(BO[l]), HF(BO[l + 1]), HF(BO[l]),
		HF(BO[l + 1])
	      };
	      t += rowA * rowB;
	    }
	  t = t * valpha;
	  CO[0 * ldc] += t[0];
	  CO[1 * ldc] += t[1];
	  CO[0 * ldc + 1] += t[2];
	  CO[1 * ldc + 1] += t[3];
#if defined(TRMMKERNEL)
	  REFRESH_AFTER_SAVE (2, 2)
#endif
	  CO += 2;
	  AO += temp << 1;
	  BO += temp << 1;
	}
      if (m & 1)
	{
	  IFLOAT *BO;
#if defined(TRMMKERNEL)
	  REFRESH_POINTERS (1, 2);
#else
	  BO = B;
	  temp = k;
#endif

	  BLASLONG l = 0;
	  v4sf_t t = { 0, 0, 0, 0 };
	  for (l = 0; l < temp; l++)
	    {
	      v4sf_t rowA = { HF(AO[l]), HF(AO[l]), 0, 0 };
	      v4sf_t rowB =
	 { HF(BO[l << 1]), HF(BO[(l << 1) + 1]), 0,
		0
	      };
	      t += rowA * rowB;
	    }
	  CO[0 * ldc] += t[0] * alpha;
	  CO[1 * ldc] += t[1] * alpha;
#if defined(TRMMKERNEL)
	  REFRESH_AFTER_SAVE (1, 2)
#endif
	  CO += 1;
	  AO += temp;
	  BO += temp << 1;
	}
      B += k << 1;
    }

  if (n & 1)
    {
      BLASLONG j, temp;
      FLOAT *CO;
      IFLOAT *AO;
      CO = C;
      C += ldc;
      AO = A;
      /* Loop for m >= 16. */
      for (j = 0; j < (m >> 4); j++)
	{
	  IFLOAT *BO;
#if defined(TRMMKERNEL)
	  REFRESH_POINTERS (16, 1);
#else
	  BO = B;
	  temp = k;
#endif

	  v4sf_t *rowC;
	  v4sf_t result[4];
	  __vector_quad acc0, acc1, acc2, acc3;
	  SET_ACC_ZERO4 ();
	  BLASLONG l = 0;
	  for (l = 0; l < temp / 2; l++)
	    {
	      vector short rowB =
		{ BO[l << 1], BO[(l << 1) + 1], 0, 0, 0, 0, 0, 0};
	      vec_t *rowA = (vec_t *) & (AO[l << 5]);
	      MMA (&acc0, (vec_t) rowB, rowA[0]);
	      MMA (&acc1, (vec_t) rowB, rowA[1]);
	      MMA (&acc2, (vec_t) rowB, rowA[2]);
	      MMA (&acc3, (vec_t) rowB, rowA[3]);
	    }
	  if (temp % 2 == 1)
	    {
	      if (temp > 1)
		l = (temp / 2) << 1;
	      vector short rowB = { BO[l], 0, 0, 0, 0, 0, 0, 0 };
	      vec_t *rowA = (vec_t *) & (AO[(l << 4)]);
	      MMA (&acc0, (vec_t) rowB, MERGE_HIGH (rowA[0], vzero));
	      MMA (&acc1, (vec_t) rowB, MERGE_LOW (rowA[0], vzero));
	      MMA (&acc2, (vec_t) rowB, MERGE_HIGH (rowA[1], vzero));
	      MMA (&acc3, (vec_t) rowB, MERGE_LOW (rowA[1], vzero));
	    }
	  rowC = (v4sf_t *) &CO[0];
	  __builtin_mma_disassemble_acc ((void *)result, &acc0);
          rowC[0] += result[0] * alpha;
	  __builtin_mma_disassemble_acc ((void *)result, &acc1);
          rowC[1] += result[0] * alpha;
	  __builtin_mma_disassemble_acc ((void *)result, &acc2);
          rowC[2] += result[0] * alpha;
	  __builtin_mma_disassemble_acc ((void *)result, &acc3);
          rowC[3] += result[0] * alpha;
	  AO += temp << 4;
	  BO += temp;
#if defined(TRMMKERNEL)
	  REFRESH_AFTER_SAVE (16, 1)
#endif
	  CO += 16;
	}
      /* Loop for m >= 8. */
      if (m & 8)
	{
	  IFLOAT *BO;
#if defined(TRMMKERNEL)
	  REFRESH_POINTERS (8, 1);
#else
	  BO = B;
	  temp = k;
#endif

	  v4sf_t *rowC;
	  v4sf_t result[4];
	  __vector_quad acc0, acc1;
	  __builtin_mma_xxsetaccz (&acc0);
	  __builtin_mma_xxsetaccz (&acc1);
	  BLASLONG l = 0;
	  for (l = 0; l < temp / 2; l++)
	    {
	      vector short rowB =
		{ BO[l << 1], BO[(l << 1) + 1], 0, 0, 0, 0, 0, 0};
	      vec_t *rowA = (vec_t *) & (AO[l << 4]);
	      MMA (&acc0, (vec_t) rowB, rowA[0]);
	      MMA (&acc1, (vec_t) rowB, rowA[1]);
	    }
	  if (temp % 2 == 1)
	    {
	      if (temp > 1)
		l = (temp / 2) << 1;
	      vector short rowB = { BO[l], 0, 0, 0, 0, 0, 0, 0 };
	      vec_t *rowA = (vec_t *) & (AO[(l << 3)]);
	      MMA (&acc0, (vec_t) rowB, MERGE_HIGH (rowA[0], vzero));
	      MMA (&acc1, (vec_t) rowB, MERGE_LOW (rowA[0], vzero));
	    }
	  rowC = (v4sf_t *) &CO[0];
	  __builtin_mma_disassemble_acc ((void *)result, &acc0);
          rowC[0] += result[0] * alpha;
	  __builtin_mma_disassemble_acc ((void *)result, &acc1);
          rowC[1] += result[0] * alpha;
	  AO += temp << 3;
	  BO += temp;
#if defined(TRMMKERNEL)
	  REFRESH_AFTER_SAVE (8, 1)
#endif
	  CO += 8;
	}
      /* Loop for m >= 4. */
      if (m & 4)
	{
	  IFLOAT *BO;
#if defined(TRMMKERNEL)
	  REFRESH_POINTERS (4, 1);
#else
	  BO = B;
	  temp = k;
#endif

	  v4sf_t *rowC;
	  v4sf_t result[4];
	  __vector_quad acc0;
	  __builtin_mma_xxsetaccz (&acc0);
	  BLASLONG l = 0;
	  for (l = 0; l < temp / 2; l++)
	    {
	      vector short rowB =
		{ BO[l << 1], BO[(l << 1) + 1], 0, 0, 0, 0, 0, 0};
	      vec_t *rowA = (vec_t *) & (AO[l << 3]);
	      MMA (&acc0, (vec_t) rowB, rowA[0]);
	    }
	  if (temp % 2 == 1)
	    {
	      if (temp > 1)
		l = (temp / 2) << 1;
	      vector short rowB = { BO[l], 0, 0, 0, 0, 0, 0, 0 };
	      vector short rowA =
	        { AO[(l << 2)], 0, AO[(l << 2) + 1] , 0 ,
		AO[(l << 2) + 2], 0, AO[(l << 2) + 3], 0 };
	      MMA (&acc0, (vec_t) rowB, (vec_t)(rowA));
	    }
	  rowC = (v4sf_t *) &CO[0];
	  __builtin_mma_disassemble_acc ((void *)result, &acc0);
          rowC[0] += result[0] * alpha;
	  AO += temp << 2;
	  BO += temp;
#if defined(TRMMKERNEL)
	  REFRESH_AFTER_SAVE (4, 1)
#endif
	  CO += 4;
	}
      /* Loop for m >= 2. */
      if (m & 2)
	{
	  IFLOAT *BO;
#if defined(TRMMKERNEL)
	  REFRESH_POINTERS (2, 1);
#else
	  BO = B;
	  temp = k;
#endif

	  BLASLONG l = 0;
	  v4sf_t t = { 0, 0, 0, 0 };
	  for (l = 0; l < temp; l++)
	    {
	      v4sf_t rowB = { HF(BO[l]), HF(BO[l]), 0, 0 };
	      v4sf_t rowA =
	 { HF(AO[l << 1]), HF(AO[(l << 1) + 1]), 0,
		0
	      };
	      t += rowA * rowB;
	    }
	  t = t * valpha;
	  CO[0] += t[0];
	  CO[1] += t[1];
	  AO += temp << 1;
	  BO += temp;
#if defined(TRMMKERNEL)
	  REFRESH_AFTER_SAVE (2, 1)
#endif
	  CO += 2;
	}
      /* Loop for m = 1. */
      if (m & 1)
	{
	  IFLOAT *BO;
#if defined(TRMMKERNEL)
	  REFRESH_POINTERS (1, 1);
#else
	  BO = B;
	  temp = k;
#endif

	  BLASLONG l = 0;
	  FLOAT t = 0;
	  for (l = 0; l < temp; l++)
	    {
	      t += HF(AO[l]) * HF(BO[l]);
	    }
	  AO += temp;
	  BO += temp;
	  CO[0] += t * alpha;
#if defined(TRMMKERNEL)
	  REFRESH_AFTER_SAVE (1, 1)
#endif
	  CO += 1;
	}

      B += k;
    }

  return 0;
}

