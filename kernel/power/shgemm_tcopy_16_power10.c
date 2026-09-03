#include <stdio.h>
#include <altivec.h>
#include "common.h"
typedef IFLOAT vec_h16 __attribute__ ((vector_size (16)));

int CNAME(BLASLONG m, BLASLONG n, IFLOAT *a, BLASLONG lda, IFLOAT *b){

  BLASLONG i, j;

  IFLOAT *aoffset;
  IFLOAT *aoffset1, *aoffset2;
  IFLOAT *boffset;

  vec_h16 vtemp01, vtemp02, vtemp03, vtemp04;
  IFLOAT ctemp01, ctemp02, ctemp03, ctemp04;
  IFLOAT ctemp05, ctemp06, ctemp07, ctemp08;

  aoffset   = a;
  boffset   = b;

#if 0
  fprintf(stderr, "m = %d n = %d\n", m, n);
#endif

  j = (n >> 4);
  if (j > 0){
    do{
      aoffset1  = aoffset;
      aoffset2  = aoffset + lda;
      aoffset += 16;

      i = (m >> 1);
      if (i > 0){
	do{
	  vtemp01 = *(vec_h16 *)(aoffset1);
	  vtemp02 = *(vec_h16 *)(aoffset1+8);
	  vtemp03 = *(vec_h16 *)(aoffset2);
	  vtemp04 = *(vec_h16 *)(aoffset2+8);
	  *(vec_h16 *)(boffset + 0) = vec_mergeh(vtemp01, vtemp03);
	  *(vec_h16 *)(boffset + 8) = vec_mergel(vtemp01, vtemp03);
	  *(vec_h16 *)(boffset + 16) = vec_mergeh(vtemp02, vtemp04);
	  *(vec_h16 *)(boffset + 24) = vec_mergel(vtemp02, vtemp04);
	  aoffset1 +=  2 * lda;
	  aoffset2 +=  2 * lda;
	  boffset   += 32;

	  i --;
	}while(i > 0);
      }

      if (m & 1){
	vtemp01 = *(vec_h16 *)(aoffset1);
	vtemp02 = *(vec_h16 *)(aoffset1+8);
	*(vec_h16 *)(boffset + 0) = vtemp01;
	*(vec_h16 *)(boffset + 8) = vtemp02;
	boffset   += 16;
      }

      j--;
    }while(j > 0);
  } /* end of if(j > 0) */

  if (n & 8){
    aoffset1  = aoffset;
    aoffset2  = aoffset + lda;
    aoffset += 8;

    i = (m >> 1);
    if (i > 0){
      do{
	vtemp01 = *(vec_h16 *)(aoffset1);
	vtemp03 = *(vec_h16 *)(aoffset2);
	*(vec_h16 *)(boffset + 0) = vec_mergeh(vtemp01, vtemp03);
	*(vec_h16 *)(boffset + 8) = vec_mergel(vtemp01, vtemp03);

	aoffset1 +=  2 * lda;
	aoffset2 +=  2 * lda;
	boffset   += 16;

	i --;
      }while(i > 0);
    }

    if (m & 1){
      vtemp01 = *(vec_h16 *)(aoffset1);
      *(vec_h16 *)(boffset + 0) = vtemp01;
      boffset   += 8;
    }
  }

  if (n & 4){
    aoffset1  = aoffset;
    aoffset2  = aoffset + lda;
    aoffset += 4;

    i = (m >> 1);
    if (i > 0){
      do{
	ctemp01 = *(aoffset1 +  0);
	ctemp02 = *(aoffset1 +  1);
	ctemp03 = *(aoffset1 +  2);
	ctemp04 = *(aoffset1 +  3);

	ctemp05 = *(aoffset2 +  0);
	ctemp06 = *(aoffset2 +  1);
	ctemp07 = *(aoffset2 +  2);
	ctemp08 = *(aoffset2 +  3);

	*(boffset +  0) = ctemp01;
	*(boffset +  1) = ctemp05;
	*(boffset +  2) = ctemp02;
	*(boffset +  3) = ctemp06;
	*(boffset +  4) = ctemp03;
	*(boffset +  5) = ctemp07;
	*(boffset +  6) = ctemp04;
	*(boffset +  7) = ctemp08;

	aoffset1 +=  2 * lda;
	aoffset2 +=  2 * lda;
	boffset   += 8;

	i --;
      }while(i > 0);
    }

    if (m & 1){
      ctemp01 = *(aoffset1 +  0);
      ctemp02 = *(aoffset1 +  1);
      ctemp03 = *(aoffset1 +  2);
      ctemp04 = *(aoffset1 +  3);

      *(boffset +  0) = ctemp01;
      *(boffset +  1) = ctemp02;
      *(boffset +  2) = ctemp03;
      *(boffset +  3) = ctemp04;

      boffset   += 4;
    }
  }

  if (n & 2){
    aoffset1  = aoffset;
    aoffset2  = aoffset + lda;
    aoffset += 2;

    i = (m >> 1);
    if (i > 0){
      do{
	ctemp01 = *(aoffset1 +  0);
	ctemp02 = *(aoffset1 +  1);
	ctemp03 = *(aoffset2 +  0);
	ctemp04 = *(aoffset2 +  1);

	*(boffset +  0) = ctemp01;
	*(boffset +  1) = ctemp02;
	*(boffset +  2) = ctemp03;
	*(boffset +  3) = ctemp04;

	aoffset1 +=  2 * lda;
	aoffset2 +=  2 * lda;
	boffset   += 4;

	i --;
      }while(i > 0);
    }

    if (m & 1){
      ctemp01 = *(aoffset1 +  0);
      ctemp02 = *(aoffset1 +  1);

      *(boffset +  0) = ctemp01;
      *(boffset +  1) = ctemp02;
      boffset   += 2;
    }
  }

  if (n & 1){
    aoffset1  = aoffset;
    aoffset2  = aoffset + lda;

    i = (m >> 1);
    if (i > 0){
      do{
	ctemp01 = *(aoffset1 +  0);
	ctemp02 = *(aoffset2 +  0);

	*(boffset +  0) = ctemp01;
	*(boffset +  1) = ctemp02;

	aoffset1 +=  2 * lda;
	aoffset2 +=  2 * lda;
	boffset   += 2;

	i --;
      }while(i > 0);
    }

    if (m & 1){
      ctemp01 = *(aoffset1 +  0);
      *(boffset +  0) = ctemp01;
      // boffset   += 1;
    }
  }

  return 0;
}

