#include "stdint.h"
#include "arm_neon.h"


extern "C" int reset(uint16_t* dst) {
    // TODO change this into a vstr
    for (int i = 0; i < 8; i++)
	    dst[i] = 0;
    return 0;
}

extern "C" int update_unipolar(uint8_t* src_a, uint8_t* src_b, int16_t* dst) {
    // todo - data loading
    uint8x16_t aa[8];
    int8x16_t a[8];
    for(int i = 0; i < 8; i++)
        aa[i] = vld1q_u8(src_a + 16*i);
    uint8x16_t b = vld1q_u8(src_b);
    int16x8_t output = vld1q_s16(dst);

    // from racket phase 1: 
    for(int i = 0; i < 8; i++) {
        uint8x16_t temp = vandq_u8(aa[i], b);
        // not b and aa
        uint8x16_t temp2 = vbicq_u8(b, aa[i]);
        temp = vcntq_u8(temp);
        temp2 = vcntq_u8(temp2);
        a[i] = vsubq_s8(temp, temp2);
    }
    
    // todo - naming the upper and lower half 
    int8x8_t d0 = vget_high_s8(a[0]);
    int8x8_t d1 = vget_low_s8(a[0]);
    int8x8_t d2 = vget_high_s8(a[1]);
    int8x8_t d3 = vget_low_s8(a[1]);
    int8x8_t d4 = vget_high_s8(a[2]);
    int8x8_t d5 = vget_low_s8(a[2]);
    int8x8_t d6 = vget_high_s8(a[3]);
    int8x8_t d7 = vget_low_s8(a[3]);
    int8x8_t d8 = vget_high_s8(a[4]);
    int8x8_t d9 = vget_low_s8(a[4]);
    int8x8_t d10 = vget_high_s8(a[5]);
    int8x8_t d11 = vget_low_s8(a[5]);
    int8x8_t d12 = vget_high_s8(a[6]);
    int8x8_t d13 = vget_low_s8(a[6]);
    int8x8_t d14 = vget_high_s8(a[7]);
    int8x8_t d15 = vget_low_s8(a[7]);


    // from racket phase 2
    int8x8_t d0_ = vpadd_s8(d0, d1);
    int8x8_t d1_ = vpadd_s8(d2, d3);
    int8x8_t d2_ = vpadd_s8(d4, d5);
    int8x8_t d3_ = vpadd_s8(d6, d7);
    int8x8_t d4_ = vpadd_s8(d8, d9);
    int8x8_t d5_ = vpadd_s8(d10, d11);
    int8x8_t d6_ = vpadd_s8(d12, d13);
    int8x8_t d7_ = vpadd_s8(d14, d15);
    int8x8_t d0__ = vpadd_s8(d0_, d1_);
    int8x8_t d1__ = vpadd_s8(d2_, d3_);
    int8x8_t d2__ = vpadd_s8(d4_, d5_);
    int8x8_t d3__ = vpadd_s8(d6_, d7_);
    int8x8_t d0___ = vpadd_s8(d0__, d1__);
    int8x8_t d1___ = vpadd_s8(d2__, d3__);
    // TODO combine
    int16x8_t d0____ = vpadalq_s8(output, vcombine_s8(d0___, d1___));

    // todo data writeback
    vst1q_s16(dst, d0____);
    return 0;
}

extern "C" int update_bipolar(uint8_t* src_a, uint8_t* src_b, uint16_t* dst) {
    // todo - data loading
    uint8x16_t a[8];
    for(int i = 0; i < 8; i++)
        a[i] = vld1q_u8(src_a + 16*i);
    uint8x16_t b = vld1q_u8(src_b);
    uint16x8_t output = vld1q_u16(dst);

    // from racket phase 1: 
    for(int i = 0; i < 8; i++) {
        uint8x16_t temp = vandq_u8(a[i], b);
        a[i] = vcntq_u8(temp);
    }
    
    // todo - naming the upper and lower half 
    uint8x8_t d0 = vget_high_u8(a[0]);
    uint8x8_t d1 = vget_low_u8(a[0]);
    uint8x8_t d2 = vget_high_u8(a[1]);
    uint8x8_t d3 = vget_low_u8(a[1]);
    uint8x8_t d4 = vget_high_u8(a[2]);
    uint8x8_t d5 = vget_low_u8(a[2]);
    uint8x8_t d6 = vget_high_u8(a[3]);
    uint8x8_t d7 = vget_low_u8(a[3]);
    uint8x8_t d8 = vget_high_u8(a[4]);
    uint8x8_t d9 = vget_low_u8(a[4]);
    uint8x8_t d10 = vget_high_u8(a[5]);
    uint8x8_t d11 = vget_low_u8(a[5]);
    uint8x8_t d12 = vget_high_u8(a[6]);
    uint8x8_t d13 = vget_low_u8(a[6]);
    uint8x8_t d14 = vget_high_u8(a[7]);
    uint8x8_t d15 = vget_low_u8(a[7]);


    // from racket phase 2
    uint8x8_t d0_ = vpadd_u8(d0, d1);
    uint8x8_t d1_ = vpadd_u8(d2, d3);
    uint8x8_t d2_ = vpadd_u8(d4, d5);
    uint8x8_t d3_ = vpadd_u8(d6, d7);
    uint8x8_t d4_ = vpadd_u8(d8, d9);
    uint8x8_t d5_ = vpadd_u8(d10, d11);
    uint8x8_t d6_ = vpadd_u8(d12, d13);
    uint8x8_t d7_ = vpadd_u8(d14, d15);
    uint8x8_t d0__ = vpadd_u8(d0_, d1_);
    uint8x8_t d1__ = vpadd_u8(d2_, d3_);
    uint8x8_t d2__ = vpadd_u8(d4_, d5_);
    uint8x8_t d3__ = vpadd_u8(d6_, d7_);
    uint8x8_t d0___ = vpadd_u8(d0__, d1__);
    uint8x8_t d1___ = vpadd_u8(d2__, d3__);
    // TODO combine
    uint16x8_t d0____ = vpadalq_u8(output, vcombine_u8(d0___, d1___));

    // todo data writeback
    vst1q_u16(dst, d0____);
    return 0;
}
