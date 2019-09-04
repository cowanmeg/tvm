#include "stdint.h"
#include "arm_neon.h"

extern "C" int reset(uint16_t* dst) {
    // Zero out the register
    uint16x8_t x = vld1q_u16(dst);
    x = veorq_u16(x, x);
    vst1q_u16(dst, x);
    return 0;
}

// 8x8x1
extern "C" int update_unipolar_a1b1_half(uint8_t* src_a, uint8_t* src_b, int16_t* dst, int a_str1, int a_str0, int b_str0){
    // todo - data loading
    uint8x8_t aa[8];
    int8x8_t a[8];
    for(int i = 0; i < 8; i++)
        aa[i] = vld1_u8(src_a + i * a_str0);
    uint8x8_t b = vld1_u8(src_b);
    int16x8_t output = vld1q_s16(dst);

    // from racket phase 1: 
    for(int i = 0; i < 8; i++) {
        uint8x8_t temp = vand_u8(aa[i], b);
        // not b and aa
        uint8x8_t temp2 = vbic_u8(b, aa[i]);
        temp = vcnt_u8(temp);
        temp2 = vcnt_u8(temp2);
        a[i] = vsub_s8(temp, temp2);
    }
    
    // from racket phase 2
    int8x8_t d0_ = vpadd_s8(a[0], a[1]);
    int8x8_t d1_ = vpadd_s8(a[2], a[3]);
    int8x8_t d2_ = vpadd_s8(a[4], a[5]);
    int8x8_t d3_ = vpadd_s8(a[6], a[7]);

    int8x8_t d0__ = vpadd_s8(d0_, d1_);
    int8x8_t d1__ = vpadd_s8(d2_, d3_);

    // TODO combine
    int16x8_t d0____ = vpadalq_s8(output, vcombine_s8(d0__, d1__));

    // todo data writebacks
    vst1q_s16(dst, d0____);
    return 0;
}

extern "C" int update_unipolar_a1b2_half(uint8_t* src_a, uint8_t* src_b, int16_t* dst, int a_str1, int a_str0, int b_str0){
    // todo - data loading
    uint8x8_t aa[8];
    int8x8_t a[8];
    for(int i = 0; i < 8; i++)
        aa[i] = vld1_u8(src_a + i*a_str0);
    uint8x8_t b = vld1_u8(src_b);
    int16x8_t output = vld1q_s16(dst);

    // from racket phase 1: 
    for(int i = 0; i < 8; i++) {
        uint8x8_t temp = vand_u8(aa[i], b);
        // not b and aa
        uint8x8_t temp2 = vbic_u8(b, aa[i]);
        temp = vcnt_u8(temp);
        temp2 = vcnt_u8(temp2);
        a[i] = vsub_s8(temp, temp2);
    }

    // B's second bitplane ooops forgot vshl has to get take a constant
    b = vld1_u8(src_b + b_str0);
    for(int i = 0; i < 8; i++) {
        uint8x8_t temp = vand_u8(aa[i], b);
        // not b and aa
        uint8x8_t temp2 = vbic_u8(b, aa[i]);
        temp = vcnt_u8(temp);
        temp2 = vcnt_u8(temp2);
        int8x8_t temp3 = vsub_s8(temp, temp2);
        temp3 = vshl_n_s8(temp3, 1);
        a[i] = vadd_s8(a[i], temp3);
    }
    
    // from racket phase 2
    int8x8_t d0_ = vpadd_s8(a[0], a[1]);
    int8x8_t d1_ = vpadd_s8(a[2], a[3]);
    int8x8_t d2_ = vpadd_s8(a[4], a[5]);
    int8x8_t d3_ = vpadd_s8(a[6], a[7]);

    int8x8_t d0__ = vpadd_s8(d0_, d1_);
    int8x8_t d1__ = vpadd_s8(d2_, d3_);

    // TODO combine
    int16x8_t d0____ = vpadalq_s8(output, vcombine_s8(d0__, d1__));

    // todo data writebacks
    vst1q_s16(dst, d0____);
    return 0;
}

extern "C" int update_unipolar_a2b1_half(uint8_t* src_a, uint8_t* src_b, int16_t* dst, 
    int a_str1, int a_str0, int b_str0){
    // todo - data loading
    int8x8_t a[8];

    uint8x8_t aa[8];
    for(int i = 0; i < 8; i++)
        aa[i] = vld1_u8(src_a + i*a_str0);
    uint8x8_t b0 = vld1_u8(src_b);
    int16x8_t output = vld1q_s16(dst);

    // from racket phase 1: Manually unrolling from example because vshl must be constant
    for(int i = 0; i < 8; i++) {
        uint8x8_t temp = vand_u8(aa[i], b0);
        // not b and aa
        uint8x8_t temp2 = vbic_u8(b0, aa[i]);
        temp = vcnt_u8(temp);
        temp2 = vcnt_u8(temp2);
        a[i] = vsub_s8(temp, temp2);
    }
    
    //Load bitplane 1 of a
     for(int i = 0; i < 8; i++)
        aa[i] = vld1_u8(src_a + a_str1 + 8*i);

    for(int i = 0; i < 8; i++) {
        uint8x8_t temp = vand_u8(aa[i], b0);
        // not b and aa
        uint8x8_t temp2 = vbic_u8(b0, aa[i]);
        temp = vcnt_u8(temp);
        temp2 = vcnt_u8(temp2);
        int8x8_t temp3 = vsub_s8(temp, temp2);
        temp3 = vshl_n_s8(temp3, 1);
        a[i] = vadd_s8(a[i], temp3);
    }
    
    // from racket phase 2
    int8x8_t d0_ = vpadd_s8(a[0], a[1]);
    int8x8_t d1_ = vpadd_s8(a[2], a[3]);
    int8x8_t d2_ = vpadd_s8(a[4], a[5]);
    int8x8_t d3_ = vpadd_s8(a[6], a[7]);

    int16x8_t q0_ = vpaddlq_s8(vcombine_s8(d0_, d1_));
    int16x8_t q1_ = vpaddlq_s8(vcombine_s8(d2_, d3_));
    int16x4_t d0__  = vpadd_s16(vget_low_s16(q0_), vget_high_s16(q0_));
    int16x4_t d1__  = vpadd_s16(vget_low_s16(q1_), vget_high_s16(q1_));
    // accumulate
    int16x8_t d0____ = vaddq_s16(output, vcombine_s16(d0__, d1__));

    // todo data writebacks
    vst1q_s16(dst, d0____);
    return 0;
}

extern "C" int update_unipolar_a1b3_half(uint8_t* src_a, uint8_t* src_b, int16_t* dst, 
    int a_str1, int a_str0, int b_str0){
    // todo - data loading
    int8x8_t a[8];

    uint8x8_t aa[8];
    for(int i = 0; i < 8; i++)
        aa[i] = vld1_u8(src_a + i*a_str0);
    uint8x8_t b0 = vld1_u8(src_b);
    uint8x8_t b1 = vld1_u8(src_b + b_str0);
    uint8x8_t b1 = vld1_u8(src_b + 2 * b_str0);
    int16x8_t output = vld1q_s16(dst);

    // from racket phase 1: Manually unrolling from example because vshl must be constant
    for(int i = 0; i < 8; i++) {
        uint8x8_t temp = vand_u8(aa[i], b0);
        // not b and aa
        uint8x8_t temp2 = vbic_u8(b0, aa[i]);
        temp = vcnt_u8(temp);
        temp2 = vcnt_u8(temp2);
        a[i] = vsub_s8(temp, temp2);
    }
    
    for(int i = 0; i < 8; i++) {
        uint8x8_t temp = vand_u8(aa[i], b1);
        // not b and aa
        uint8x8_t temp2 = vbic_u8(b1, aa[i]);
        temp = vcnt_u8(temp);
        temp2 = vcnt_u8(temp2);
        int8x8_t temp3 = vsub_s8(temp, temp2);
        temp3 = vshl_n_s8(temp3, 1);
        a[i] = vadd_s8(a[i], temp3);
    }

    for(int i = 0; i < 8; i++) {
        uint8x8_t temp = vand_u8(aa[i], b2);
        // not b and aa
        uint8x8_t temp2 = vbic_u8(b2, aa[i]);
        temp = vcnt_u8(temp);
        temp2 = vcnt_u8(temp2);
        int8x8_t temp3 = vsub_s8(temp, temp2);
        temp3 = vshl_n_s8(temp3, 2);
        a[i] = vadd_s8(a[i], temp3);
    }

    // from racket phase 2
    int8x8_t d0_ = vpadd_s8(a[0], a[1]);
    int8x8_t d1_ = vpadd_s8(a[2], a[3]);
    int8x8_t d2_ = vpadd_s8(a[4], a[5]);
    int8x8_t d3_ = vpadd_s8(a[6], a[7]);

    int16x8_t q0_ = vpaddlq_s8(vcombine_s8(d0_, d1_));
    int16x8_t q1_ = vpaddlq_s8(vcombine_s8(d2_, d3_));
    int16x4_t d0__  = vpadd_s16(vget_low_s16(q0_), vget_high_s16(q0_));
    int16x4_t d1__  = vpadd_s16(vget_low_s16(q1_), vget_high_s16(q1_));
    // accumulate
    int16x8_t d0____ = vaddq_s16(output, vcombine_s16(d0__, d1__));

    // todo data writebacks
    vst1q_s16(dst, d0____);
    return 0;
}

extern "C" int update_unipolar_a3b1_half(uint8_t* src_a, uint8_t* src_b, int16_t* dst, 
    int a_str1, int a_str0, int b_str0){
    // todo - data loading
    int8x8_t a[8];

    uint8x8_t aa[8];
    for(int i = 0; i < 8; i++)
        aa[i] = vld1_u8(src_a + i*a_str0);
    uint8x8_t b0 = vld1_u8(src_b);
    int16x8_t output = vld1q_s16(dst);

    // from racket phase 1: Manually unrolling from example because vshl must be constant
    for(int i = 0; i < 8; i++) {
        uint8x8_t temp = vand_u8(aa[i], b0);
        // not b and aa
        uint8x8_t temp2 = vbic_u8(b0, aa[i]);
        temp = vcnt_u8(temp);
        temp2 = vcnt_u8(temp2);
        a[i] = vsub_s8(temp, temp2);
    }
    
    //Load bitplane 1 of a
     for(int i = 0; i < 8; i++)
        aa[i] = vld1_u8(src_a + a_str1 + 8*i);

    for(int i = 0; i < 8; i++) {
        uint8x8_t temp = vand_u8(aa[i], b0);
        // not b and aa
        uint8x8_t temp2 = vbic_u8(b0, aa[i]);
        temp = vcnt_u8(temp);
        temp2 = vcnt_u8(temp2);
        int8x8_t temp3 = vsub_s8(temp, temp2);
        temp3 = vshl_n_s8(temp3, 1);
        a[i] = vadd_s8(a[i], temp3);
    }

    //Load bitplane 2 of a
     for(int i = 0; i < 8; i++)
        aa[i] = vld1_u8(src_a + 2*a_str1 + 8*i);

    for(int i = 0; i < 8; i++) {
        uint8x8_t temp = vand_u8(aa[i], b0);
        // not b and aa
        uint8x8_t temp2 = vbic_u8(b0, aa[i]);
        temp = vcnt_u8(temp);
        temp2 = vcnt_u8(temp2);
        int8x8_t temp3 = vsub_s8(temp, temp2);
        temp3 = vshl_n_s8(temp3, 1);
        a[i] = vadd_s8(a[i], temp3);
    }
    
    // from racket phase 2
    int8x8_t d0_ = vpadd_s8(a[0], a[1]);
    int8x8_t d1_ = vpadd_s8(a[2], a[3]);
    int8x8_t d2_ = vpadd_s8(a[4], a[5]);
    int8x8_t d3_ = vpadd_s8(a[6], a[7]);

    int16x8_t q0_ = vpaddlq_s8(vcombine_s8(d0_, d1_));
    int16x8_t q1_ = vpaddlq_s8(vcombine_s8(d2_, d3_));
    int16x4_t d0__  = vpadd_s16(vget_low_s16(q0_), vget_high_s16(q0_));
    int16x4_t d1__  = vpadd_s16(vget_low_s16(q1_), vget_high_s16(q1_));
    // accumulate
    int16x8_t d0____ = vaddq_s16(output, vcombine_s16(d0__, d1__));

    // todo data writebacks
    vst1q_s16(dst, d0____);
    return 0;
}

extern "C" int update_unipolar_a2b2_half(uint8_t* src_a, uint8_t* src_b, int16_t* dst, 
    int a_str1, int a_str0, int b_str0){
    // todo - data loading
    int8x8_t a[8];

    uint8x8_t aa[8];
    for(int i = 0; i < 8; i++)
        aa[i] = vld1_u8(src_a + i*a_str0);
    uint8x8_t b0 = vld1_u8(src_b);
    uint8x8_t b1 = vld1_u8(src_b + b_str0);
    int16x8_t output = vld1q_s16(dst);

    // from racket phase 1: Manually unrolling from example because vshl must be constant
    for(int i = 0; i < 8; i++) {
        uint8x8_t temp = vand_u8(aa[i], b0);
        // not b and aa
        uint8x8_t temp2 = vbic_u8(b0, aa[i]);
        temp = vcnt_u8(temp);
        temp2 = vcnt_u8(temp2);
        a[i] = vsub_s8(temp, temp2);
    }
    
    for(int i = 0; i < 8; i++) {
        uint8x8_t temp = vand_u8(aa[i], b1);
        // not b and aa
        uint8x8_t temp2 = vbic_u8(b1, aa[i]);
        temp = vcnt_u8(temp);
        temp2 = vcnt_u8(temp2);
        int8x8_t temp3 = vsub_s8(temp, temp2);
        temp3 = vshl_n_s8(temp3, 1);
        a[i] = vadd_s8(a[i], temp3);
    }

    //Load bitplane 1 of a
     for(int i = 0; i < 8; i++)
        aa[i] = vld1_u8(src_a + a_str1 + 8*i);

    for(int i = 0; i < 8; i++) {
        uint8x8_t temp = vand_u8(aa[i], b0);
        // not b and aa
        uint8x8_t temp2 = vbic_u8(b0, aa[i]);
        temp = vcnt_u8(temp);
        temp2 = vcnt_u8(temp2);
        int8x8_t temp3 = vsub_s8(temp, temp2);
        temp3 = vshl_n_s8(temp3, 1);
        a[i] = vadd_s8(a[i], temp3);
    }
    
    for(int i = 0; i < 8; i++) {
        uint8x8_t temp = vand_u8(aa[i], b1);
        // not b and aa
        uint8x8_t temp2 = vbic_u8(b1, aa[i]);
        temp = vcnt_u8(temp);
        temp2 = vcnt_u8(temp2);
        int8x8_t temp3 = vsub_s8(temp, temp2);
        temp3 = vshl_n_s8(temp3, 2);
        a[i] = vadd_s8(a[i], temp3);
    }


    // from racket phase 2
    int8x8_t d0_ = vpadd_s8(a[0], a[1]);
    int8x8_t d1_ = vpadd_s8(a[2], a[3]);
    int8x8_t d2_ = vpadd_s8(a[4], a[5]);
    int8x8_t d3_ = vpadd_s8(a[6], a[7]);

    int16x8_t q0_ = vpaddlq_s8(vcombine_s8(d0_, d1_));
    int16x8_t q1_ = vpaddlq_s8(vcombine_s8(d2_, d3_));
    int16x4_t d0__  = vpadd_s16(vget_low_s16(q0_), vget_high_s16(q0_));
    int16x4_t d1__  = vpadd_s16(vget_low_s16(q1_), vget_high_s16(q1_));
    // accumulate
    int16x8_t d0____ = vaddq_s16(output, vcombine_s16(d0__, d1__));

    // todo data writebacks
    vst1q_s16(dst, d0____);
    return 0;
}

extern "C" int update_bipolar_a1b1_half(uint8_t* src_a, uint8_t* src_b, uint16_t* dst, 
    int a_str1, int a_str0, int b_str0){
    // todo - data loading
    uint8x8_t a[8];
    for(int i = 0; i < 8; i++)
        a[i] = vld1_u8(src_a + i*a_str0);
    uint8x8_t b = vld1_u8(src_b);
    uint16x8_t output = vld1q_u16(dst);

    // from racket phase 1: 
    for(int i = 0; i < 8; i++) {
        uint8x8_t temp = vand_u8(a[i], b);
        a[i] = vcnt_u8(temp);
    }

    // from racket phase 2
    uint8x8_t d0_ = vpadd_u8(a[0], a[1]);
    uint8x8_t d1_ = vpadd_u8(a[2], a[3]);
    uint8x8_t d2_ = vpadd_u8(a[4], a[5]);
    uint8x8_t d3_ = vpadd_u8(a[6], a[7]);
    
    uint8x8_t d0__ = vpadd_u8(d0_, d1_);
    uint8x8_t d1__ = vpadd_u8(d2_, d3_);
    
    // TODO: print combine
    uint16x8_t d0____ = vpadalq_u8(output, vcombine_u8(d0__, d1__));

    // todo data writeback
    vst1q_u16(dst, d0____);
    return 0;
}

extern "C" int update_bipolar_a1b2_half(uint8_t* src_a, uint8_t* src_b, uint16_t* dst, 
    int a_str1, int a_str0, int b_str0){
    // todo - data loading
    uint8x8_t aa[8];
    uint8x8_t a[8];
    for(int i = 0; i < 8; i++)
        aa[i] = vld1_u8(src_a + i*a_str0);
    uint8x8_t b = vld1_u8(src_b);
    uint16x8_t output = vld1q_u16(dst);

    // from racket phase 1: 
    for(int i = 0; i < 8; i++) {
        uint8x8_t temp = vand_u8(aa[i], b);
        a[i] = vcnt_u8(temp);
    }

    // B's second bitplane ooops forgot vshl has to get take a constant
    b = vld1_u8(src_b + b_str0);
    for(int i = 0; i < 8; i++) {
        uint8x8_t temp = vand_u8(aa[i], b);
        temp = vcnt_u8(temp);
        temp = vshl_n_u8(temp, 1);
        a[i] = vadd_u8(a[i], temp);
    }

    // from racket phase 2
    uint8x8_t d0_ = vpadd_u8(a[0], a[1]);
    uint8x8_t d1_ = vpadd_u8(a[2], a[3]);
    uint8x8_t d2_ = vpadd_u8(a[4], a[5]);
    uint8x8_t d3_ = vpadd_u8(a[6], a[7]);
    
    uint8x8_t d0__ = vpadd_u8(d0_, d1_);
    uint8x8_t d1__ = vpadd_u8(d2_, d3_);
    
    // TODO: print combine
    uint16x8_t d0____ = vpadalq_u8(output, vcombine_u8(d0__, d1__));

    // todo data writeback
    vst1q_u16(dst, d0____);
    return 0;
}

extern "C" int update_bipolar_a2b2_half(uint8_t* src_a, uint8_t* src_b, uint16_t* dst, 
    int a_str1, int a_str0, int b_str0){
    // todo - data loading
    uint8x8_t a[8];

    uint8x8_t aa[8];
    for(int i = 0; i < 8; i++)
        aa[i] = vld1_u8(src_a + i*a_str0);
    uint8x8_t b0 = vld1_u8(src_b);
    uint8x8_t b1 = vld1_u8(src_b + b_str0);
    uint16x8_t output = vld1q_u16(dst);

    // from racket phase 1: Manually unrolling from example because vshl must be constant
    for(int i = 0; i < 8; i++) {
        uint8x8_t temp = vand_u8(aa[i], b0);
        a[i] = vcnt_u8(temp);
    }
    
    for(int i = 0; i < 8; i++) {
        uint8x8_t temp = vand_u8(aa[i], b1);
        temp = vcnt_u8(temp);
        temp = vshl_n_u8(temp, 1);
        a[i] = vadd_u8(a[i], temp);
    }

    //Load bitplane 1 of a
     for(int i = 0; i < 8; i++)
        aa[i] = vld1_u8(src_a + a_str1 + i*a_str0);

    for(int i = 0; i < 8; i++) {
        uint8x8_t temp = vand_u8(aa[i], b0);
        temp = vcnt_u8(temp);
        temp = vshl_n_u8(temp, 1);
        a[i] = vadd_u8(a[i], temp);
    }
    
    for(int i = 0; i < 8; i++) {
        uint8x8_t temp = vand_u8(aa[i], b1);
        temp = vcnt_u8(temp);
        temp = vshl_n_u8(temp, 2);
        a[i] = vadd_u8(a[i], temp);
    }


    // from racket phase 2
    uint8x8_t d0_ = vpadd_u8(a[0], a[1]);
    uint8x8_t d1_ = vpadd_u8(a[2], a[3]);
    uint8x8_t d2_ = vpadd_u8(a[4], a[5]);
    uint8x8_t d3_ = vpadd_u8(a[6], a[7]);

    uint16x8_t q0_ = vpaddlq_u8(vcombine_u8(d0_, d1_));
    uint16x8_t q1_ = vpaddlq_u8(vcombine_u8(d2_, d3_));
    
    uint16x4_t d0__  = vpadd_u16(vget_low_u16(q0_), vget_high_u16(q0_));
    uint16x4_t d1__  = vpadd_u16(vget_low_u16(q1_), vget_high_u16(q1_));

    // accumulate
    uint16x8_t d0___ = vaddq_u16(output, vcombine_u16(d0__, d1__));

    // todo data writebacks
    vst1q_u16(dst, d0___);
    return 0;
}

///// 8x16x1 microkernels
extern "C" int update_unipolar_a1b1(uint8_t* src_a, uint8_t* src_b, int16_t* dst, 
    int a_str1, int a_str0, int b_str0){
    // todo - data loading
    uint8x16_t aa[8];
    int8x16_t a[8];
    for(int i = 0; i < 8; i++)
        aa[i] = vld1q_u8(src_a + i*a_str0);
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

extern "C" int update_unipolar_a1b2(uint8_t* src_a, uint8_t* src_b, int16_t* dst, 
    int a_str1, int a_str0, int b_str0){
    // todo - data loading
    int8x16_t x[8];
    int16x8_t output = vld1q_s16(dst);

    uint8x16_t a[8];
    for(int i = 0; i < 8; i++)
        a[i] = vld1q_u8(src_a + i*a_str0);
    uint8x16_t b = vld1q_u8(src_b);
            

    // Part 1: elementwise ops, popcount, shifts, adding bitplanes
    for(int i = 0; i < 8; i++) {
        uint8x16_t temp = vandq_u8(a[i], b);
        // not b and aa
        uint8x16_t temp2 = vbicq_u8(b, a[i]);
        temp = vcntq_u8(temp);
        temp2 = vcntq_u8(temp2);
        x[i] = vsubq_s8(temp, temp2);
    }

    // B's second bitplane ooops forgot vshl has to get take a constant
    b = vld1q_u8(src_b + b_str0);
    for(int i = 0; i < 8; i++) {
        uint8x16_t temp = vandq_u8(a[i], b);
        // not b and aa
        uint8x16_t temp2 = vbicq_u8(b, a[i]);
        temp = vcntq_u8(temp);
        temp2 = vcntq_u8(temp2);
        int8x16_t temp3 = vsubq_s8(temp, temp2);
        temp3 = vshlq_n_s8(temp3, 1);
        x[i] = vaddq_s8(x[i], temp3);
    }
    
    // todo - naming the upper and lower half 
    int8x8_t d0 = vget_high_s8(x[0]);
    int8x8_t d1 = vget_low_s8(x[0]);
    int8x8_t d2 = vget_high_s8(x[1]);
    int8x8_t d3 = vget_low_s8(x[1]);
    int8x8_t d4 = vget_high_s8(x[2]);
    int8x8_t d5 = vget_low_s8(x[2]);
    int8x8_t d6 = vget_high_s8(x[3]);
    int8x8_t d7 = vget_low_s8(x[3]);
    int8x8_t d8 = vget_high_s8(x[4]);
    int8x8_t d9 = vget_low_s8(x[4]);
    int8x8_t d10 = vget_high_s8(x[5]);
    int8x8_t d11 = vget_low_s8(x[5]);
    int8x8_t d12 = vget_high_s8(x[6]);
    int8x8_t d13 = vget_low_s8(x[6]);
    int8x8_t d14 = vget_high_s8(x[7]);
    int8x8_t d15 = vget_low_s8(x[7]);


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

extern "C" int update_unipolar_a2b2(uint8_t* src_a, uint8_t* src_b, int16_t* dst, 
    int a_str1, int a_str0, int b_str0){
    // todo - data loading
    int8x16_t x[8];
    int16x8_t output = vld1q_s16(dst);

    uint8x16_t a[8];
    for(int i = 0; i < 8; i++)
        a[i] = vld1q_u8(src_a + i*a_str0);
    uint8x16_t b0 = vld1q_u8(src_b);
    uint8x16_t b1 = vld1q_u8(src_b + b_str0);
            

    // Part 1: elementwise ops, popcount, shifts, adding bitplanes
    for(int i = 0; i < 8; i++) {
        uint8x16_t temp = vandq_u8(a[i], b0);
        // not b and aa
        uint8x16_t temp2 = vbicq_u8(b0, a[i]);
        temp = vcntq_u8(temp);
        temp2 = vcntq_u8(temp2);
        x[i] = vsubq_s8(temp, temp2);
    }

    // B's second bitplane ooops forgot vshl has to get take a constant
    
    for(int i = 0; i < 8; i++) {
        uint8x16_t temp = vandq_u8(a[i], b1);
        // not b and aa
        uint8x16_t temp2 = vbicq_u8(b1, a[i]);
        temp = vcntq_u8(temp);
        temp2 = vcntq_u8(temp2);
        int8x16_t temp3 = vsubq_s8(temp, temp2);
        temp3 = vshlq_n_s8(temp3, 1);
        x[i] = vaddq_s8(x[i], temp3);
    }
    // Load next bitplane of a
    for(int i = 0; i < 8; i++)
        a[i] = vld1q_u8(src_a + a_str1 + i*a_str0);

    for(int i = 0; i < 8; i++) {
        uint8x16_t temp = vandq_u8(a[i], b0);
        // not b and aa
        uint8x16_t temp2 = vbicq_u8(b0, a[i]);
        temp = vcntq_u8(temp);
        temp2 = vcntq_u8(temp2);
        int8x16_t temp3 = vsubq_s8(temp, temp2);
        temp3 = vshlq_n_s8(temp3, 1);
        x[i] = vaddq_s8(x[i], temp3);
    }

    // B's second bitplane ooops forgot vshl has to get take a constant
    
    for(int i = 0; i < 8; i++) {
        uint8x16_t temp = vandq_u8(a[i], b1);
        // not b and aa
        uint8x16_t temp2 = vbicq_u8(b1, a[i]);
        temp = vcntq_u8(temp);
        temp2 = vcntq_u8(temp2);
        int8x16_t temp3 = vsubq_s8(temp, temp2);
        temp3 = vshlq_n_s8(temp3, 2);
        x[i] = vaddq_s8(x[i], temp3);
    }
    
    // todo - naming the upper and lower half 
    int8x8_t d0 = vget_high_s8(x[0]);
    int8x8_t d1 = vget_low_s8(x[0]);
    int8x8_t d2 = vget_high_s8(x[1]);
    int8x8_t d3 = vget_low_s8(x[1]);
    int8x8_t d4 = vget_high_s8(x[2]);
    int8x8_t d5 = vget_low_s8(x[2]);
    int8x8_t d6 = vget_high_s8(x[3]);
    int8x8_t d7 = vget_low_s8(x[3]);
    int8x8_t d8 = vget_high_s8(x[4]);
    int8x8_t d9 = vget_low_s8(x[4]);
    int8x8_t d10 = vget_high_s8(x[5]);
    int8x8_t d11 = vget_low_s8(x[5]);
    int8x8_t d12 = vget_high_s8(x[6]);
    int8x8_t d13 = vget_low_s8(x[6]);
    int8x8_t d14 = vget_high_s8(x[7]);
    int8x8_t d15 = vget_low_s8(x[7]);


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

extern "C" int update_bipolar_a1b1(uint8_t* src_a, uint8_t* src_b, uint16_t* dst,  
    int a_str1, int a_str0, int b_str0){

    // todo - data loading
    uint8x16_t a[8];
    for(int i = 0; i < 8; i++)
        a[i] = vld1q_u8(src_a + i*a_str0);
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
    // TODO: print combine
    uint16x8_t d0____ = vpadalq_u8(output, vcombine_u8(d0___, d1___));

    // todo data writeback
    vst1q_u16(dst, d0____);
    return 0;

}

extern "C" int update_bipolar_a1b2(uint8_t* src_a, uint8_t* src_b, uint16_t* dst, 
    int a_str1, int a_str0, int b_str0){
    // todo - data loading
    uint8x16_t x[8];
    uint16x8_t output = vld1q_u16(dst);

    // Zero out the register
    for (int i = 0; i < 8; i++) {
        x[i] = veorq_u8(x[i], x[i]);
    }

    uint8x16_t a[8];
    for(int i = 0; i < 8; i++)
        a[i] = vld1q_u8(src_a + a_str0*i);
    uint8x16_t b = vld1q_u8(src_b);
            

    // Part 1: elementwise ops, popcount, shifts, adding bitplanes
    for(int i = 0; i < 8; i++) {
        uint8x16_t temp = vandq_u8(a[i], b);
        x[i] = vcntq_u8(temp);
    }

    // B's second bitplane ooops forgot vshl has to get take a constant
    b = vld1q_u8(src_b + b_str0);
    for(int i = 0; i < 8; i++) {
        uint8x16_t temp = vandq_u8(a[i], b);
        temp = vcntq_u8(temp);
        temp = vshlq_n_u8(temp, 1);
        x[i] = vaddq_u8(x[i], temp);
    }

    // todo - naming the upper and lower half 
    uint8x8_t d0 = vget_high_u8(x[0]);
    uint8x8_t d1 = vget_low_u8(x[0]);
    uint8x8_t d2 = vget_high_u8(x[1]);
    uint8x8_t d3 = vget_low_u8(x[1]);
    uint8x8_t d4 = vget_high_u8(x[2]);
    uint8x8_t d5 = vget_low_u8(x[2]);
    uint8x8_t d6 = vget_high_u8(x[3]);
    uint8x8_t d7 = vget_low_u8(x[3]);
    uint8x8_t d8 = vget_high_u8(x[4]);
    uint8x8_t d9 = vget_low_u8(x[4]);
    uint8x8_t d10 = vget_high_u8(x[5]);
    uint8x8_t d11 = vget_low_u8(x[5]);
    uint8x8_t d12 = vget_high_u8(x[6]);
    uint8x8_t d13 = vget_low_u8(x[6]);
    uint8x8_t d14 = vget_high_u8(x[7]);
    uint8x8_t d15 = vget_low_u8(x[7]);


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
    // TODO: print combine
    uint16x8_t d0____ = vpadalq_u8(output, vcombine_u8(d0___, d1___));

    // todo data writeback
    vst1q_u16(dst, d0____);
    return 0;
}

extern "C" int update_bipolar_a2b2(uint8_t* src_a, uint8_t* src_b, uint16_t* dst, 
    int a_str1, int a_str0, int b_str0){
    // todo - data loading
    uint8x16_t x[8];

    uint8x16_t aa[8];
    for(int i = 0; i < 8; i++)
        aa[i] = vld1q_u8(src_a + i*a_str0);
    uint8x16_t b0 = vld1q_u8(src_b);
    uint8x16_t b1 = vld1q_u8(src_b + b_str0);
    uint16x8_t output = vld1q_u16(dst);

    // from racket phase 1: Manually unrolling from example because vshl must be constant
    for(int i = 0; i < 8; i++) {
        uint8x16_t temp = vandq_u8(aa[i], b0);
        x[i] = vcntq_u8(temp);
    }
    
    for(int i = 0; i < 8; i++) {
        uint8x16_t temp = vandq_u8(aa[i], b1);
        temp = vcntq_u8(temp);
        temp = vshlq_n_u8(temp, 1);
        x[i] = vaddq_u8(x[i], temp);
    }

    //Load bitplane 1 of a
     for(int i = 0; i < 8; i++)
        aa[i] = vld1q_u8(src_a + a_str1 + i*a_str0);

    for(int i = 0; i < 8; i++) {
        uint8x16_t temp = vandq_u8(aa[i], b0);
        temp = vcntq_u8(temp);
        temp = vshlq_n_u8(temp, 1);
        x[i] = vaddq_u8(x[i], temp);
    }
    
    for(int i = 0; i < 8; i++) {
        uint8x16_t temp = vandq_u8(aa[i], b1);
        temp = vcntq_u8(temp);
        temp = vshlq_n_u8(temp, 2);
        x[i] = vaddq_u8(x[i], temp);
    }

    // todo - naming the upper and lower half 
    uint8x8_t d0 = vget_high_u8(x[0]);
    uint8x8_t d1 = vget_low_u8(x[0]);
    uint8x8_t d2 = vget_high_u8(x[1]);
    uint8x8_t d3 = vget_low_u8(x[1]);
    uint8x8_t d4 = vget_high_u8(x[2]);
    uint8x8_t d5 = vget_low_u8(x[2]);
    uint8x8_t d6 = vget_high_u8(x[3]);
    uint8x8_t d7 = vget_low_u8(x[3]);
    uint8x8_t d8 = vget_high_u8(x[4]);
    uint8x8_t d9 = vget_low_u8(x[4]);
    uint8x8_t d10 = vget_high_u8(x[5]);
    uint8x8_t d11 = vget_low_u8(x[5]);
    uint8x8_t d12 = vget_high_u8(x[6]);
    uint8x8_t d13 = vget_low_u8(x[6]);
    uint8x8_t d14 = vget_high_u8(x[7]);
    uint8x8_t d15 = vget_low_u8(x[7]);

    // from racket phase 2
    uint8x8_t d0_ = vpadd_u8(d0, d1);
    uint8x8_t d1_ = vpadd_u8(d2, d3);
    uint8x8_t d2_ = vpadd_u8(d4, d5);
    uint8x8_t d3_ = vpadd_u8(d6, d7);
    uint8x8_t d4_ = vpadd_u8(d8, d9);
    uint8x8_t d5_ = vpadd_u8(d10, d11);
    uint8x8_t d6_ = vpadd_u8(d12, d13);
    uint8x8_t d7_ = vpadd_u8(d14, d15);

    uint16x8_t q0_ = vpaddlq_u8(vcombine_u8(d0_, d1_));
    uint16x8_t q1_ = vpaddlq_u8(vcombine_u8(d2_, d3_));
    uint16x8_t q2_ = vpaddlq_u8(vcombine_u8(d4_, d5_));
    uint16x8_t q3_ = vpaddlq_u8(vcombine_u8(d6_, d7_));
    
    uint16x4_t d0__  = vpadd_u16(vget_low_u16(q0_), vget_high_u16(q0_));
    uint16x4_t d1__  = vpadd_u16(vget_low_u16(q1_), vget_high_u16(q1_));
    uint16x4_t d2__  = vpadd_u16(vget_low_u16(q2_), vget_high_u16(q2_));
    uint16x4_t d3__  = vpadd_u16(vget_low_u16(q3_), vget_high_u16(q3_));

    uint16x4_t d0___  = vpadd_u16(d0__, d1__);
    uint16x4_t d1___  = vpadd_u16(d2__, d3__);

    // accumulate
    uint16x8_t d0____ = vaddq_u16(output, vcombine_u16(d0___, d1___));

    // todo data writebacks
    vst1q_u16(dst, d0____);
    return 0;
}
