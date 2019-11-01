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
extern "C" int update_unipolar_a1b1_half(int8_t* src_a, int8_t* src_b, int16_t* dst, int a_str1, int a_str0, int b_str0){ 
    // Data loading
    int8x8_t a[8];
    for(int i = 0; i < 4; i++) {
        int8x16_t aa0 = vld1q_s8(src_a + i*2*a_str0);
        a[2*i] = vget_low_s8(aa0);
        a[2*i + 1] = vget_high_s8(aa0);
    }
    int8x8_t b = vld1_s8(src_b);
    int16x8_t output = vld1q_s16(dst);

    int8x8_t x002_ = vand_s8(a[0], b);
    int8x8_t x003o = vcnt_s8(x002_);
    int8x8_t x004_ = vbic_s8(b, a[0]);
    int8x8_t x005o = vcnt_s8(x004_);
    int8x8_t x0 = vsub_s8(x003o, x005o);
    int8x8_t x102_ = vand_s8(a[1], b);
    int8x8_t x103o = vcnt_s8(x102_);
    int8x8_t x104_ = vbic_s8(b, a[1]);
    int8x8_t x105o = vcnt_s8(x104_);
    int8x8_t x1 = vsub_s8(x103o, x105o);
    int8x8_t x202_ = vand_s8(a[2], b);
    int8x8_t x203o = vcnt_s8(x202_);
    int8x8_t x204_ = vbic_s8(b, a[2]);
    int8x8_t x205o = vcnt_s8(x204_);
    int8x8_t x2 = vsub_s8(x203o, x205o);
    int8x8_t x302_ = vand_s8(a[3], b);
    int8x8_t x303o = vcnt_s8(x302_);
    int8x8_t x304_ = vbic_s8(b, a[3]);
    int8x8_t x305o = vcnt_s8(x304_);
    int8x8_t x3 = vsub_s8(x303o, x305o);
    int8x8_t x402_ = vand_s8(a[4], b);
    int8x8_t x403o = vcnt_s8(x402_);
    int8x8_t x404_ = vbic_s8(b, a[4]);
    int8x8_t x405o = vcnt_s8(x404_);
    int8x8_t x4 = vsub_s8(x403o, x405o);
    int8x8_t x502_ = vand_s8(a[5], b);
    int8x8_t x503o = vcnt_s8(x502_);
    int8x8_t x504_ = vbic_s8(b, a[5]);
    int8x8_t x505o = vcnt_s8(x504_);
    int8x8_t x5 = vsub_s8(x503o, x505o);
    int8x8_t x602_ = vand_s8(a[6], b);
    int8x8_t x603o = vcnt_s8(x602_);
    int8x8_t x604_ = vbic_s8(b, a[6]);
    int8x8_t x605o = vcnt_s8(x604_);
    int8x8_t x6 = vsub_s8(x603o, x605o);
    int8x8_t x702_ = vand_s8(a[7], b);
    int8x8_t x703o = vcnt_s8(x702_);
    int8x8_t x704_ = vbic_s8(b, a[7]);
    int8x8_t x705o = vcnt_s8(x704_);
    int8x8_t x7 = vsub_s8(x703o, x705o);
    int8x8_t x0_ = vpadd_s8(x0, x1);
    int8x8_t x1o = vpadd_s8(x2, x3);
    int8x8_t x2_ = vpadd_s8(x4, x5);
    int8x8_t x3o = vpadd_s8(x6, x7);
    int8x8_t x0__ = vpadd_s8(x0_, x1o);
    int8x8_t x1oo = vpadd_s8(x2_, x3o);
    int8x16_t out = vpadalq_s8(output, vcombine_s8(x0__, x1oo));

     // Not synthesized
    vst1q_s16(dst, out);
    return 0;
}

extern "C" int update_unipolar_a1b2_half(int8_t* src_a, int8_t* src_b, int16_t* dst, int a_str1, int a_str0, int b_str0){
    // manual data load
    int8x8_t a0[8];
    for(int i = 0; i < 4; i++) {
        int8x16_t aa0 = vld1q_s8(src_a + i*2*a_str0);
        a0[2*i] = vget_low_s8(aa0);
        a0[2*i + 1] = vget_high_s8(aa0);
    }
    int8x8_t b0 = vld1_s8(src_b);
    int8x8_t b1 = vld1_s8(src_b + b_str0);
    int16x8_t output = vld1q_s16(dst);

    int8x8_t x003o = vand_s8(a0[0], b0);
    int8x8_t x004_ = vbic_s8(b0, a0[0]);
    int8x8_t x005o = vcnt_s8(x003o);
    int8x8_t x006_ = vcnt_s8(x004_);
    int8x8_t x0010_ = vsub_s8(x005o, x006_);
    int8x8_t x003oo = vand_s8(a0[0], b1);
    int8x8_t x004__ = vbic_s8(b1, a0[0]);
    int8x8_t x005oo = vcnt_s8(x003oo);
    int8x8_t x006__ = vcnt_s8(x004__);
    int8x8_t x007o = vsub_s8(x005oo, x006__);
    int8x8_t x0011o = vshl_n_s8(x007o, 1);
    int8x8_t x0 = vadd_s8(x0010_, x0011o);

    int8x8_t x103o = vand_s8(a0[1], b0);
    int8x8_t x104_ = vbic_s8(b0, a0[1]);
    int8x8_t x105o = vcnt_s8(x103o);
    int8x8_t x106_ = vcnt_s8(x104_);
    int8x8_t x1010_ = vsub_s8(x105o, x106_);
    int8x8_t x103oo = vand_s8(a0[1], b1);
    int8x8_t x104__ = vbic_s8(b1, a0[1]);
    int8x8_t x105oo = vcnt_s8(x103oo);
    int8x8_t x106__ = vcnt_s8(x104__);
    int8x8_t x107o = vsub_s8(x105oo, x106__);
    int8x8_t x1011o = vshl_n_s8(x107o, 1);

    int8x8_t x1 = vadd_s8(x1010_, x1011o);
    int8x8_t x203o = vand_s8(a0[2], b0);
    int8x8_t x204_ = vbic_s8(b0, a0[2]);
    int8x8_t x205o = vcnt_s8(x203o);
    int8x8_t x206_ = vcnt_s8(x204_);
    int8x8_t x2010_ = vsub_s8(x205o, x206_);
    int8x8_t x203oo = vand_s8(a0[2], b1);
    int8x8_t x204__ = vbic_s8(b1, a0[2]);
    int8x8_t x205oo = vcnt_s8(x203oo);
    int8x8_t x206__ = vcnt_s8(x204__);
    int8x8_t x207o = vsub_s8(x205oo, x206__);
    int8x8_t x2011o = vshl_n_s8(x207o, 1);

    int8x8_t x2 = vadd_s8(x2010_, x2011o);
    int8x8_t x303o = vand_s8(a0[3], b0);
    int8x8_t x304_ = vbic_s8(b0, a0[3]);
    int8x8_t x305o = vcnt_s8(x303o);
    int8x8_t x306_ = vcnt_s8(x304_);
    int8x8_t x3010_ = vsub_s8(x305o, x306_);
    int8x8_t x303oo = vand_s8(a0[3], b1);
    int8x8_t x304__ = vbic_s8(b1, a0[3]);
    int8x8_t x305oo = vcnt_s8(x303oo);
    int8x8_t x306__ = vcnt_s8(x304__);
    int8x8_t x307o = vsub_s8(x305oo, x306__);
    int8x8_t x3011o = vshl_n_s8(x307o, 1);

    int8x8_t x3 = vadd_s8(x3010_, x3011o);
    int8x8_t x403o = vand_s8(a0[4], b0);
    int8x8_t x404_ = vbic_s8(b0, a0[4]);
    int8x8_t x405o = vcnt_s8(x403o);
    int8x8_t x406_ = vcnt_s8(x404_);
    int8x8_t x4010_ = vsub_s8(x405o, x406_);
    int8x8_t x403oo = vand_s8(a0[4], b1);
    int8x8_t x404__ = vbic_s8(b1, a0[4]);
    int8x8_t x405oo = vcnt_s8(x403oo);
    int8x8_t x406__ = vcnt_s8(x404__);
    int8x8_t x407o = vsub_s8(x405oo, x406__);
    int8x8_t x4011o = vshl_n_s8(x407o, 1);

    int8x8_t x4 = vadd_s8(x4010_, x4011o);
    int8x8_t x503o = vand_s8(a0[5], b0);
    int8x8_t x504_ = vbic_s8(b0, a0[5]);
    int8x8_t x505o = vcnt_s8(x503o);
    int8x8_t x506_ = vcnt_s8(x504_);
    int8x8_t x5010_ = vsub_s8(x505o, x506_);
    int8x8_t x503oo = vand_s8(a0[5], b1);
    int8x8_t x504__ = vbic_s8(b1, a0[5]);
    int8x8_t x505oo = vcnt_s8(x503oo);
    int8x8_t x506__ = vcnt_s8(x504__);
    int8x8_t x507o = vsub_s8(x505oo, x506__);
    int8x8_t x5011o = vshl_n_s8(x507o, 1);

    int8x8_t x5 = vadd_s8(x5010_, x5011o);
    int8x8_t x603o = vand_s8(a0[6], b0);
    int8x8_t x604_ = vbic_s8(b0, a0[6]);
    int8x8_t x605o = vcnt_s8(x603o);
    int8x8_t x606_ = vcnt_s8(x604_);
    int8x8_t x6010_ = vsub_s8(x605o, x606_);
    int8x8_t x603oo = vand_s8(a0[6], b1);
    int8x8_t x604__ = vbic_s8(b1, a0[6]);
    int8x8_t x605oo = vcnt_s8(x603oo);
    int8x8_t x606__ = vcnt_s8(x604__);
    int8x8_t x607o = vsub_s8(x605oo, x606__);
    int8x8_t x6011o = vshl_n_s8(x607o, 1);
    int8x8_t x6 = vadd_s8(x6010_, x6011o);

    int8x8_t x703o = vand_s8(a0[7], b0);
    int8x8_t x704_ = vbic_s8(b0, a0[7]);
    int8x8_t x705o = vcnt_s8(x703o);
    int8x8_t x706_ = vcnt_s8(x704_);
    int8x8_t x7010_ = vsub_s8(x705o, x706_);
    int8x8_t x703oo = vand_s8(a0[7], b1);
    int8x8_t x704__ = vbic_s8(b1, a0[7]);
    int8x8_t x705oo = vcnt_s8(x703oo);
    int8x8_t x706__ = vcnt_s8(x704__);
    int8x8_t x707o = vsub_s8(x705oo, x706__);
    int8x8_t x7011o = vshl_n_s8(x707o, 1);
    int8x8_t x7 = vadd_s8(x7010_, x7011o);

    int8x8_t x0_ = vpadd_s8(x0, x1);
    int8x8_t x1o = vpadd_s8(x2, x3);
    int8x8_t x2_ = vpadd_s8(x4, x5);
    int8x8_t x3o = vpadd_s8(x6, x7);
    int8x8_t x0__ = vpadd_s8(x0_, x1o);
    int8x8_t x1oo = vpadd_s8(x2_, x3o);
    int8x16_t out = vpadalq_s8(output, vcombine_s8(x0__, x1oo));

    // manual store back
    vst1q_s16(dst, out);
    return 0;
}

extern "C" int update_unipolar_a2b1_half(int8_t* src_a, int8_t* src_b, int16_t* dst, 
    int a_str1, int a_str0, int b_str0){
    // Data loading
    int8x8_t a0[8];
    int8x8_t a1[8];
    int8x8_t b0 = vld1_s8(src_b);
    int16x8_t output = vld1q_s16(dst);
    for(int i = 0; i < 8; i++) {
        a0[i] = vld1_s8(src_a + i*a_str0);
        a1[i] = vld1_s8(src_a + a_str1 + 8*a_str0);
    }
    
    int8x8_t x003o = vand_s8(a0[0], b0);
    int8x8_t x004_ = vbic_s8(b0, a0[0]);
    int8x8_t x005o = vcnt_s8(x003o);
    int8x8_t x006_ = vcnt_s8(x004_);
    int8x8_t x0010_ = vsub_s8(x005o, x006_);
    int8x8_t x003oo = vand_s8(a1[0], b0);
    int8x8_t x004__ = vbic_s8(b0, a1[0]);
    int8x8_t x005oo = vcnt_s8(x003oo);
    int8x8_t x006__ = vcnt_s8(x004__);
    int8x8_t x007o = vsub_s8(x005oo, x006__);
    int8x8_t x0011o = vadd_s8(x007o, x007o);
    int8x8_t y0 = vadd_s8(x0010_, x0011o);
    int8x8_t x103o = vand_s8(a0[1], b0);
    int8x8_t x104_ = vbic_s8(b0, a0[1]);
    int8x8_t x105o = vcnt_s8(x103o);
    int8x8_t x106_ = vcnt_s8(x104_);
    int8x8_t x1010_ = vsub_s8(x105o, x106_);
    int8x8_t x103oo = vand_s8(a1[1], b0);
    int8x8_t x104__ = vbic_s8(b0, a1[1]);
    int8x8_t x105oo = vcnt_s8(x103oo);
    int8x8_t x106__ = vcnt_s8(x104__);
    int8x8_t x107o = vsub_s8(x105oo, x106__);
    int8x8_t x1011o = vadd_s8(x107o, x107o);
    int8x8_t y1 = vadd_s8(x1010_, x1011o);
    int8x8_t x203o = vand_s8(a0[2], b0);
    int8x8_t x204_ = vbic_s8(b0, a0[2]);
    int8x8_t x205o = vcnt_s8(x203o);
    int8x8_t x206_ = vcnt_s8(x204_);
    int8x8_t x2010_ = vsub_s8(x205o, x206_);
    int8x8_t x203oo = vand_s8(a1[2], b0);
    int8x8_t x204__ = vbic_s8(b0, a1[2]);
    int8x8_t x205oo = vcnt_s8(x203oo);
    int8x8_t x206__ = vcnt_s8(x204__);
    int8x8_t x207o = vsub_s8(x205oo, x206__);
    int8x8_t x2011o = vadd_s8(x207o, x207o);
    int8x8_t y2 = vadd_s8(x2010_, x2011o);
    int8x8_t x303o = vand_s8(a0[3], b0);
    int8x8_t x304_ = vbic_s8(b0, a0[3]);
    int8x8_t x305o = vcnt_s8(x303o);
    int8x8_t x306_ = vcnt_s8(x304_);
    int8x8_t x3010_ = vsub_s8(x305o, x306_);
    int8x8_t x303oo = vand_s8(a1[3], b0);
    int8x8_t x304__ = vbic_s8(b0, a1[3]);
    int8x8_t x305oo = vcnt_s8(x303oo);
    int8x8_t x306__ = vcnt_s8(x304__);
    int8x8_t x307o = vsub_s8(x305oo, x306__);
    int8x8_t x3011o = vadd_s8(x307o, x307o);
    int8x8_t y3 = vadd_s8(x3010_, x3011o);
    int8x8_t x403o = vand_s8(a0[4], b0);
    int8x8_t x404_ = vbic_s8(b0, a0[4]);
    int8x8_t x405o = vcnt_s8(x403o);
    int8x8_t x406_ = vcnt_s8(x404_);
    int8x8_t x4010_ = vsub_s8(x405o, x406_);
    int8x8_t x403oo = vand_s8(a1[4], b0);
    int8x8_t x404__ = vbic_s8(b0, a1[4]);
    int8x8_t x405oo = vcnt_s8(x403oo);
    int8x8_t x406__ = vcnt_s8(x404__);
    int8x8_t x407o = vsub_s8(x405oo, x406__);
    int8x8_t x4011o = vadd_s8(x407o, x407o);
    int8x8_t y4 = vadd_s8(x4010_, x4011o);
    int8x8_t x503o = vand_s8(a0[5], b0);
    int8x8_t x504_ = vbic_s8(b0, a0[5]);
    int8x8_t x505o = vcnt_s8(x503o);
    int8x8_t x506_ = vcnt_s8(x504_);
    int8x8_t x5010_ = vsub_s8(x505o, x506_);
    int8x8_t x503oo = vand_s8(a1[5], b0);
    int8x8_t x504__ = vbic_s8(b0, a1[5]);
    int8x8_t x505oo = vcnt_s8(x503oo);
    int8x8_t x506__ = vcnt_s8(x504__);
    int8x8_t x507o = vsub_s8(x505oo, x506__);
    int8x8_t x5011o = vadd_s8(x507o, x507o);
    int8x8_t y5 = vadd_s8(x5010_, x5011o);
    int8x8_t x603o = vand_s8(a0[6], b0);
    int8x8_t x604_ = vbic_s8(b0, a0[6]);
    int8x8_t x605o = vcnt_s8(x603o);
    int8x8_t x606_ = vcnt_s8(x604_);
    int8x8_t x6010_ = vsub_s8(x605o, x606_);
    int8x8_t x603oo = vand_s8(a1[6], b0);
    int8x8_t x604__ = vbic_s8(b0, a1[6]);
    int8x8_t x605oo = vcnt_s8(x603oo);
    int8x8_t x606__ = vcnt_s8(x604__);
    int8x8_t x607o = vsub_s8(x605oo, x606__);
    int8x8_t x6011o = vadd_s8(x607o, x607o);
    int8x8_t y6 = vadd_s8(x6010_, x6011o);
    int8x8_t x703o = vand_s8(a0[7], b0);
    int8x8_t x704_ = vbic_s8(b0, a0[7]);
    int8x8_t x705o = vcnt_s8(x703o);
    int8x8_t x706_ = vcnt_s8(x704_);
    int8x8_t x7010_ = vsub_s8(x705o, x706_);
    int8x8_t x703oo = vand_s8(a1[7], b0);
    int8x8_t x704__ = vbic_s8(b0, a1[7]);
    int8x8_t x705oo = vcnt_s8(x703oo);
    int8x8_t x706__ = vcnt_s8(x704__);
    int8x8_t x707o = vsub_s8(x705oo, x706__);
    int8x8_t x7011o = vadd_s8(x707o, x707o);
    int8x8_t y7 = vadd_s8(x7010_, x7011o);
    int8x8_t y0_ = vpadd_s8(y0, y1);
    int8x8_t y1o = vpadd_s8(y2, y3);
    int8x8_t y2_ = vpadd_s8(y4, y5);
    int8x8_t y3o = vpadd_s8(y6, y7);
    int8x8_t y0__ = vpadd_s8(y0_, y1o);
    int8x8_t y1oo = vpadd_s8(y2_, y3o);
    int8x16_t out = vpadalq_s8(output, vcombine_s8(y0__, y1oo));

    // Write back
    vst1q_s16(dst, out);
    return 0;
}

extern "C" int update_unipolar_a1b3_half(int8_t* src_a, int8_t* src_b, int16_t* dst, 
    int a_str1, int a_str0, int b_str0){
    // Data loading
    int8x8_t a0[8];
    for(int i = 0; i < 8; i++)
        a0[i] = vld1_s8(src_a + i*a_str0);
    int8x8_t b0 = vld1_s8(src_b);
    int8x8_t b1 = vld1_s8(src_b + b_str0);
    int8x8_t b2 = vld1_s8(src_b + 2 * b_str0);
    int16x8_t output = vld1q_s16(dst);

    int8x8_t x004_ = vand_s8(a0[0], b0);
    int8x8_t x005o = vbic_s8(b0, a0[0]);
    int8x8_t x006_ = vcnt_s8(x004_);
    int8x8_t x007o = vcnt_s8(x005o);
    int8x8_t x0010_ = vsub_s8(x006_, x007o);
    int8x8_t x004__ = vand_s8(a0[0], b1);
    int8x8_t x005oo = vcnt_s8(x004__);
    int8x8_t x006__ = vbic_s8(b1, a0[0]);
    int8x8_t x007oo = vcnt_s8(x006__);
    int8x8_t x008_ = vsub_s8(x005oo, x007oo);
    int8x8_t x0011o = vshl_n_s8(x008_, 1);
    int8x8_t x004___ = vbic_s8(b2, a0[0]);
    int8x8_t x005ooo = vand_s8(a0[0], b2);
    int8x8_t x006___ = vcnt_s8(x005ooo);
    int8x8_t x007ooo = vcnt_s8(x004___);
    int8x8_t x008__ = vsub_s8(x006___, x007ooo);
    int8x8_t x0012_ = vshl_n_s8(x008__, 2);
    int8x8_t x0010__ = vadd_s8(x0010_, x0011o);
    int8x8_t y0 = vadd_s8(x0010__, x0012_);
    int8x8_t x104_ = vand_s8(a0[1], b0);
    int8x8_t x105o = vbic_s8(b0, a0[1]);
    int8x8_t x106_ = vcnt_s8(x104_);
    int8x8_t x107o = vcnt_s8(x105o);
    int8x8_t x1010_ = vsub_s8(x106_, x107o);
    int8x8_t x104__ = vand_s8(a0[1], b1);
    int8x8_t x105oo = vcnt_s8(x104__);
    int8x8_t x106__ = vbic_s8(b1, a0[1]);
    int8x8_t x107oo = vcnt_s8(x106__);
    int8x8_t x108_ = vsub_s8(x105oo, x107oo);
    int8x8_t x1011o = vshl_n_s8(x108_, 1);
    int8x8_t x104___ = vbic_s8(b2, a0[1]);
    int8x8_t x105ooo = vand_s8(a0[1], b2);
    int8x8_t x106___ = vcnt_s8(x105ooo);
    int8x8_t x107ooo = vcnt_s8(x104___);
    int8x8_t x108__ = vsub_s8(x106___, x107ooo);
    int8x8_t x1012_ = vshl_n_s8(x108__, 2);
    int8x8_t x1010__ = vadd_s8(x1010_, x1011o);
    int8x8_t y1 = vadd_s8(x1010__, x1012_);
    int8x8_t x204_ = vand_s8(a0[2], b0);
    int8x8_t x205o = vbic_s8(b0, a0[2]);
    int8x8_t x206_ = vcnt_s8(x204_);
    int8x8_t x207o = vcnt_s8(x205o);
    int8x8_t x2010_ = vsub_s8(x206_, x207o);
    int8x8_t x204__ = vand_s8(a0[2], b1);
    int8x8_t x205oo = vcnt_s8(x204__);
    int8x8_t x206__ = vbic_s8(b1, a0[2]);
    int8x8_t x207oo = vcnt_s8(x206__);
    int8x8_t x208_ = vsub_s8(x205oo, x207oo);
    int8x8_t x2011o = vshl_n_s8(x208_, 1);
    int8x8_t x204___ = vbic_s8(b2, a0[2]);
    int8x8_t x205ooo = vand_s8(a0[2], b2);
    int8x8_t x206___ = vcnt_s8(x205ooo);
    int8x8_t x207ooo = vcnt_s8(x204___);
    int8x8_t x208__ = vsub_s8(x206___, x207ooo);
    int8x8_t x2012_ = vshl_n_s8(x208__, 2);
    int8x8_t x2010__ = vadd_s8(x2010_, x2011o);
    int8x8_t y2 = vadd_s8(x2010__, x2012_);
    int8x8_t x304_ = vand_s8(a0[3], b0);
    int8x8_t x305o = vbic_s8(b0, a0[3]);
    int8x8_t x306_ = vcnt_s8(x304_);
    int8x8_t x307o = vcnt_s8(x305o);
    int8x8_t x3010_ = vsub_s8(x306_, x307o);
    int8x8_t x304__ = vand_s8(a0[3], b1);
    int8x8_t x305oo = vcnt_s8(x304__);
    int8x8_t x306__ = vbic_s8(b1, a0[3]);
    int8x8_t x307oo = vcnt_s8(x306__);
    int8x8_t x308_ = vsub_s8(x305oo, x307oo);
    int8x8_t x3011o = vshl_n_s8(x308_, 1);
    int8x8_t x304___ = vbic_s8(b2, a0[3]);
    int8x8_t x305ooo = vand_s8(a0[3], b2);
    int8x8_t x306___ = vcnt_s8(x305ooo);
    int8x8_t x307ooo = vcnt_s8(x304___);
    int8x8_t x308__ = vsub_s8(x306___, x307ooo);
    int8x8_t x3012_ = vshl_n_s8(x308__, 2);
    int8x8_t x3010__ = vadd_s8(x3010_, x3011o);
    int8x8_t y3 = vadd_s8(x3010__, x3012_);
    int8x8_t x404_ = vand_s8(a0[4], b0);
    int8x8_t x405o = vbic_s8(b0, a0[4]);
    int8x8_t x406_ = vcnt_s8(x404_);
    int8x8_t x407o = vcnt_s8(x405o);
    int8x8_t x4010_ = vsub_s8(x406_, x407o);
    int8x8_t x404__ = vand_s8(a0[4], b1);
    int8x8_t x405oo = vcnt_s8(x404__);
    int8x8_t x406__ = vbic_s8(b1, a0[4]);
    int8x8_t x407oo = vcnt_s8(x406__);
    int8x8_t x408_ = vsub_s8(x405oo, x407oo);
    int8x8_t x4011o = vshl_n_s8(x408_, 1);
    int8x8_t x404___ = vbic_s8(b2, a0[4]);
    int8x8_t x405ooo = vand_s8(a0[4], b2);
    int8x8_t x406___ = vcnt_s8(x405ooo);
    int8x8_t x407ooo = vcnt_s8(x404___);
    int8x8_t x408__ = vsub_s8(x406___, x407ooo);
    int8x8_t x4012_ = vshl_n_s8(x408__, 2);
    int8x8_t x4010__ = vadd_s8(x4010_, x4011o);
    int8x8_t y4 = vadd_s8(x4010__, x4012_);
    int8x8_t x504_ = vand_s8(a0[5], b0);
    int8x8_t x505o = vbic_s8(b0, a0[5]);
    int8x8_t x506_ = vcnt_s8(x504_);
    int8x8_t x507o = vcnt_s8(x505o);
    int8x8_t x5010_ = vsub_s8(x506_, x507o);
    int8x8_t x504__ = vand_s8(a0[5], b1);
    int8x8_t x505oo = vcnt_s8(x504__);
    int8x8_t x506__ = vbic_s8(b1, a0[5]);
    int8x8_t x507oo = vcnt_s8(x506__);
    int8x8_t x508_ = vsub_s8(x505oo, x507oo);
    int8x8_t x5011o = vshl_n_s8(x508_, 1);
    int8x8_t x504___ = vbic_s8(b2, a0[5]);
    int8x8_t x505ooo = vand_s8(a0[5], b2);
    int8x8_t x506___ = vcnt_s8(x505ooo);
    int8x8_t x507ooo = vcnt_s8(x504___);
    int8x8_t x508__ = vsub_s8(x506___, x507ooo);
    int8x8_t x5012_ = vshl_n_s8(x508__, 2);
    int8x8_t x5010__ = vadd_s8(x5010_, x5011o);
    int8x8_t y5 = vadd_s8(x5010__, x5012_);
    int8x8_t x604_ = vand_s8(a0[6], b0);
    int8x8_t x605o = vbic_s8(b0, a0[6]);
    int8x8_t x606_ = vcnt_s8(x604_);
    int8x8_t x607o = vcnt_s8(x605o);
    int8x8_t x6010_ = vsub_s8(x606_, x607o);
    int8x8_t x604__ = vand_s8(a0[6], b1);
    int8x8_t x605oo = vcnt_s8(x604__);
    int8x8_t x606__ = vbic_s8(b1, a0[6]);
    int8x8_t x607oo = vcnt_s8(x606__);
    int8x8_t x608_ = vsub_s8(x605oo, x607oo);
    int8x8_t x6011o = vshl_n_s8(x608_, 1);
    int8x8_t x604___ = vbic_s8(b2, a0[6]);
    int8x8_t x605ooo = vand_s8(a0[6], b2);
    int8x8_t x606___ = vcnt_s8(x605ooo);
    int8x8_t x607ooo = vcnt_s8(x604___);
    int8x8_t x608__ = vsub_s8(x606___, x607ooo);
    int8x8_t x6012_ = vshl_n_s8(x608__, 2);
    int8x8_t x6010__ = vadd_s8(x6010_, x6011o);
    int8x8_t y6 = vadd_s8(x6010__, x6012_);
    int8x8_t x704_ = vand_s8(a0[7], b0);
    int8x8_t x705o = vbic_s8(b0, a0[7]);
    int8x8_t x706_ = vcnt_s8(x704_);
    int8x8_t x707o = vcnt_s8(x705o);
    int8x8_t x7010_ = vsub_s8(x706_, x707o);
    int8x8_t x704__ = vand_s8(a0[7], b1);
    int8x8_t x705oo = vcnt_s8(x704__);
    int8x8_t x706__ = vbic_s8(b1, a0[7]);
    int8x8_t x707oo = vcnt_s8(x706__);
    int8x8_t x708_ = vsub_s8(x705oo, x707oo);
    int8x8_t x7011o = vshl_n_s8(x708_, 1);
    int8x8_t x704___ = vbic_s8(b2, a0[7]);
    int8x8_t x705ooo = vand_s8(a0[7], b2);
    int8x8_t x706___ = vcnt_s8(x705ooo);
    int8x8_t x707ooo = vcnt_s8(x704___);
    int8x8_t x708__ = vsub_s8(x706___, x707ooo);
    int8x8_t x7012_ = vshl_n_s8(x708__, 2);
    int8x8_t x7010__ = vadd_s8(x7010_, x7011o);
    int8x8_t y7 = vadd_s8(x7010__, x7012_);
    int8x8_t y0_ = vpadd_s8(y0, y1);
    int8x8_t y1o = vpadd_s8(y2, y3);
    int8x8_t y2_ = vpadd_s8(y4, y5);
    int8x8_t y3o = vpadd_s8(y6, y7);
    int8x16_t y0__ = vpaddlq_s8(vcombine_s8(y0_, y1o));
    int8x16_t y2__ = vpaddlq_s8(vcombine_s8(y2_, y3o));
    int16x4_t y0___ = vpadd_s16(vget_low_s16(y0__), vget_high_s16(y0__));
    int16x4_t y0__o = vpadd_s16(vget_low_s16(y2__), vget_high_s16(y2__));
    int16x8_t out = vaddq_s16(vcombine_s16(y0___, y0__o), output);
    
    // Writeback
    vst1q_s16(dst, out);
    return 0;
}

extern "C" int update_unipolar_a3b1_half(int8_t* src_a, int8_t* src_b, int16_t* dst, 
    int a_str1, int a_str0, int b_str0){
    // Data loading
    int8x8_t a0[8];
    int8x8_t a1[8];
    int8x8_t a2[8];
    int8x8_t b0 = vld1_s8(src_b);
    int16x8_t output = vld1q_s16(dst);
    for(int i = 0; i < 8; i++) {
        a0[i] = vld1_s8(src_a + i*a_str0);
        a1[i] = vld1_s8(src_a + a_str1 + i*a_str0);
        a2[i] = vld1_s8(src_a + 2*a_str1 + i*a_str0);
    }

    int8x8_t x004_ = vand_s8(a0[0], b0);
    int8x8_t x005o = vcnt_s8(x004_);
    int8x8_t x006_ = vbic_s8(b0, a0[0]);
    int8x8_t x007o = vcnt_s8(x006_);
    int8x8_t x0010_ = vsub_s8(x005o, x007o);
    int8x8_t x004__ = vand_s8(a1[0], b0);
    int8x8_t x005oo = vbic_s8(b0, a1[0]);
    int8x8_t x006__ = vcnt_s8(x004__);
    int8x8_t x007oo = vcnt_s8(x005oo);
    int8x8_t x008_ = vsub_s8(x006__, x007oo);
    int8x8_t x0011o = vadd_s8(x008_, x008_);
    int8x8_t x004___ = vand_s8(a2[0], b0);
    int8x8_t x005ooo = vbic_s8(b0, a2[0]);
    int8x8_t x006___ = vcnt_s8(x004___);
    int8x8_t x007ooo = vcnt_s8(x005ooo);
    int8x8_t x008__ = vsub_s8(x006___, x007ooo);
    int8x8_t x0012_ = vshl_n_s8(x008__, 2);
    int8x8_t x0010__ = vadd_s8(x0010_, x0011o);
    int8x8_t y0 = vadd_s8(x0010__, x0012_);
    int8x8_t x104_ = vand_s8(a0[1], b0);
    int8x8_t x105o = vcnt_s8(x104_);
    int8x8_t x106_ = vbic_s8(b0, a0[1]);
    int8x8_t x107o = vcnt_s8(x106_);
    int8x8_t x1010_ = vsub_s8(x105o, x107o);
    int8x8_t x104__ = vand_s8(a1[1], b0);
    int8x8_t x105oo = vbic_s8(b0, a1[1]);
    int8x8_t x106__ = vcnt_s8(x104__);
    int8x8_t x107oo = vcnt_s8(x105oo);
    int8x8_t x108_ = vsub_s8(x106__, x107oo);
    int8x8_t x1011o = vadd_s8(x108_, x108_);
    int8x8_t x104___ = vand_s8(a2[1], b0);
    int8x8_t x105ooo = vbic_s8(b0, a2[1]);
    int8x8_t x106___ = vcnt_s8(x104___);
    int8x8_t x107ooo = vcnt_s8(x105ooo);
    int8x8_t x108__ = vsub_s8(x106___, x107ooo);
    int8x8_t x1012_ = vshl_n_s8(x108__, 2);
    int8x8_t x1010__ = vadd_s8(x1010_, x1011o);
    int8x8_t y1 = vadd_s8(x1010__, x1012_);
    int8x8_t x204_ = vand_s8(a0[2], b0);
    int8x8_t x205o = vcnt_s8(x204_);
    int8x8_t x206_ = vbic_s8(b0, a0[2]);
    int8x8_t x207o = vcnt_s8(x206_);
    int8x8_t x2010_ = vsub_s8(x205o, x207o);
    int8x8_t x204__ = vand_s8(a1[2], b0);
    int8x8_t x205oo = vbic_s8(b0, a1[2]);
    int8x8_t x206__ = vcnt_s8(x204__);
    int8x8_t x207oo = vcnt_s8(x205oo);
    int8x8_t x208_ = vsub_s8(x206__, x207oo);
    int8x8_t x2011o = vadd_s8(x208_, x208_);
    int8x8_t x204___ = vand_s8(a2[2], b0);
    int8x8_t x205ooo = vbic_s8(b0, a2[2]);
    int8x8_t x206___ = vcnt_s8(x204___);
    int8x8_t x207ooo = vcnt_s8(x205ooo);
    int8x8_t x208__ = vsub_s8(x206___, x207ooo);
    int8x8_t x2012_ = vshl_n_s8(x208__, 2);
    int8x8_t x2010__ = vadd_s8(x2010_, x2011o);
    int8x8_t y2 = vadd_s8(x2010__, x2012_);
    int8x8_t x304_ = vand_s8(a0[3], b0);
    int8x8_t x305o = vcnt_s8(x304_);
    int8x8_t x306_ = vbic_s8(b0, a0[3]);
    int8x8_t x307o = vcnt_s8(x306_);
    int8x8_t x3010_ = vsub_s8(x305o, x307o);
    int8x8_t x304__ = vand_s8(a1[3], b0);
    int8x8_t x305oo = vbic_s8(b0, a1[3]);
    int8x8_t x306__ = vcnt_s8(x304__);
    int8x8_t x307oo = vcnt_s8(x305oo);
    int8x8_t x308_ = vsub_s8(x306__, x307oo);
    int8x8_t x3011o = vadd_s8(x308_, x308_);
    int8x8_t x304___ = vand_s8(a2[3], b0);
    int8x8_t x305ooo = vbic_s8(b0, a2[3]);
    int8x8_t x306___ = vcnt_s8(x304___);
    int8x8_t x307ooo = vcnt_s8(x305ooo);
    int8x8_t x308__ = vsub_s8(x306___, x307ooo);
    int8x8_t x3012_ = vshl_n_s8(x308__, 2);
    int8x8_t x3010__ = vadd_s8(x3010_, x3011o);
    int8x8_t y3 = vadd_s8(x3010__, x3012_);
    int8x8_t x404_ = vand_s8(a0[4], b0);
    int8x8_t x405o = vcnt_s8(x404_);
    int8x8_t x406_ = vbic_s8(b0, a0[4]);
    int8x8_t x407o = vcnt_s8(x406_);
    int8x8_t x4010_ = vsub_s8(x405o, x407o);
    int8x8_t x404__ = vand_s8(a1[4], b0);
    int8x8_t x405oo = vbic_s8(b0, a1[4]);
    int8x8_t x406__ = vcnt_s8(x404__);
    int8x8_t x407oo = vcnt_s8(x405oo);
    int8x8_t x408_ = vsub_s8(x406__, x407oo);
    int8x8_t x4011o = vadd_s8(x408_, x408_);
    int8x8_t x404___ = vand_s8(a2[4], b0);
    int8x8_t x405ooo = vbic_s8(b0, a2[4]);
    int8x8_t x406___ = vcnt_s8(x404___);
    int8x8_t x407ooo = vcnt_s8(x405ooo);
    int8x8_t x408__ = vsub_s8(x406___, x407ooo);
    int8x8_t x4012_ = vshl_n_s8(x408__, 2);
    int8x8_t x4010__ = vadd_s8(x4010_, x4011o);
    int8x8_t y4 = vadd_s8(x4010__, x4012_);
    int8x8_t x504_ = vand_s8(a0[5], b0);
    int8x8_t x505o = vcnt_s8(x504_);
    int8x8_t x506_ = vbic_s8(b0, a0[5]);
    int8x8_t x507o = vcnt_s8(x506_);
    int8x8_t x5010_ = vsub_s8(x505o, x507o);
    int8x8_t x504__ = vand_s8(a1[5], b0);
    int8x8_t x505oo = vbic_s8(b0, a1[5]);
    int8x8_t x506__ = vcnt_s8(x504__);
    int8x8_t x507oo = vcnt_s8(x505oo);
    int8x8_t x508_ = vsub_s8(x506__, x507oo);
    int8x8_t x5011o = vadd_s8(x508_, x508_);
    int8x8_t x504___ = vand_s8(a2[5], b0);
    int8x8_t x505ooo = vbic_s8(b0, a2[5]);
    int8x8_t x506___ = vcnt_s8(x504___);
    int8x8_t x507ooo = vcnt_s8(x505ooo);
    int8x8_t x508__ = vsub_s8(x506___, x507ooo);
    int8x8_t x5012_ = vshl_n_s8(x508__, 2);
    int8x8_t x5010__ = vadd_s8(x5010_, x5011o);
    int8x8_t y5 = vadd_s8(x5010__, x5012_);
    int8x8_t x604_ = vand_s8(a0[6], b0);
    int8x8_t x605o = vcnt_s8(x604_);
    int8x8_t x606_ = vbic_s8(b0, a0[6]);
    int8x8_t x607o = vcnt_s8(x606_);
    int8x8_t x6010_ = vsub_s8(x605o, x607o);
    int8x8_t x604__ = vand_s8(a1[6], b0);
    int8x8_t x605oo = vbic_s8(b0, a1[6]);
    int8x8_t x606__ = vcnt_s8(x604__);
    int8x8_t x607oo = vcnt_s8(x605oo);
    int8x8_t x608_ = vsub_s8(x606__, x607oo);
    int8x8_t x6011o = vadd_s8(x608_, x608_);
    int8x8_t x604___ = vand_s8(a2[6], b0);
    int8x8_t x605ooo = vbic_s8(b0, a2[6]);
    int8x8_t x606___ = vcnt_s8(x604___);
    int8x8_t x607ooo = vcnt_s8(x605ooo);
    int8x8_t x608__ = vsub_s8(x606___, x607ooo);
    int8x8_t x6012_ = vshl_n_s8(x608__, 2);
    int8x8_t x6010__ = vadd_s8(x6010_, x6011o);
    int8x8_t y6 = vadd_s8(x6010__, x6012_);
    int8x8_t x704_ = vand_s8(a0[7], b0);
    int8x8_t x705o = vcnt_s8(x704_);
    int8x8_t x706_ = vbic_s8(b0, a0[7]);
    int8x8_t x707o = vcnt_s8(x706_);
    int8x8_t x7010_ = vsub_s8(x705o, x707o);
    int8x8_t x704__ = vand_s8(a1[7], b0);
    int8x8_t x705oo = vbic_s8(b0, a1[7]);
    int8x8_t x706__ = vcnt_s8(x704__);
    int8x8_t x707oo = vcnt_s8(x705oo);
    int8x8_t x708_ = vsub_s8(x706__, x707oo);
    int8x8_t x7011o = vadd_s8(x708_, x708_);
    int8x8_t x704___ = vand_s8(a2[7], b0);
    int8x8_t x705ooo = vbic_s8(b0, a2[7]);
    int8x8_t x706___ = vcnt_s8(x704___);
    int8x8_t x707ooo = vcnt_s8(x705ooo);
    int8x8_t x708__ = vsub_s8(x706___, x707ooo);
    int8x8_t x7012_ = vshl_n_s8(x708__, 2);
    int8x8_t x7010__ = vadd_s8(x7010_, x7011o);
    int8x8_t y7 = vadd_s8(x7010__, x7012_);
    int8x8_t y0_ = vpadd_s8(y0, y1);
    int8x8_t y1o = vpadd_s8(y2, y3);
    int8x8_t y2_ = vpadd_s8(y4, y5);
    int8x8_t y3o = vpadd_s8(y6, y7);
    int8x16_t y0__ = vpaddlq_s8(vcombine_s8(y0_, y1o));
    int8x16_t y2__ = vpaddlq_s8(vcombine_s8(y2_, y3o));
    int16x4_t y0___ = vpadd_s16(vget_low_s16(y0__), vget_high_s16(y0__));
    int16x4_t y0__o = vpadd_s16(vget_low_s16(y2__), vget_high_s16(y2__));
    int16x8_t out = vaddq_s16(vcombine_s16(y0___, y0__o), output);
    // todo data writebacks
    vst1q_s16(dst, out);
    return 0;
}

extern "C" int update_unipolar_a2b2_half(int8_t* src_a, int8_t* src_b, int16_t* dst, 
    int a_str1, int a_str0, int b_str0){

    int8x8_t a0[8];
    int8x8_t a1[8];
    for(int i = 0; i < 4; i++) {
        int8x16_t aa0 = vld1q_s8(src_a + i*2*a_str0);
        int8x16_t aa1 = vld1q_s8(src_a + a_str1 + i*2*a_str0);
        a0[2*i] = vget_low_s8(aa0);
        a0[2*i + 1] = vget_high_s8(aa0);
        a1[2*i] = vget_low_s8(aa1);
        a1[2*i + 1] = vget_high_s8(aa1);
    }
    
    uint8x8_t b0 = vld1_s8(src_b);
    uint8x8_t b1 = vld1_s8(src_b + b_str0);
    int16x8_t output = vld1q_s16(dst);

    int8x8_t x004_ = vand_s8(a0[0], b0);
    int8x8_t x005o = vcnt_s8(x004_);
    int8x8_t x006_ = vbic_s8(b0, a0[0]);
    int8x8_t x007o = vcnt_s8(x006_);
    int8x8_t x0010_ = vsub_s8(x005o, x007o);
    int8x8_t x004__ = vbic_s8(b1, a0[0]);
    int8x8_t x005oo = vand_s8(a0[0], b1);
    int8x8_t x006__ = vcnt_s8(x005oo);
    int8x8_t x007oo = vcnt_s8(x004__);
    int8x8_t x008_ = vsub_s8(x006__, x007oo);
    int8x8_t x0011o = vshl_n_s8(x008_, 1);
    int8x8_t x004___ = vbic_s8(b0, a1[0]);
    int8x8_t x005ooo = vand_s8(a1[0], b0);
    int8x8_t x006___ = vcnt_s8(x005ooo);
    int8x8_t x007ooo = vcnt_s8(x004___);
    int8x8_t x008__ = vsub_s8(x006___, x007ooo);
    int8x8_t x0012_ = vshl_n_s8(x008__, 1);
    int8x8_t x004____ = vbic_s8(b1, a1[0]);
    int8x8_t x005oooo = vand_s8(a1[0], b1);
    int8x8_t x006____ = vcnt_s8(x005oooo);
    int8x8_t x007oooo = vcnt_s8(x004____);
    int8x8_t x008___ = vsub_s8(x006____, x007oooo);
    int8x8_t x0013o = vshl_n_s8(x008___, 2);
    int8x8_t x0010__ = vadd_s8(x0010_, x0011o);
    int8x8_t x0010___ = vadd_s8(x0010__, x0012_);
    int8x8_t x0 = vadd_s8(x0010___, x0013o);
    int8x8_t x104_ = vand_s8(a0[1], b0);
    int8x8_t x105o = vcnt_s8(x104_);
    int8x8_t x106_ = vbic_s8(b0, a0[1]);
    int8x8_t x107o = vcnt_s8(x106_);
    int8x8_t x1010_ = vsub_s8(x105o, x107o);
    int8x8_t x104__ = vbic_s8(b1, a0[1]);
    int8x8_t x105oo = vand_s8(a0[1], b1);
    int8x8_t x106__ = vcnt_s8(x105oo);
    int8x8_t x107oo = vcnt_s8(x104__);
    int8x8_t x108_ = vsub_s8(x106__, x107oo);
    int8x8_t x1011o = vshl_n_s8(x108_, 1);
    int8x8_t x104___ = vbic_s8(b0, a1[1]);
    int8x8_t x105ooo = vand_s8(a1[1], b0);
    int8x8_t x106___ = vcnt_s8(x105ooo);
    int8x8_t x107ooo = vcnt_s8(x104___);
    int8x8_t x108__ = vsub_s8(x106___, x107ooo);
    int8x8_t x1012_ = vshl_n_s8(x108__, 1);
    int8x8_t x104____ = vbic_s8(b1, a1[1]);
    int8x8_t x105oooo = vand_s8(a1[1], b1);
    int8x8_t x106____ = vcnt_s8(x105oooo);
    int8x8_t x107oooo = vcnt_s8(x104____);
    int8x8_t x108___ = vsub_s8(x106____, x107oooo);
    int8x8_t x1013o = vshl_n_s8(x108___, 2);
    int8x8_t x1010__ = vadd_s8(x1010_, x1011o);
    int8x8_t x1010___ = vadd_s8(x1010__, x1012_);
    int8x8_t x1 = vadd_s8(x1010___, x1013o);
    int8x8_t x204_ = vand_s8(a0[2], b0);
    int8x8_t x205o = vcnt_s8(x204_);
    int8x8_t x206_ = vbic_s8(b0, a0[2]);
    int8x8_t x207o = vcnt_s8(x206_);
    int8x8_t x2010_ = vsub_s8(x205o, x207o);
    int8x8_t x204__ = vbic_s8(b1, a0[2]);
    int8x8_t x205oo = vand_s8(a0[2], b1);
    int8x8_t x206__ = vcnt_s8(x205oo);
    int8x8_t x207oo = vcnt_s8(x204__);
    int8x8_t x208_ = vsub_s8(x206__, x207oo);
    int8x8_t x2011o = vshl_n_s8(x208_, 1);
    int8x8_t x204___ = vbic_s8(b0, a1[2]);
    int8x8_t x205ooo = vand_s8(a1[2], b0);
    int8x8_t x206___ = vcnt_s8(x205ooo);
    int8x8_t x207ooo = vcnt_s8(x204___);
    int8x8_t x208__ = vsub_s8(x206___, x207ooo);
    int8x8_t x2012_ = vshl_n_s8(x208__, 1);
    int8x8_t x204____ = vbic_s8(b1, a1[2]);
    int8x8_t x205oooo = vand_s8(a1[2], b1);
    int8x8_t x206____ = vcnt_s8(x205oooo);
    int8x8_t x207oooo = vcnt_s8(x204____);
    int8x8_t x208___ = vsub_s8(x206____, x207oooo);
    int8x8_t x2013o = vshl_n_s8(x208___, 2);
    int8x8_t x2010__ = vadd_s8(x2010_, x2011o);
    int8x8_t x2010___ = vadd_s8(x2010__, x2012_);
    int8x8_t x2 = vadd_s8(x2010___, x2013o);
    int8x8_t x304_ = vand_s8(a0[3], b0);
    int8x8_t x305o = vcnt_s8(x304_);
    int8x8_t x306_ = vbic_s8(b0, a0[3]);
    int8x8_t x307o = vcnt_s8(x306_);
    int8x8_t x3010_ = vsub_s8(x305o, x307o);
    int8x8_t x304__ = vbic_s8(b1, a0[3]);
    int8x8_t x305oo = vand_s8(a0[3], b1);
    int8x8_t x306__ = vcnt_s8(x305oo);
    int8x8_t x307oo = vcnt_s8(x304__);
    int8x8_t x308_ = vsub_s8(x306__, x307oo);
    int8x8_t x3011o = vshl_n_s8(x308_, 1);
    int8x8_t x304___ = vbic_s8(b0, a1[3]);
    int8x8_t x305ooo = vand_s8(a1[3], b0);
    int8x8_t x306___ = vcnt_s8(x305ooo);
    int8x8_t x307ooo = vcnt_s8(x304___);
    int8x8_t x308__ = vsub_s8(x306___, x307ooo);
    int8x8_t x3012_ = vshl_n_s8(x308__, 1);
    int8x8_t x304____ = vbic_s8(b1, a1[3]);
    int8x8_t x305oooo = vand_s8(a1[3], b1);
    int8x8_t x306____ = vcnt_s8(x305oooo);
    int8x8_t x307oooo = vcnt_s8(x304____);
    int8x8_t x308___ = vsub_s8(x306____, x307oooo);
    int8x8_t x3013o = vshl_n_s8(x308___, 2);
    int8x8_t x3010__ = vadd_s8(x3010_, x3011o);
    int8x8_t x3010___ = vadd_s8(x3010__, x3012_);
    int8x8_t x3 = vadd_s8(x3010___, x3013o);
    int8x8_t x404_ = vand_s8(a0[4], b0);
    int8x8_t x405o = vcnt_s8(x404_);
    int8x8_t x406_ = vbic_s8(b0, a0[4]);
    int8x8_t x407o = vcnt_s8(x406_);
    int8x8_t x4010_ = vsub_s8(x405o, x407o);
    int8x8_t x404__ = vbic_s8(b1, a0[4]);
    int8x8_t x405oo = vand_s8(a0[4], b1);
    int8x8_t x406__ = vcnt_s8(x405oo);
    int8x8_t x407oo = vcnt_s8(x404__);
    int8x8_t x408_ = vsub_s8(x406__, x407oo);
    int8x8_t x4011o = vshl_n_s8(x408_, 1);
    int8x8_t x404___ = vbic_s8(b0, a1[4]);
    int8x8_t x405ooo = vand_s8(a1[4], b0);
    int8x8_t x406___ = vcnt_s8(x405ooo);
    int8x8_t x407ooo = vcnt_s8(x404___);
    int8x8_t x408__ = vsub_s8(x406___, x407ooo);
    int8x8_t x4012_ = vshl_n_s8(x408__, 1);
    int8x8_t x404____ = vbic_s8(b1, a1[4]);
    int8x8_t x405oooo = vand_s8(a1[4], b1);
    int8x8_t x406____ = vcnt_s8(x405oooo);
    int8x8_t x407oooo = vcnt_s8(x404____);
    int8x8_t x408___ = vsub_s8(x406____, x407oooo);
    int8x8_t x4013o = vshl_n_s8(x408___, 2);
    int8x8_t x4010__ = vadd_s8(x4010_, x4011o);
    int8x8_t x4010___ = vadd_s8(x4010__, x4012_);
    int8x8_t x4 = vadd_s8(x4010___, x4013o);
    int8x8_t x504_ = vand_s8(a0[5], b0);
    int8x8_t x505o = vcnt_s8(x504_);
    int8x8_t x506_ = vbic_s8(b0, a0[5]);
    int8x8_t x507o = vcnt_s8(x506_);
    int8x8_t x5010_ = vsub_s8(x505o, x507o);
    int8x8_t x504__ = vbic_s8(b1, a0[5]);
    int8x8_t x505oo = vand_s8(a0[5], b1);
    int8x8_t x506__ = vcnt_s8(x505oo);
    int8x8_t x507oo = vcnt_s8(x504__);
    int8x8_t x508_ = vsub_s8(x506__, x507oo);
    int8x8_t x5011o = vshl_n_s8(x508_, 1);
    int8x8_t x504___ = vbic_s8(b0, a1[5]);
    int8x8_t x505ooo = vand_s8(a1[5], b0);
    int8x8_t x506___ = vcnt_s8(x505ooo);
    int8x8_t x507ooo = vcnt_s8(x504___);
    int8x8_t x508__ = vsub_s8(x506___, x507ooo);
    int8x8_t x5012_ = vshl_n_s8(x508__, 1);
    int8x8_t x504____ = vbic_s8(b1, a1[5]);
    int8x8_t x505oooo = vand_s8(a1[5], b1);
    int8x8_t x506____ = vcnt_s8(x505oooo);
    int8x8_t x507oooo = vcnt_s8(x504____);
    int8x8_t x508___ = vsub_s8(x506____, x507oooo);
    int8x8_t x5013o = vshl_n_s8(x508___, 2);
    int8x8_t x5010__ = vadd_s8(x5010_, x5011o);
    int8x8_t x5010___ = vadd_s8(x5010__, x5012_);
    int8x8_t x5 = vadd_s8(x5010___, x5013o);
    int8x8_t x604_ = vand_s8(a0[6], b0);
    int8x8_t x605o = vcnt_s8(x604_);
    int8x8_t x606_ = vbic_s8(b0, a0[6]);
    int8x8_t x607o = vcnt_s8(x606_);
    int8x8_t x6010_ = vsub_s8(x605o, x607o);
    int8x8_t x604__ = vbic_s8(b1, a0[6]);
    int8x8_t x605oo = vand_s8(a0[6], b1);
    int8x8_t x606__ = vcnt_s8(x605oo);
    int8x8_t x607oo = vcnt_s8(x604__);
    int8x8_t x608_ = vsub_s8(x606__, x607oo);
    int8x8_t x6011o = vshl_n_s8(x608_, 1);
    int8x8_t x604___ = vbic_s8(b0, a1[6]);
    int8x8_t x605ooo = vand_s8(a1[6], b0);
    int8x8_t x606___ = vcnt_s8(x605ooo);
    int8x8_t x607ooo = vcnt_s8(x604___);
    int8x8_t x608__ = vsub_s8(x606___, x607ooo);
    int8x8_t x6012_ = vshl_n_s8(x608__, 1);
    int8x8_t x604____ = vbic_s8(b1, a1[6]);
    int8x8_t x605oooo = vand_s8(a1[6], b1);
    int8x8_t x606____ = vcnt_s8(x605oooo);
    int8x8_t x607oooo = vcnt_s8(x604____);
    int8x8_t x608___ = vsub_s8(x606____, x607oooo);
    int8x8_t x6013o = vshl_n_s8(x608___, 2);
    int8x8_t x6010__ = vadd_s8(x6010_, x6011o);
    int8x8_t x6010___ = vadd_s8(x6010__, x6012_);
    int8x8_t x6 = vadd_s8(x6010___, x6013o);
    int8x8_t x704_ = vand_s8(a0[7], b0);
    int8x8_t x705o = vcnt_s8(x704_);
    int8x8_t x706_ = vbic_s8(b0, a0[7]);
    int8x8_t x707o = vcnt_s8(x706_);
    int8x8_t x7010_ = vsub_s8(x705o, x707o);
    int8x8_t x704__ = vbic_s8(b1, a0[7]);
    int8x8_t x705oo = vand_s8(a0[7], b1);
    int8x8_t x706__ = vcnt_s8(x705oo);
    int8x8_t x707oo = vcnt_s8(x704__);
    int8x8_t x708_ = vsub_s8(x706__, x707oo);
    int8x8_t x7011o = vshl_n_s8(x708_, 1);
    int8x8_t x704___ = vbic_s8(b0, a1[7]);
    int8x8_t x705ooo = vand_s8(a1[7], b0);
    int8x8_t x706___ = vcnt_s8(x705ooo);
    int8x8_t x707ooo = vcnt_s8(x704___);
    int8x8_t x708__ = vsub_s8(x706___, x707ooo);
    int8x8_t x7012_ = vshl_n_s8(x708__, 1);
    int8x8_t x704____ = vbic_s8(b1, a1[7]);
    int8x8_t x705oooo = vand_s8(a1[7], b1);
    int8x8_t x706____ = vcnt_s8(x705oooo);
    int8x8_t x707oooo = vcnt_s8(x704____);
    int8x8_t x708___ = vsub_s8(x706____, x707oooo);
    int8x8_t x7013o = vshl_n_s8(x708___, 2);
    int8x8_t x7010__ = vadd_s8(x7010_, x7011o);
    int8x8_t x7010___ = vadd_s8(x7010__, x7012_);
    int8x8_t x7 = vadd_s8(x7010___, x7013o);
    int8x8_t x0_ = vpadd_s8(x0, x1);
    int8x8_t x1o = vpadd_s8(x2, x3);
    int8x8_t x2_ = vpadd_s8(x4, x5);
    int8x8_t x3o = vpadd_s8(x6, x7);
    int8x16_t x0__ = vpaddlq_s8(vcombine_s8(x0_, x1o));
    int8x16_t x2__ = vpaddlq_s8(vcombine_s8(x2_, x3o));
    int16x4_t x0___ = vpadd_s16(vget_low_s16(x0__), vget_high_s16(x0__));
    int16x4_t x0__o = vpadd_s16(vget_low_s16(x2__), vget_high_s16(x2__));
    int16x8_t out = vaddq_s16(vcombine_s16(x0___, x0__o), output);

    // manual
    vst1q_s16(dst, out);
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
extern "C" int update_unipolar_a1b1(int8_t* src_a, int8_t* src_b, int16_t* dst, 
    int a_str1, int a_str0, int b_str0){
    // Data loading
    int8x16_t a0[8];
    for(int i = 0; i < 8; i++)
        a0[i] = vld1q_s8(src_a + i*a_str0);
    uint8x16_t b0 = vld1q_s8(src_b);
    int16x8_t output = vld1q_s16(dst);

    int8x16_t x008_ = vandq_s8(a0[0], b0);
    int8x16_t x0010_ = vcntq_s8(x008_);
    int8x16_t x0012_ = vbicq_s8(b0, a0[0]);
    int8x16_t x0014_ = vcntq_s8(x0012_);
    int8x16_t y0 = vsubq_s8(x0010_, x0014_);
    int8x16_t x108_ = vandq_s8(a0[1], b0);
    int8x16_t x1010_ = vcntq_s8(x108_);
    int8x16_t x1012_ = vbicq_s8(b0, a0[1]);
    int8x16_t x1014_ = vcntq_s8(x1012_);
    int8x16_t y1 = vsubq_s8(x1010_, x1014_);
    int8x16_t x208_ = vandq_s8(a0[2], b0);
    int8x16_t x2010_ = vcntq_s8(x208_);
    int8x16_t x2012_ = vbicq_s8(b0, a0[2]);
    int8x16_t x2014_ = vcntq_s8(x2012_);
    int8x16_t y2 = vsubq_s8(x2010_, x2014_);
    int8x16_t x308_ = vandq_s8(a0[3], b0);
    int8x16_t x3010_ = vcntq_s8(x308_);
    int8x16_t x3012_ = vbicq_s8(b0, a0[3]);
    int8x16_t x3014_ = vcntq_s8(x3012_);
    int8x16_t y3 = vsubq_s8(x3010_, x3014_);
    int8x16_t x408_ = vandq_s8(a0[4], b0);
    int8x16_t x4010_ = vcntq_s8(x408_);
    int8x16_t x4012_ = vbicq_s8(b0, a0[4]);
    int8x16_t x4014_ = vcntq_s8(x4012_);
    int8x16_t y4 = vsubq_s8(x4010_, x4014_);
    int8x16_t x508_ = vandq_s8(a0[5], b0);
    int8x16_t x5010_ = vcntq_s8(x508_);
    int8x16_t x5012_ = vbicq_s8(b0, a0[5]);
    int8x16_t x5014_ = vcntq_s8(x5012_);
    int8x16_t y5 = vsubq_s8(x5010_, x5014_);
    int8x16_t x608_ = vandq_s8(a0[6], b0);
    int8x16_t x6010_ = vcntq_s8(x608_);
    int8x16_t x6012_ = vbicq_s8(b0, a0[6]);
    int8x16_t x6014_ = vcntq_s8(x6012_);
    int8x16_t y6 = vsubq_s8(x6010_, x6014_);
    int8x16_t x708_ = vandq_s8(a0[7], b0);
    int8x16_t x7010_ = vcntq_s8(x708_);
    int8x16_t x7012_ = vbicq_s8(b0, a0[7]);
    int8x16_t x7014_ = vcntq_s8(x7012_);
    int8x16_t y7 = vsubq_s8(x7010_, x7014_);
    int8x8_t y0_ = vpadd_s8(vget_low_s8(y0), vget_high_s8(y0));
    int8x8_t y0o = vpadd_s8(vget_low_s8(y1), vget_high_s8(y1));
    int8x8_t y1_ = vpadd_s8(vget_low_s8(y2), vget_high_s8(y2));
    int8x8_t y1o = vpadd_s8(vget_low_s8(y3), vget_high_s8(y3));
    int8x8_t y2_ = vpadd_s8(vget_low_s8(y4), vget_high_s8(y4));
    int8x8_t y2o = vpadd_s8(vget_low_s8(y5), vget_high_s8(y5));
    int8x8_t y3_ = vpadd_s8(vget_low_s8(y6), vget_high_s8(y6));
    int8x8_t y3o = vpadd_s8(vget_low_s8(y7), vget_high_s8(y7));
    int8x8_t y0__ = vpadd_s8(y0_, y0o);
    int8x8_t y0oo = vpadd_s8(y1_, y1o);
    int8x8_t y1__ = vpadd_s8(y2_, y2o);
    int8x8_t y1oo = vpadd_s8(y3_, y3o);
    int8x8_t y0___ = vpadd_s8(y0__, y0oo);
    int8x8_t y0ooo = vpadd_s8(y1__, y1oo);
    int8x16_t out = vpadalq_s8(output, vcombine_s8(y0___, y0ooo));
    // Data writeback
    vst1q_s16(dst, out);
    return 0;
}

extern "C" int update_unipolar_a1b2(int8_t* src_a, int8_t* src_b, int16_t* dst, 
    int a_str1, int a_str0, int b_str0){
    // Data loading
    int8x16_t a0[8];
    int8x16_t b0 = vld1q_s8(src_b);
    int8x16_t b1 = vld1q_s8(src_b + b_str0);
    int16x8_t output = vld1q_s16(dst);
    for(int i = 0; i < 8; i++)
        a0[i] = vld1q_s8(src_a + i*a_str0);

    int8x16_t x0012_ = vandq_s8(a0[0], b0);
    int8x16_t x0014_ = vbicq_s8(b0, a0[0]);
    int8x16_t x0016_ = vcntq_s8(x0012_);
    int8x16_t x0018_ = vcntq_s8(x0014_);
    int8x16_t x0020_ = vsubq_s8(x0016_, x0018_);
    int8x16_t x0012__ = vandq_s8(a0[0], b1);
    int8x16_t x0014__ = vcntq_s8(x0012__);
    int8x16_t x0016__ = vbicq_s8(b1, a0[0]);
    int8x16_t x0018__ = vcntq_s8(x0016__);
    int8x16_t x0020__ = vsubq_s8(x0014__, x0018__);
    int8x16_t x0022_ = vaddq_s8(x0020__, x0020__);
    int8x16_t y0 = vaddq_s8(x0020__, x0022_);
    int8x16_t x1012_ = vandq_s8(a0[1], b0);
    int8x16_t x1014_ = vbicq_s8(b0, a0[1]);
    int8x16_t x1016_ = vcntq_s8(x1012_);
    int8x16_t x1018_ = vcntq_s8(x1014_);
    int8x16_t x1020_ = vsubq_s8(x1016_, x1018_);
    int8x16_t x1012__ = vandq_s8(a0[1], b1);
    int8x16_t x1014__ = vcntq_s8(x1012__);
    int8x16_t x1016__ = vbicq_s8(b1, a0[1]);
    int8x16_t x1018__ = vcntq_s8(x1016__);
    int8x16_t x1020__ = vsubq_s8(x1014__, x1018__);
    int8x16_t x1022_ = vaddq_s8(x1020__, x1020__);
    int8x16_t y1 = vaddq_s8(x1020__, x1022_);
    int8x16_t x2012_ = vandq_s8(a0[2], b0);
    int8x16_t x2014_ = vbicq_s8(b0, a0[2]);
    int8x16_t x2016_ = vcntq_s8(x2012_);
    int8x16_t x2018_ = vcntq_s8(x2014_);
    int8x16_t x2020_ = vsubq_s8(x2016_, x2018_);
    int8x16_t x2012__ = vandq_s8(a0[2], b1);
    int8x16_t x2014__ = vcntq_s8(x2012__);
    int8x16_t x2016__ = vbicq_s8(b1, a0[2]);
    int8x16_t x2018__ = vcntq_s8(x2016__);
    int8x16_t x2020__ = vsubq_s8(x2014__, x2018__);
    int8x16_t x2022_ = vaddq_s8(x2020__, x2020__);
    int8x16_t y2 = vaddq_s8(x2020__, x2022_);
    int8x16_t x3012_ = vandq_s8(a0[3], b0);
    int8x16_t x3014_ = vbicq_s8(b0, a0[3]);
    int8x16_t x3016_ = vcntq_s8(x3012_);
    int8x16_t x3018_ = vcntq_s8(x3014_);
    int8x16_t x3020_ = vsubq_s8(x3016_, x3018_);
    int8x16_t x3012__ = vandq_s8(a0[3], b1);
    int8x16_t x3014__ = vcntq_s8(x3012__);
    int8x16_t x3016__ = vbicq_s8(b1, a0[3]);
    int8x16_t x3018__ = vcntq_s8(x3016__);
    int8x16_t x3020__ = vsubq_s8(x3014__, x3018__);
    int8x16_t x3022_ = vaddq_s8(x3020__, x3020__);
    int8x16_t y3 = vaddq_s8(x3020__, x3022_);
    int8x16_t x4012_ = vandq_s8(a0[4], b0);
    int8x16_t x4014_ = vbicq_s8(b0, a0[4]);
    int8x16_t x4016_ = vcntq_s8(x4012_);
    int8x16_t x4018_ = vcntq_s8(x4014_);
    int8x16_t x4020_ = vsubq_s8(x4016_, x4018_);
    int8x16_t x4012__ = vandq_s8(a0[4], b1);
    int8x16_t x4014__ = vcntq_s8(x4012__);
    int8x16_t x4016__ = vbicq_s8(b1, a0[4]);
    int8x16_t x4018__ = vcntq_s8(x4016__);
    int8x16_t x4020__ = vsubq_s8(x4014__, x4018__);
    int8x16_t x4022_ = vaddq_s8(x4020__, x4020__);
    int8x16_t y4 = vaddq_s8(x4020__, x4022_);
    int8x16_t x5012_ = vandq_s8(a0[5], b0);
    int8x16_t x5014_ = vbicq_s8(b0, a0[5]);
    int8x16_t x5016_ = vcntq_s8(x5012_);
    int8x16_t x5018_ = vcntq_s8(x5014_);
    int8x16_t x5020_ = vsubq_s8(x5016_, x5018_);
    int8x16_t x5012__ = vandq_s8(a0[5], b1);
    int8x16_t x5014__ = vcntq_s8(x5012__);
    int8x16_t x5016__ = vbicq_s8(b1, a0[5]);
    int8x16_t x5018__ = vcntq_s8(x5016__);
    int8x16_t x5020__ = vsubq_s8(x5014__, x5018__);
    int8x16_t x5022_ = vaddq_s8(x5020__, x5020__);
    int8x16_t y5 = vaddq_s8(x5020__, x5022_);
    int8x16_t x6012_ = vandq_s8(a0[6], b0);
    int8x16_t x6014_ = vbicq_s8(b0, a0[6]);
    int8x16_t x6016_ = vcntq_s8(x6012_);
    int8x16_t x6018_ = vcntq_s8(x6014_);
    int8x16_t x6020_ = vsubq_s8(x6016_, x6018_);
    int8x16_t x6012__ = vandq_s8(a0[6], b1);
    int8x16_t x6014__ = vcntq_s8(x6012__);
    int8x16_t x6016__ = vbicq_s8(b1, a0[6]);
    int8x16_t x6018__ = vcntq_s8(x6016__);
    int8x16_t x6020__ = vsubq_s8(x6014__, x6018__);
    int8x16_t x6022_ = vaddq_s8(x6020__, x6020__);
    int8x16_t y6 = vaddq_s8(x6020__, x6022_);
    int8x16_t x7012_ = vandq_s8(a0[7], b0);
    int8x16_t x7014_ = vbicq_s8(b0, a0[7]);
    int8x16_t x7016_ = vcntq_s8(x7012_);
    int8x16_t x7018_ = vcntq_s8(x7014_);
    int8x16_t x7020_ = vsubq_s8(x7016_, x7018_);
    int8x16_t x7012__ = vandq_s8(a0[7], b1);
    int8x16_t x7014__ = vcntq_s8(x7012__);
    int8x16_t x7016__ = vbicq_s8(b1, a0[7]);
    int8x16_t x7018__ = vcntq_s8(x7016__);
    int8x16_t x7020__ = vsubq_s8(x7014__, x7018__);
    int8x16_t x7022_ = vaddq_s8(x7020__, x7020__);
    int8x16_t y7 = vaddq_s8(x7020__, x7022_);
    int8x8_t y0_ = vpadd_s8(vget_low_s8(y0), vget_high_s8(y0));
    int8x8_t y0o = vpadd_s8(vget_low_s8(y1), vget_high_s8(y1));
    int8x8_t y1_ = vpadd_s8(vget_low_s8(y2), vget_high_s8(y2));
    int8x8_t y1o = vpadd_s8(vget_low_s8(y3), vget_high_s8(y3));
    int8x8_t y2_ = vpadd_s8(vget_low_s8(y4), vget_high_s8(y4));
    int8x8_t y2o = vpadd_s8(vget_low_s8(y5), vget_high_s8(y5));
    int8x8_t y3_ = vpadd_s8(vget_low_s8(y6), vget_high_s8(y6));
    int8x8_t y3o = vpadd_s8(vget_low_s8(y7), vget_high_s8(y7));
    int8x8_t y0__ = vpadd_s8(y0_, y0o);
    int8x8_t y0oo = vpadd_s8(y1_, y1o);
    int8x8_t y1__ = vpadd_s8(y2_, y2o);
    int8x8_t y1oo = vpadd_s8(y3_, y3o);
    int8x16_t y0___ = vpaddlq_s8(vcombine_s8(y0__, y0oo));
    int8x16_t y1___ = vpaddlq_s8(vcombine_s8(y1__, y1oo));
    int16x4_t y0____ = vpadd_s16(vget_low_s16(y0___), vget_high_s16(y0___));
    int16x4_t y0___o = vpadd_s16(vget_low_s16(y1___), vget_high_s16(y1___));
    int16x8_t out = vaddq_s16(vcombine_s16(y0____, y0___o), output);

    // Data writeback
    vst1q_s16(dst, out);
    return 0;
}

extern "C" int update_unipolar_a2b2(int8_t* src_a, int8_t* src_b, int16_t* dst, 
    int a_str1, int a_str0, int b_str0){
    // Data loading
    int8x16_t a0[8];
    int8x16_t a1[8];
    int16x8_t output = vld1q_s16(dst);

    uint8x16_t a[8];
    for(int i = 0; i < 8; i++) {
        a0[i] = vld1q_s8(src_a + i*a_str0);
        a1[i] = vld1q_s8(src_a + a_str1 + i*a_str0);
    }
    uint8x16_t b0 = vld1q_s8(src_b);
    uint8x16_t b1 = vld1q_s8(src_b + b_str0);
            
    int8x16_t x0016_ = vbicq_s8(b0, a0[0]);
    int8x16_t x0018_ = vandq_s8(a0[0], b0);
    int8x16_t x0020_ = vcntq_s8(x0018_);
    int8x16_t x0022_ = vcntq_s8(x0016_);
    int8x16_t x0020__ = vsubq_s8(x0020_, x0022_);
    int8x16_t x0016__ = vandq_s8(a0[0], b1);
    int8x16_t x0018__ = vbicq_s8(b1, a0[0]);
    int8x16_t x0020___ = vcntq_s8(x0016__);
    int8x16_t x0022__ = vcntq_s8(x0018__);
    int8x16_t x0024_ = vsubq_s8(x0020___, x0022__);
    int8x16_t x0022___ = vaddq_s8(x0024_, x0024_);
    int8x16_t x0016___ = vandq_s8(a1[0], b0);
    int8x16_t x0018___ = vcntq_s8(x0016___);
    int8x16_t x0020____ = vbicq_s8(b0, a1[0]);
    int8x16_t x0022____ = vcntq_s8(x0020____);
    int8x16_t x0024__ = vsubq_s8(x0018___, x0022____);
    int8x16_t x0024___ = vaddq_s8(x0024__, x0024__);
    int8x16_t x0016____ = vandq_s8(a1[0], b1);
    int8x16_t x0018____ = vcntq_s8(x0016____);
    int8x16_t x0020_____ = vbicq_s8(b1, a1[0]);
    int8x16_t x0022_____ = vcntq_s8(x0020_____);
    int8x16_t x0024____ = vsubq_s8(x0018____, x0022_____);
    int8x16_t x0026_ = vshlq_n_s8(x0024____, 2);
    int8x16_t x0020______ = vaddq_s8(x0020_____, x0022_____);
    int8x16_t x0020_______ = vaddq_s8(x0020______, x0024____);
    int8x16_t y0 = vaddq_s8(x0020_______, x0026_);
    int8x16_t x1016_ = vbicq_s8(b0, a0[1]);
    int8x16_t x1018_ = vandq_s8(a0[1], b0);
    int8x16_t x1020_ = vcntq_s8(x1018_);
    int8x16_t x1022_ = vcntq_s8(x1016_);
    int8x16_t x1020__ = vsubq_s8(x1020_, x1022_);
    int8x16_t x1016__ = vandq_s8(a0[1], b1);
    int8x16_t x1018__ = vbicq_s8(b1, a0[1]);
    int8x16_t x1020___ = vcntq_s8(x1016__);
    int8x16_t x1022__ = vcntq_s8(x1018__);
    int8x16_t x1024_ = vsubq_s8(x1020___, x1022__);
    int8x16_t x1022___ = vaddq_s8(x1024_, x1024_);
    int8x16_t x1016___ = vandq_s8(a1[1], b0);
    int8x16_t x1018___ = vcntq_s8(x1016___);
    int8x16_t x1020____ = vbicq_s8(b0, a1[1]);
    int8x16_t x1022____ = vcntq_s8(x1020____);
    int8x16_t x1024__ = vsubq_s8(x1018___, x1022____);
    int8x16_t x1024___ = vaddq_s8(x1024__, x1024__);
    int8x16_t x1016____ = vandq_s8(a1[1], b1);
    int8x16_t x1018____ = vcntq_s8(x1016____);
    int8x16_t x1020_____ = vbicq_s8(b1, a1[1]);
    int8x16_t x1022_____ = vcntq_s8(x1020_____);
    int8x16_t x1024____ = vsubq_s8(x1018____, x1022_____);
    int8x16_t x1026_ = vshlq_n_s8(x1024____, 2);
    int8x16_t x1020______ = vaddq_s8(x1020_____, x1022_____);
    int8x16_t x1020_______ = vaddq_s8(x1020______, x1024____);
    int8x16_t y1 = vaddq_s8(x1020_______, x1026_);
    int8x16_t x2016_ = vbicq_s8(b0, a0[2]);
    int8x16_t x2018_ = vandq_s8(a0[2], b0);
    int8x16_t x2020_ = vcntq_s8(x2018_);
    int8x16_t x2022_ = vcntq_s8(x2016_);
    int8x16_t x2020__ = vsubq_s8(x2020_, x2022_);
    int8x16_t x2016__ = vandq_s8(a0[2], b1);
    int8x16_t x2018__ = vbicq_s8(b1, a0[2]);
    int8x16_t x2020___ = vcntq_s8(x2016__);
    int8x16_t x2022__ = vcntq_s8(x2018__);
    int8x16_t x2024_ = vsubq_s8(x2020___, x2022__);
    int8x16_t x2022___ = vaddq_s8(x2024_, x2024_);
    int8x16_t x2016___ = vandq_s8(a1[2], b0);
    int8x16_t x2018___ = vcntq_s8(x2016___);
    int8x16_t x2020____ = vbicq_s8(b0, a1[2]);
    int8x16_t x2022____ = vcntq_s8(x2020____);
    int8x16_t x2024__ = vsubq_s8(x2018___, x2022____);
    int8x16_t x2024___ = vaddq_s8(x2024__, x2024__);
    int8x16_t x2016____ = vandq_s8(a1[2], b1);
    int8x16_t x2018____ = vcntq_s8(x2016____);
    int8x16_t x2020_____ = vbicq_s8(b1, a1[2]);
    int8x16_t x2022_____ = vcntq_s8(x2020_____);
    int8x16_t x2024____ = vsubq_s8(x2018____, x2022_____);
    int8x16_t x2026_ = vshlq_n_s8(x2024____, 2);
    int8x16_t x2020______ = vaddq_s8(x2020_____, x2022_____);
    int8x16_t x2020_______ = vaddq_s8(x2020______, x2024____);
    int8x16_t y2 = vaddq_s8(x2020_______, x2026_);
    int8x16_t x3016_ = vbicq_s8(b0, a0[3]);
    int8x16_t x3018_ = vandq_s8(a0[3], b0);
    int8x16_t x3020_ = vcntq_s8(x3018_);
    int8x16_t x3022_ = vcntq_s8(x3016_);
    int8x16_t x3020__ = vsubq_s8(x3020_, x3022_);
    int8x16_t x3016__ = vandq_s8(a0[3], b1);
    int8x16_t x3018__ = vbicq_s8(b1, a0[3]);
    int8x16_t x3020___ = vcntq_s8(x3016__);
    int8x16_t x3022__ = vcntq_s8(x3018__);
    int8x16_t x3024_ = vsubq_s8(x3020___, x3022__);
    int8x16_t x3022___ = vaddq_s8(x3024_, x3024_);
    int8x16_t x3016___ = vandq_s8(a1[3], b0);
    int8x16_t x3018___ = vcntq_s8(x3016___);
    int8x16_t x3020____ = vbicq_s8(b0, a1[3]);
    int8x16_t x3022____ = vcntq_s8(x3020____);
    int8x16_t x3024__ = vsubq_s8(x3018___, x3022____);
    int8x16_t x3024___ = vaddq_s8(x3024__, x3024__);
    int8x16_t x3016____ = vandq_s8(a1[3], b1);
    int8x16_t x3018____ = vcntq_s8(x3016____);
    int8x16_t x3020_____ = vbicq_s8(b1, a1[3]);
    int8x16_t x3022_____ = vcntq_s8(x3020_____);
    int8x16_t x3024____ = vsubq_s8(x3018____, x3022_____);
    int8x16_t x3026_ = vshlq_n_s8(x3024____, 2);
    int8x16_t x3020______ = vaddq_s8(x3020_____, x3022_____);
    int8x16_t x3020_______ = vaddq_s8(x3020______, x3024____);
    int8x16_t y3 = vaddq_s8(x3020_______, x3026_);
    int8x16_t x4016_ = vbicq_s8(b0, a0[4]);
    int8x16_t x4018_ = vandq_s8(a0[4], b0);
    int8x16_t x4020_ = vcntq_s8(x4018_);
    int8x16_t x4022_ = vcntq_s8(x4016_);
    int8x16_t x4020__ = vsubq_s8(x4020_, x4022_);
    int8x16_t x4016__ = vandq_s8(a0[4], b1);
    int8x16_t x4018__ = vbicq_s8(b1, a0[4]);
    int8x16_t x4020___ = vcntq_s8(x4016__);
    int8x16_t x4022__ = vcntq_s8(x4018__);
    int8x16_t x4024_ = vsubq_s8(x4020___, x4022__);
    int8x16_t x4022___ = vaddq_s8(x4024_, x4024_);
    int8x16_t x4016___ = vandq_s8(a1[4], b0);
    int8x16_t x4018___ = vcntq_s8(x4016___);
    int8x16_t x4020____ = vbicq_s8(b0, a1[4]);
    int8x16_t x4022____ = vcntq_s8(x4020____);
    int8x16_t x4024__ = vsubq_s8(x4018___, x4022____);
    int8x16_t x4024___ = vaddq_s8(x4024__, x4024__);
    int8x16_t x4016____ = vandq_s8(a1[4], b1);
    int8x16_t x4018____ = vcntq_s8(x4016____);
    int8x16_t x4020_____ = vbicq_s8(b1, a1[4]);
    int8x16_t x4022_____ = vcntq_s8(x4020_____);
    int8x16_t x4024____ = vsubq_s8(x4018____, x4022_____);
    int8x16_t x4026_ = vshlq_n_s8(x4024____, 2);
    int8x16_t x4020______ = vaddq_s8(x4020_____, x4022_____);
    int8x16_t x4020_______ = vaddq_s8(x4020______, x4024____);
    int8x16_t y4 = vaddq_s8(x4020_______, x4026_);
    int8x16_t x5016_ = vbicq_s8(b0, a0[5]);
    int8x16_t x5018_ = vandq_s8(a0[5], b0);
    int8x16_t x5020_ = vcntq_s8(x5018_);
    int8x16_t x5022_ = vcntq_s8(x5016_);
    int8x16_t x5020__ = vsubq_s8(x5020_, x5022_);
    int8x16_t x5016__ = vandq_s8(a0[5], b1);
    int8x16_t x5018__ = vbicq_s8(b1, a0[5]);
    int8x16_t x5020___ = vcntq_s8(x5016__);
    int8x16_t x5022__ = vcntq_s8(x5018__);
    int8x16_t x5024_ = vsubq_s8(x5020___, x5022__);
    int8x16_t x5022___ = vaddq_s8(x5024_, x5024_);
    int8x16_t x5016___ = vandq_s8(a1[5], b0);
    int8x16_t x5018___ = vcntq_s8(x5016___);
    int8x16_t x5020____ = vbicq_s8(b0, a1[5]);
    int8x16_t x5022____ = vcntq_s8(x5020____);
    int8x16_t x5024__ = vsubq_s8(x5018___, x5022____);
    int8x16_t x5024___ = vaddq_s8(x5024__, x5024__);
    int8x16_t x5016____ = vandq_s8(a1[5], b1);
    int8x16_t x5018____ = vcntq_s8(x5016____);
    int8x16_t x5020_____ = vbicq_s8(b1, a1[5]);
    int8x16_t x5022_____ = vcntq_s8(x5020_____);
    int8x16_t x5024____ = vsubq_s8(x5018____, x5022_____);
    int8x16_t x5026_ = vshlq_n_s8(x5024____, 2);
    int8x16_t x5020______ = vaddq_s8(x5020_____, x5022_____);
    int8x16_t x5020_______ = vaddq_s8(x5020______, x5024____);
    int8x16_t y5 = vaddq_s8(x5020_______, x5026_);
    int8x16_t x6016_ = vbicq_s8(b0, a0[6]);
    int8x16_t x6018_ = vandq_s8(a0[6], b0);
    int8x16_t x6020_ = vcntq_s8(x6018_);
    int8x16_t x6022_ = vcntq_s8(x6016_);
    int8x16_t x6020__ = vsubq_s8(x6020_, x6022_);
    int8x16_t x6016__ = vandq_s8(a0[6], b1);
    int8x16_t x6018__ = vbicq_s8(b1, a0[6]);
    int8x16_t x6020___ = vcntq_s8(x6016__);
    int8x16_t x6022__ = vcntq_s8(x6018__);
    int8x16_t x6024_ = vsubq_s8(x6020___, x6022__);
    int8x16_t x6022___ = vaddq_s8(x6024_, x6024_);
    int8x16_t x6016___ = vandq_s8(a1[6], b0);
    int8x16_t x6018___ = vcntq_s8(x6016___);
    int8x16_t x6020____ = vbicq_s8(b0, a1[6]);
    int8x16_t x6022____ = vcntq_s8(x6020____);
    int8x16_t x6024__ = vsubq_s8(x6018___, x6022____);
    int8x16_t x6024___ = vaddq_s8(x6024__, x6024__);
    int8x16_t x6016____ = vandq_s8(a1[6], b1);
    int8x16_t x6018____ = vcntq_s8(x6016____);
    int8x16_t x6020_____ = vbicq_s8(b1, a1[6]);
    int8x16_t x6022_____ = vcntq_s8(x6020_____);
    int8x16_t x6024____ = vsubq_s8(x6018____, x6022_____);
    int8x16_t x6026_ = vshlq_n_s8(x6024____, 2);
    int8x16_t x6020______ = vaddq_s8(x6020_____, x6022_____);
    int8x16_t x6020_______ = vaddq_s8(x6020______, x6024____);
    int8x16_t y6 = vaddq_s8(x6020_______, x6026_);
    int8x16_t x7016_ = vbicq_s8(b0, a0[7]);
    int8x16_t x7018_ = vandq_s8(a0[7], b0);
    int8x16_t x7020_ = vcntq_s8(x7018_);
    int8x16_t x7022_ = vcntq_s8(x7016_);
    int8x16_t x7020__ = vsubq_s8(x7020_, x7022_);
    int8x16_t x7016__ = vandq_s8(a0[7], b1);
    int8x16_t x7018__ = vbicq_s8(b1, a0[7]);
    int8x16_t x7020___ = vcntq_s8(x7016__);
    int8x16_t x7022__ = vcntq_s8(x7018__);
    int8x16_t x7024_ = vsubq_s8(x7020___, x7022__);
    int8x16_t x7022___ = vaddq_s8(x7024_, x7024_);
    int8x16_t x7016___ = vandq_s8(a1[7], b0);
    int8x16_t x7018___ = vcntq_s8(x7016___);
    int8x16_t x7020____ = vbicq_s8(b0, a1[7]);
    int8x16_t x7022____ = vcntq_s8(x7020____);
    int8x16_t x7024__ = vsubq_s8(x7018___, x7022____);
    int8x16_t x7024___ = vaddq_s8(x7024__, x7024__);
    int8x16_t x7016____ = vandq_s8(a1[7], b1);
    int8x16_t x7018____ = vcntq_s8(x7016____);
    int8x16_t x7020_____ = vbicq_s8(b1, a1[7]);
    int8x16_t x7022_____ = vcntq_s8(x7020_____);
    int8x16_t x7024____ = vsubq_s8(x7018____, x7022_____);
    int8x16_t x7026_ = vshlq_n_s8(x7024____, 2);
    int8x16_t x7020______ = vaddq_s8(x7020_____, x7022_____);
    int8x16_t x7020_______ = vaddq_s8(x7020______, x7024____);
    int8x16_t y7 = vaddq_s8(x7020_______, x7026_);
    int8x8_t y0_ = vpadd_s8(vget_low_s8(y0), vget_high_s8(y0));
    int8x8_t y0o = vpadd_s8(vget_low_s8(y1), vget_high_s8(y1));
    int8x8_t y1_ = vpadd_s8(vget_low_s8(y2), vget_high_s8(y2));
    int8x8_t y1o = vpadd_s8(vget_low_s8(y3), vget_high_s8(y3));
    int8x8_t y2_ = vpadd_s8(vget_low_s8(y4), vget_high_s8(y4));
    int8x8_t y2o = vpadd_s8(vget_low_s8(y5), vget_high_s8(y5));
    int8x8_t y3_ = vpadd_s8(vget_low_s8(y6), vget_high_s8(y6));
    int8x8_t y3o = vpadd_s8(vget_low_s8(y7), vget_high_s8(y7));
    int8x16_t y0__ = vpaddlq_s8(vcombine_s8(y0_, y0o));
    int8x16_t y1__ = vpaddlq_s8(vcombine_s8(y1_, y1o));
    int8x16_t y2__ = vpaddlq_s8(vcombine_s8(y2_, y2o));
    int8x16_t y3__ = vpaddlq_s8(vcombine_s8(y3_, y3o));
    int16x4_t y0___ = vpadd_s16(vget_low_s16(y0__), vget_high_s16(y0__));
    int16x4_t y0__o = vpadd_s16(vget_low_s16(y1__), vget_high_s16(y1__));
    int16x4_t y1___ = vpadd_s16(vget_low_s16(y2__), vget_high_s16(y2__));
    int16x4_t y1__o = vpadd_s16(vget_low_s16(y3__), vget_high_s16(y3__));
    int16x4_t y0____ = vpadd_s16(y0___, y0__o);
    int16x4_t y0__oo = vpadd_s16(y1___, y1__o);
    int16x8_t out = vaddq_s16(vcombine_s16(y0____, y0__oo), output);

    // todo data writeback
    vst1q_s16(dst, out);
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
