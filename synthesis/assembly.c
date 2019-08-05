#include "stdint.h"
#include "arm_neon.h"

// Matrix multiply using inline arm assembly.
// Compilation arm-linux-gnueabihf-g++ translate.cpp -o translate -static -mfpu=neon

extern "C" int reset(int16_t* dst) {
    for (int i = 0; i < 8; i++)
	    dst[i] = 0;
    return 0;
}

extern "C" int first(int8_t* src_a, int16_t* dst) {
    asm volatile(
        "vld1.8 {d0, d1}, [%0]!\n"
        "vld1.8 {d2, d3}, [%0]!\n"
        "vld1.8 {d4, d5}, [%0]!\n"
        "vld1.8 {d6, d7}, [%0]!\n"
        "vld1.8 {d8, d9}, [%0]!\n"
        "vld1.8 {d10, d11}, [%0]!\n"
        "vld1.8 {d12, d13}, [%0]!\n"
        "vld1.8 {d14, d15}, [%0]\n"
        "vpadd.i8 D0, D0, D1\n"
        "vpadd.i8 D1, D2, D3\n"
        "vpadd.i8 D2, D4, D5\n"
        "vpadd.i8 D3, D6, D7\n"
        "vpadd.i8 D4, D8, D9\n"
        "vpadd.i8 D5, D10, D11\n"
        "vpadd.i8 D6, D12, D13\n"
        "vpadd.i8 D7, D14, D15\n"
        "vpadd.i8 D0, D0, D1\n"
        "vpadd.i8 D1, D2, D3\n"
        "vpadd.i8 D2, D4, D5\n"
        "vpadd.i8 D3, D6, D7\n"
        "vpadd.i8 D10, D0, D1\n"
        "vpadd.i8 D11, D2, D3\n"
        "vpaddl.u8 Q2, Q5 \n"

        "vst1.16 {d4, d5}, [%1]\n"
        : "=r"(src_a), "=r"(dst)
        : "0"(src_a), "1"(dst)
        : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11"
    );
   return 0;
}

extern "C" int update(int8_t* src_a, int16_t* dst) {
    asm volatile(
        "vld1.8 {d0, d1}, [%0]!\n"
        "vld1.8 {d2, d3}, [%0]!\n"
        "vld1.8 {d4, d5}, [%0]!\n"
        "vld1.8 {d6, d7}, [%0]!\n"
        "vld1.8 {d8, d9}, [%0]!\n"
        "vld1.8 {d10, d11}, [%0]!\n"
        "vld1.8 {d12, d13}, [%0]!\n"
        "vld1.8 {d14, d15}, [%0]\n"
        "vpadd.i8 D0, D0, D1\n"
        "vpadd.i8 D1, D2, D3\n"
        "vpadd.i8 D2, D4, D5\n"
        "vpadd.i8 D3, D6, D7\n"
        "vpadd.i8 D4, D8, D9\n"
        "vpadd.i8 D5, D10, D11\n"
        "vpadd.i8 D6, D12, D13\n"
        "vpadd.i8 D7, D14, D15\n"
        "vpadd.i8 D0, D0, D1\n"
        "vpadd.i8 D1, D2, D3\n"
        "vpadd.i8 D2, D4, D5\n"
        "vpadd.i8 D3, D6, D7\n"
        "vpadd.i8 D10, D0, D1\n"
        "vpadd.i8 D11, D2, D3\n"
        "vld1.16 {d4, d5}, [%1]\n"
	"vpadal.u8 Q2, Q5 \n"

        "vst1.16 {d4, d5}, [%1]\n"
        : "=r"(src_a), "=r"(dst)
        : "0"(src_a), "1"(dst)
        : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11"
    );
   return 0;
}
