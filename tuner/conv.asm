	.text
	.syntax unified
	.eabi_attribute	67, "2.09"
	.eabi_attribute	6, 1
	.eabi_attribute	8, 1
	.fpu	neon
	.eabi_attribute	34, 1
	.eabi_attribute	15, 1
	.eabi_attribute	16, 1
	.eabi_attribute	17, 2
	.eabi_attribute	20, 2
	.eabi_attribute	23, 1
	.eabi_attribute	24, 1
	.eabi_attribute	25, 1
	.eabi_attribute	28, 1
	.eabi_attribute	38, 1
	.eabi_attribute	14, 0
	.file	"default_function"
	.globl	default_function
	.p2align	3
	.type	default_function,%function
	.code	32
default_function:
	.fnstart
	push	{r4, r5, r6, r7, r8, r9, r10, r11, lr}
	sub	sp, sp, #12
	cmp	r2, #3
	bne	.LBB0_59
	ldr	r6, [r0]
	ldr	lr, [r0, #8]
	ldr	r12, [r0, #16]
	ldr	r5, [r1]
	ldmib	r1, {r4, r9}
	ldr	r0, [r6]
	str	r0, [sp, #4]
	ldr	r0, [r6, #24]
	ldr	r7, [r6, #20]
	cmp	r0, #0
	beq	.LBB0_6
	ldr	r1, [r0]
	cmp	r1, #200704
	ldreq	r1, [r0, #8]
	cmpeq	r1, #3584
	beq	.LBB0_4
.LBB0_3:
	ldr	r0, .LCPI0_32
.LPC0_31:
	add	r0, pc, r0
	b	.LBB0_49
.LBB0_4:
	add	r0, r0, #16
	vldr	d18, .LCPI0_35
	vld1.64	{d16, d17}, [r0]
	vmovn.i64	d16, q8
	vceq.i32	d16, d16, d18
	vmov.32	r0, d16[0]
	tst	r0, #1
	beq	.LBB0_3
	vmov.32	r0, d16[1]
	tst	r0, #1
	beq	.LBB0_3
.LBB0_6:
	ldr	r2, [lr, #24]
	ldr	r10, [lr, #20]
	ldr	r3, [r6, #4]
	cmp	r2, #0
	ldr	r0, [lr]
	str	r0, [sp]
	ldr	r0, [r6, #8]
	str	r0, [sp, #8]
	beq	.LBB0_11
	ldr	r1, [r2]
	cmp	r1, #12288
	ldreq	r1, [r2, #8]
	cmpeq	r1, #4096
	beq	.LBB0_9
.LBB0_8:
	ldr	r0, .LCPI0_33
.LPC0_32:
	add	r0, pc, r0
	b	.LBB0_49
.LBB0_9:
	add	r1, r2, #16
	vldr	d18, .LCPI0_0
	vld1.64	{d16, d17}, [r1]
	vmovn.i64	d16, q8
	vceq.i32	d16, d16, d18
	vmov.32	r1, d16[0]
	tst	r1, #1
	beq	.LBB0_8
	vmov.32	r1, d16[1]
	tst	r1, #1
	beq	.LBB0_8
.LBB0_11:
	ldr	r1, [r12, #24]
	ldr	r2, [r12]
	ldr	r8, [r12, #20]
	cmp	r1, #0
	beq	.LBB0_16
	ldr	r0, [r1]
	cmp	r0, #200704
	ldreq	r0, [r1, #8]
	cmpeq	r0, #3584
	beq	.LBB0_14
.LBB0_13:
	ldr	r0, .LCPI0_34
.LPC0_33:
	add	r0, pc, r0
	b	.LBB0_49
.LBB0_14:
	add	r0, r1, #16
	vldr	d18, .LCPI0_0
	vld1.64	{d16, d17}, [r0]
	vmovn.i64	d16, q8
	vceq.i32	d16, d16, d18
	vmov.32	r0, d16[0]
	tst	r0, #1
	beq	.LBB0_13
	vmov.32	r0, d16[1]
	tst	r0, #1
	beq	.LBB0_13
.LBB0_16:
	cmp	r5, #13
	bhi	.LBB0_35
	mov	r11, #152
	mov	r0, #1
	orr	r11, r11, #8192
	tst	r11, r0, lsl r5
	beq	.LBB0_35
	cmp	r4, #13
	bhi	.LBB0_36
	mov	r0, #1
	tst	r11, r0, lsl r4
	beq	.LBB0_36
	cmp	r9, #13
	bhi	.LBB0_37
	mov	r0, #1
	tst	r11, r0, lsl r9
	beq	.LBB0_37
	cmp	r3, #1
	bne	.LBB0_60
	ldr	r0, [r6, #12]
	cmp	r0, #4
	bne	.LBB0_61
	ldrh	r0, [r6, #18]
	cmp	r0, #1
	ldrbeq	r0, [r6, #17]
	cmpeq	r0, #8
	beq	.LBB0_26
.LBB0_25:
	ldr	r0, .LCPI0_8
.LPC0_7:
	add	r0, pc, r0
	b	.LBB0_49
.LBB0_26:
	ldrb	r0, [r6, #16]
	cmp	r0, #1
	bne	.LBB0_25
	ldr	r0, [r7]
	cmp	r0, #1
	bne	.LBB0_62
	ldr	r0, [r7, #8]
	cmp	r0, #56
	bne	.LBB0_63
	ldr	r0, [r7, #16]
	cmp	r0, #56
	bne	.LBB0_64
	ldr	r0, [r7, #24]
	cmp	r0, #64
	bne	.LBB0_65
	ldr	r0, [r6, #32]
	ldr	r1, [r6, #36]
	orrs	r0, r0, r1
	bne	.LBB0_66
	ldr	r0, [lr, #12]
	cmp	r0, #4
	bne	.LBB0_67
	ldrh	r0, [lr, #18]
	cmp	r0, #1
	ldrbeq	r0, [lr, #17]
	cmpeq	r0, #8
	beq	.LBB0_38
.LBB0_34:
	ldr	r0, .LCPI0_15
.LPC0_14:
	add	r0, pc, r0
	b	.LBB0_49
.LBB0_35:
	ldr	r0, .LCPI0_3
.LPC0_2:
	add	r0, pc, r0
	b	.LBB0_49
.LBB0_36:
	ldr	r0, .LCPI0_4
.LPC0_3:
	add	r0, pc, r0
	b	.LBB0_49
.LBB0_37:
	ldr	r0, .LCPI0_5
.LPC0_4:
	add	r0, pc, r0
	b	.LBB0_49
.LBB0_38:
	ldrb	r0, [lr, #16]
	cmp	r0, #1
	bne	.LBB0_34
	ldr	r0, [r10]
	cmp	r0, #3
	bne	.LBB0_68
	ldr	r0, [r10, #8]
	cmp	r0, #3
	bne	.LBB0_69
	ldr	r0, [r10, #16]
	cmp	r0, #64
	bne	.LBB0_70
	ldr	r0, [r10, #24]
	cmp	r0, #64
	bne	.LBB0_71
	ldr	r0, [lr, #32]
	ldr	r1, [lr, #36]
	orrs	r0, r0, r1
	bne	.LBB0_72
	ldr	r0, [lr, #4]
	cmp	r0, #1
	bne	.LBB0_73
	ldr	r0, [lr, #8]
	ldr	r3, [sp, #8]
	cmp	r3, r0
	bne	.LBB0_74
	ldr	r0, [r12, #12]
	cmp	r0, #4
	bne	.LBB0_76
	ldrh	r0, [r12, #18]
	cmp	r0, #1
	ldrbeq	r0, [r12, #17]
	cmpeq	r0, #16
	beq	.LBB0_50
.LBB0_48:
	ldr	r0, .LCPI0_24
.LPC0_23:
	add	r0, pc, r0
.LBB0_49:
	ldr	r1, .LCPI0_2
.LPC0_1:
	ldr	r1, [pc, r1]
	ldr	r1, [r1]
	mov	lr, pc
	mov	pc, r1
	mvn	r0, #0
	add	sp, sp, #12
	pop	{r4, r5, r6, r7, r8, r9, r10, r11, lr}
	mov	pc, lr
.LBB0_50:
	ldrb	r0, [r12, #16]
	cmp	r0, #1
	bne	.LBB0_48
	ldr	r0, [r8]
	cmp	r0, #1
	bne	.LBB0_77
	ldr	r0, [r8, #8]
	cmp	r0, #56
	bne	.LBB0_78
	ldr	r0, [r8, #16]
	cmp	r0, #56
	bne	.LBB0_79
	ldr	r0, [r8, #24]
	cmp	r0, #64
	bne	.LBB0_80
	ldr	r0, [r12, #32]
	ldr	r1, [r12, #36]
	orrs	r0, r0, r1
	bne	.LBB0_81
	ldr	r0, [r12, #4]
	cmp	r0, #1
	bne	.LBB0_82
	ldr	r0, [r12, #8]
	cmp	r3, r0
	bne	.LBB0_83
	ldm	sp, {r0, r1}
	add	sp, sp, #12
	pop	{r4, r5, r6, r7, r8, r9, r10, r11, lr}
	b	.Ldefault_function_compute_
.LBB0_59:
	ldr	r0, .LCPI0_1
.LPC0_0:
	add	r0, pc, r0
	b	.LBB0_49
.LBB0_60:
	ldr	r0, .LCPI0_6
.LPC0_5:
	add	r0, pc, r0
	b	.LBB0_49
.LBB0_61:
	ldr	r0, .LCPI0_7
.LPC0_6:
	add	r0, pc, r0
	b	.LBB0_49
.LBB0_62:
	ldr	r0, .LCPI0_9
.LPC0_8:
	add	r0, pc, r0
	b	.LBB0_49
.LBB0_63:
	ldr	r0, .LCPI0_10
.LPC0_9:
	add	r0, pc, r0
	b	.LBB0_49
.LBB0_64:
	ldr	r0, .LCPI0_11
.LPC0_10:
	add	r0, pc, r0
	b	.LBB0_49
.LBB0_65:
	ldr	r0, .LCPI0_12
.LPC0_11:
	add	r0, pc, r0
	b	.LBB0_49
.LBB0_66:
	ldr	r0, .LCPI0_13
.LPC0_12:
	add	r0, pc, r0
	b	.LBB0_49
.LBB0_67:
	ldr	r0, .LCPI0_14
.LPC0_13:
	add	r0, pc, r0
	b	.LBB0_49
.LBB0_68:
	ldr	r0, .LCPI0_16
.LPC0_15:
	add	r0, pc, r0
	b	.LBB0_49
.LBB0_69:
	ldr	r0, .LCPI0_17
.LPC0_16:
	add	r0, pc, r0
	b	.LBB0_49
.LBB0_70:
	ldr	r0, .LCPI0_18
.LPC0_17:
	add	r0, pc, r0
	b	.LBB0_49
.LBB0_71:
	ldr	r0, .LCPI0_19
.LPC0_18:
	add	r0, pc, r0
	b	.LBB0_49
.LBB0_72:
	ldr	r0, .LCPI0_20
.LPC0_19:
	add	r0, pc, r0
	b	.LBB0_49
.LBB0_73:
	ldr	r0, .LCPI0_21
.LPC0_20:
	add	r0, pc, r0
	b	.LBB0_49
.LBB0_74:
	ldr	r0, .LCPI0_22
.LPC0_21:
	add	r0, pc, r0
	b	.LBB0_49
	.p2align	3
.LCPI0_35:
	.long	64
	.long	1
	.p2align	2
.LBB0_76:
	ldr	r0, .LCPI0_23
.LPC0_22:
	add	r0, pc, r0
	b	.LBB0_49
.LBB0_77:
	ldr	r0, .LCPI0_25
.LPC0_24:
	add	r0, pc, r0
	b	.LBB0_49
.LBB0_78:
	ldr	r0, .LCPI0_26
.LPC0_25:
	add	r0, pc, r0
	b	.LBB0_49
.LBB0_79:
	ldr	r0, .LCPI0_27
.LPC0_26:
	add	r0, pc, r0
	b	.LBB0_49
.LBB0_80:
	ldr	r0, .LCPI0_28
.LPC0_27:
	add	r0, pc, r0
	b	.LBB0_49
.LBB0_81:
	ldr	r0, .LCPI0_29
.LPC0_28:
	add	r0, pc, r0
	b	.LBB0_49
.LBB0_82:
	ldr	r0, .LCPI0_30
.LPC0_29:
	add	r0, pc, r0
	b	.LBB0_49
.LBB0_83:
	ldr	r0, .LCPI0_31
.LPC0_30:
	add	r0, pc, r0
	b	.LBB0_49
	.p2align	3
.LCPI0_0:
	.long	64
	.long	1
.LCPI0_1:
	.long	.L.str-(.LPC0_0+8)
.LCPI0_2:
.Ltmp0:
	.long	__TVMAPISetLastError(GOT_PREL)-((.LPC0_1+8)-.Ltmp0)
.LCPI0_3:
	.long	.L.str.4-(.LPC0_2+8)
.LCPI0_4:
	.long	.L.str.5-(.LPC0_3+8)
.LCPI0_5:
	.long	.L.str.6-(.LPC0_4+8)
.LCPI0_6:
	.long	.L.str.7-(.LPC0_5+8)
.LCPI0_7:
	.long	.L.str.8-(.LPC0_6+8)
.LCPI0_8:
	.long	.L.str.9-(.LPC0_7+8)
.LCPI0_9:
	.long	.L.str.10-(.LPC0_8+8)
.LCPI0_10:
	.long	.L.str.11-(.LPC0_9+8)
.LCPI0_11:
	.long	.L.str.12-(.LPC0_10+8)
.LCPI0_12:
	.long	.L.str.13-(.LPC0_11+8)
.LCPI0_13:
	.long	.L.str.14-(.LPC0_12+8)
.LCPI0_14:
	.long	.L.str.15-(.LPC0_13+8)
.LCPI0_15:
	.long	.L.str.16-(.LPC0_14+8)
.LCPI0_16:
	.long	.L.str.17-(.LPC0_15+8)
.LCPI0_17:
	.long	.L.str.18-(.LPC0_16+8)
.LCPI0_18:
	.long	.L.str.19-(.LPC0_17+8)
.LCPI0_19:
	.long	.L.str.20-(.LPC0_18+8)
.LCPI0_20:
	.long	.L.str.21-(.LPC0_19+8)
.LCPI0_21:
	.long	.L.str.22-(.LPC0_20+8)
.LCPI0_22:
	.long	.L.str.23-(.LPC0_21+8)
.LCPI0_23:
	.long	.L.str.24-(.LPC0_22+8)
.LCPI0_24:
	.long	.L.str.25-(.LPC0_23+8)
.LCPI0_25:
	.long	.L.str.26-(.LPC0_24+8)
.LCPI0_26:
	.long	.L.str.27-(.LPC0_25+8)
.LCPI0_27:
	.long	.L.str.28-(.LPC0_26+8)
.LCPI0_28:
	.long	.L.str.29-(.LPC0_27+8)
.LCPI0_29:
	.long	.L.str.30-(.LPC0_28+8)
.LCPI0_30:
	.long	.L.str.31-(.LPC0_29+8)
.LCPI0_31:
	.long	.L.str.32-(.LPC0_30+8)
.LCPI0_32:
	.long	.L.str.1-(.LPC0_31+8)
.LCPI0_33:
	.long	.L.str.2-(.LPC0_32+8)
.LCPI0_34:
	.long	.L.str.3-(.LPC0_33+8)
.Lfunc_end0:
	.size	default_function, .Lfunc_end0-default_function
	.fnend

	.p2align	4
	.type	.Ldefault_function_compute_,%function
	.code	32
.Ldefault_function_compute_:
	.fnstart
	push	{r4, r5, r6, r7, r8, r9, r10, r11, lr}
	add	r11, sp, #28
	sub	sp, sp, #4
	vpush	{d8, d9, d10, d11, d12, d13}
	sub	sp, sp, #152
	str	r2, [r11, #-220]
	mov	r6, r3
	str	r0, [r11, #-188]
	mov	r7, r1
	ldr	r4, .LCPI1_16
.LPC1_0:
	ldr	r4, [pc, r4]
	ldr	r5, [r4]
	sub	sp, sp, #8
	mov	r2, #2336
	mov	r0, #0
	orr	r2, r2, #24576
	mov	r8, #8
	mov	r9, #1
	str	r0, [r11, #-200]
	mov	r1, r3
	mov	r0, #1
	mov	r3, #0
	str	r9, [sp]
	str	r8, [sp, #4]
	mov	lr, pc
	mov	pc, r5
	add	sp, sp, #8
	ldr	r5, [r4]
	mov	r10, r0
	sub	sp, sp, #8
	mov	r0, #1
	mov	r1, r6
	mov	r2, #4608
	mov	r3, #0
	str	r9, [sp]
	str	r8, [sp, #4]
	mov	lr, pc
	mov	pc, r5
	add	sp, sp, #8
	str	r0, [r11, #-216]
	ldr	r4, [r4]
	sub	sp, sp, #8
	mov	r2, #29184
	mov	r0, #1
	orr	r2, r2, #196608
	mov	r1, r6
	mov	r3, #0
	str	r9, [sp]
	str	r8, [sp, #4]
	ldr	r5, [r11, #-188]
	str	r6, [r11, #-228]
	mov	lr, pc
	mov	pc, r4
	add	sp, sp, #8
	vmov.i16	d16, #0x2
	add	r1, r10, #1024
	vmov.i16	d17, #0x1
	str	r0, [r11, #-224]
	add	r0, r10, #512
	str	r0, [r11, #-192]
	str	r10, [r11, #-212]
	str	r7, [r11, #-208]
.LBB1_1:
	str	r1, [r11, #-196]
	mov	r0, r10
	mov	r1, r5
	mov	r2, #0
	str	r10, [r11, #-204]
	str	r5, [r11, #-188]
.LBB1_2:
	sub	r9, r11, #104
	sub	r7, r11, #100
	sub	r4, r11, #96
	sub	r5, r11, #92
	sub	r12, r11, #88
	mov	r6, #0
	str	r2, [r11, #-184]
.LBB1_3:
	mov	r2, r1
	ldr	r3, [r2, r6]!
	str	r3, [r11, #-116]
	sub	r3, r11, #116
	ldr	lr, [r2, #128]
	vld1.32	{d18[0]}, [r3:32]
	ldr	r3, [r2, #64]
	vmovl.u8	q10, d18
	ldr	r10, [r2, #192]
	ldr	r8, [r2, #256]
	str	r3, [r11, #-112]
	sub	r3, r11, #112
	vshl.i16	d20, d20, #1
	vld1.32	{d19[0]}, [r3:32]
	sub	r3, r11, #108
	vand	d20, d20, d16
	vmovl.u8	q11, d19
	str	lr, [r11, #-108]
	vld1.32	{d18[0]}, [r3:32]
	ldr	r3, [r2, #320]
	vand	d19, d22, d17
	vorr	d19, d20, d19
	vmovl.u8	q10, d18
	str	r10, [r11, #-104]
	str	r8, [r11, #-100]
	str	r3, [r11, #-96]
	vshl.i16	d19, d19, #1
	vld1.32	{d18[0]}, [r9:32]
	vand	d20, d20, d17
	vorr	d19, d19, d20
	vmovl.u8	q11, d18
	vld1.32	{d21[0]}, [r7:32]
	vshl.i16	d19, d19, #1
	vand	d18, d22, d17
	vld1.32	{d20[0]}, [r4:32]
	vmovl.u8	q11, d21
	vorr	d18, d19, d18
	vmovl.u8	q10, d20
	vand	d19, d22, d17
	ldr	r3, [r2, #384]
	vshl.i16	d18, d18, #1
	str	r3, [r11, #-92]
	vand	d20, d20, d17
	ldr	r2, [r2, #448]
	vorr	d18, d18, d19
	vld1.32	{d19[0]}, [r5:32]
	vshl.i16	d18, d18, #1
	str	r2, [r11, #-88]
	add	r2, r0, r6
	vorr	d18, d18, d20
	add	r6, r6, #4
	vmovl.u8	q10, d19
	cmp	r6, #64
	vshl.i16	d18, d18, #1
	vand	d19, d20, d17
	vld1.32	{d20[0]}, [r12:32]
	vorr	d18, d18, d19
	vmovl.u8	q10, d20
	vshl.i16	d18, d18, #1
	vand	d19, d20, d17
	vorr	d18, d18, d19
	vuzp.8	d18, d19
	vst1.32	{d18[0]}, [r2]
	bne	.LBB1_3
	ldr	r2, [r11, #-184]
	add	r0, r0, #64
	add	r1, r1, #512
	add	r2, r2, #1
	cmp	r2, #8
	bne	.LBB1_2
	ldr	r1, [r11, #-192]
	mov	r0, #0
	ldr	r6, [r11, #-188]
.LBB1_6:
	sub	r12, r11, #128
	sub	r7, r11, #124
	sub	r10, r11, #120
	mov	r2, #0
	mov	r9, #4544
	mov	r8, #4160
	mov	lr, #4096
	str	r0, [r11, #-184]
.LBB1_7:
	add	r3, r6, r2
	sub	r0, r11, #144
	ldr	r5, [r3, r8]
	ldr	r4, [r3, lr]
	str	r5, [r11, #-144]
	vld1.32	{d18[0]}, [r0:32]
	mov	r0, #4224
	vmovl.u8	q11, d18
	ldr	r5, [r3, r0]
	sub	r0, r11, #148
	str	r4, [r11, #-148]
	vld1.32	{d19[0]}, [r0:32]
	sub	r0, r11, #140
	vmovl.u8	q9, d19
	str	r5, [r11, #-140]
	vld1.32	{d20[0]}, [r0:32]
	mov	r0, #4288
	vshl.i16	d18, d18, #1
	vand	d19, d22, d17
	vand	d18, d18, d16
	ldr	r5, [r3, r0]
	mov	r0, #4352
	ldr	r4, [r3, r0]
	vorr	d18, d18, d19
	vmovl.u8	q11, d20
	sub	r0, r11, #136
	str	r5, [r11, #-136]
	vshl.i16	d18, d18, #1
	vld1.32	{d21[0]}, [r0:32]
	mov	r0, #4416
	vand	d19, d22, d17
	vmovl.u8	q10, d21
	ldr	r5, [r3, r0]
	mov	r0, #4480
	vorr	d18, d18, d19
	ldr	r0, [r3, r0]
	vand	d20, d20, d17
	str	r4, [r11, #-132]
	sub	r4, r11, #132
	vshl.i16	d18, d18, #1
	vld1.32	{d19[0]}, [r4:32]
	vorr	d18, d18, d20
	vmovl.u8	q10, d19
	str	r5, [r11, #-128]
	vshl.i16	d18, d18, #1
	str	r0, [r11, #-124]
	vld1.32	{d19[0]}, [r12:32]
	vand	d20, d20, d17
	vorr	d18, d18, d20
	vld1.32	{d20[0]}, [r7:32]
	vmovl.u8	q11, d19
	vshl.i16	d18, d18, #1
	vand	d19, d22, d17
	vmovl.u8	q10, d20
	ldr	r0, [r3, r9]
	vorr	d18, d18, d19
	str	r0, [r11, #-120]
	add	r0, r1, r2
	vand	d19, d20, d17
	add	r2, r2, #4
	vld1.32	{d20[0]}, [r10:32]
	vshl.i16	d18, d18, #1
	cmp	r2, #64
	vorr	d18, d18, d19
	vmovl.u8	q10, d20
	vshl.i16	d18, d18, #1
	vand	d19, d20, d17
	vorr	d18, d18, d19
	vuzp.8	d18, d19
	vst1.32	{d18[0]}, [r0]
	bne	.LBB1_7
	ldr	r0, [r11, #-184]
	add	r1, r1, #64
	add	r6, r6, #512
	add	r0, r0, #1
	cmp	r0, #8
	bne	.LBB1_6
	ldr	r1, [r11, #-196]
	mov	r0, #0
	ldr	r6, [r11, #-188]
.LBB1_10:
	sub	lr, r11, #152
	sub	r12, r11, #176
	mov	r2, #0
	mov	r10, #8640
	mov	r7, #8256
	mov	r8, #8192
	mov	r9, #8320
	str	r0, [r11, #-184]
.LBB1_11:
	add	r3, r6, r2
	sub	r0, r11, #180
	ldr	r5, [r3, r7]
	ldr	r4, [r3, r8]
	str	r5, [r11, #-176]
	vld1.32	{d18[0]}, [r12:32]
	ldr	r5, [r3, r9]
	vmovl.u8	q11, d18
	str	r4, [r11, #-180]
	vld1.32	{d19[0]}, [r0:32]
	sub	r0, r11, #172
	vmovl.u8	q9, d19
	str	r5, [r11, #-172]
	vld1.32	{d20[0]}, [r0:32]
	mov	r0, #8384
	vshl.i16	d18, d18, #1
	vand	d19, d22, d17
	vand	d18, d18, d16
	ldr	r5, [r3, r0]
	mov	r0, #8448
	ldr	r4, [r3, r0]
	vorr	d18, d18, d19
	vmovl.u8	q11, d20
	sub	r0, r11, #168
	str	r5, [r11, #-168]
	vshl.i16	d18, d18, #1
	vld1.32	{d21[0]}, [r0:32]
	mov	r0, #8512
	vand	d19, d22, d17
	vmovl.u8	q10, d21
	ldr	r5, [r3, r0]
	mov	r0, #8576
	vorr	d18, d18, d19
	ldr	r0, [r3, r0]
	vand	d20, d20, d17
	str	r4, [r11, #-164]
	sub	r4, r11, #164
	vshl.i16	d18, d18, #1
	vld1.32	{d19[0]}, [r4:32]
	vorr	d18, d18, d20
	vmovl.u8	q10, d19
	str	r5, [r11, #-160]
	vshl.i16	d18, d18, #1
	str	r0, [r11, #-156]
	sub	r0, r11, #160
	vand	d20, d20, d17
	vld1.32	{d19[0]}, [r0:32]
	sub	r0, r11, #156
	vorr	d18, d18, d20
	vld1.32	{d20[0]}, [r0:32]
	vmovl.u8	q11, d19
	vshl.i16	d18, d18, #1
	vand	d19, d22, d17
	vmovl.u8	q10, d20
	ldr	r0, [r3, r10]
	vorr	d18, d18, d19
	str	r0, [r11, #-152]
	add	r0, r1, r2
	vand	d19, d20, d17
	add	r2, r2, #4
	vld1.32	{d20[0]}, [lr:32]
	vshl.i16	d18, d18, #1
	cmp	r2, #64
	vorr	d18, d18, d19
	vmovl.u8	q10, d20
	vshl.i16	d18, d18, #1
	vand	d19, d20, d17
	vorr	d18, d18, d19
	vuzp.8	d18, d19
	vst1.32	{d18[0]}, [r0]
	bne	.LBB1_11
	ldr	r0, [r11, #-184]
	add	r1, r1, #64
	add	r6, r6, #512
	add	r0, r0, #1
	cmp	r0, #8
	bne	.LBB1_10
	ldr	r0, [r11, #-192]
	ldr	r1, [r11, #-196]
	add	r0, r0, #1536
	ldr	r10, [r11, #-204]
	ldr	r5, [r11, #-188]
	add	r1, r1, #1536
	str	r0, [r11, #-192]
	add	r10, r10, #1536
	ldr	r0, [r11, #-200]
	add	r5, r5, #12288
	ldr	r7, [r11, #-208]
	add	r0, r0, #1
	str	r0, [r11, #-200]
	cmp	r0, #3
	bne	.LBB1_1
	ldr	r0, .LCPI1_17
	mov	r2, sp
	sub	r1, r2, #8
.LPC1_1:
	add	r0, pc, r0
	mov	sp, r1
	ldr	r3, [r11, #-216]
	mov	r10, #0
	ldr	r4, [r11, #-212]
	str	r3, [r2, #-8]
	str	r4, [r2, #-4]
	ldr	r2, .LCPI1_18
.LPC1_2:
	ldr	r2, [pc, r2]
	ldr	r3, [r2]
	mov	r2, #0
	mov	lr, pc
	mov	pc, r3
	cmp	r0, #0
	bne	.LBB1_40
	vmov.i16	d16, #0x1
	add	r9, r4, #4
	vmov.i16	d17, #0x2
	mov	r0, #0
.LBB1_16:
	mvn	lr, #0
	str	r0, [r11, #-196]
	sub	r0, r0, #1
	str	r0, [r11, #-184]
	str	r9, [r11, #-192]
	str	r10, [r11, #-188]
.LBB1_17:
	ldr	r0, [r11, #-184]
	cmp	r0, #55
	cmpls	lr, #55
	bhi	.LBB1_35
	adr	r1, .LCPI1_23
	vdup.32	q9, r10
	vld1.64	{d20, d21}, [r1:128]
	adr	r1, .LCPI1_24
	vshl.i32	q9, q9, #3
	vld1.64	{d22, d23}, [r1:128]
	adr	r1, .LCPI1_25
	vadd.i32	q13, q9, q10
	vld1.64	{d24, d25}, [r1:128]
	adr	r1, .LCPI1_26
	vadd.i32	q4, q9, q11
	vld1.64	{d28, d29}, [r1:128]
	adr	r1, .LCPI1_27
	vadd.i32	q11, q9, q12
	vld1.64	{d30, d31}, [r1:128]
	adr	r1, .LCPI1_28
	vadd.i32	q5, q9, q15
	vld1.64	{d0, d1}, [r1:128]
	adr	r1, .LCPI1_29
	vadd.i32	q10, q9, q0
	vld1.64	{d0, d1}, [r1:128]
	adr	r1, .LCPI1_30
	vadd.i32	q3, q9, q0
	vmov.32	r5, d10[0]
	vadd.i32	q14, q9, q14
	vld1.64	{d24, d25}, [r1:128]
	vadd.i32	q0, q9, q12
	vmov.32	r2, d6[0]
	vmov.32	r0, d22[0]
	vmov.32	r1, d0[0]
	vmov.32	r6, d0[1]
	vmov.32	r4, d8[0]
	vmov.32	r12, d6[1]
	vmov.32	r3, d11[0]
	ldrb	r2, [r7, r2]
	ldrb	r1, [r7, r1]
	vmov.16	d30[0], r2
	ldrb	r2, [r7, r5]
	ldrb	r4, [r7, r4]
	vmov.32	r5, d10[1]
	vmov.16	d3[0], r1
	ldrb	r1, [r7, r0]
	ldrb	r3, [r7, r3]
	vmov.32	r0, d22[1]
	vmov.16	d25[0], r2
	ldrb	r2, [r7, r6]
	vmov.32	r6, d8[1]
	vmov.16	d31[0], r4
	ldrb	r4, [r7, r12]
	vmov.16	d24[0], r1
	vmov.16	d3[1], r2
	vmov.32	r2, d26[0]
	vmov.16	d30[1], r4
	vmov.32	r4, d1[0]
	ldrb	r1, [r7, r5]
	vmov.32	r5, d7[0]
	ldrb	r0, [r7, r0]
	vmov.16	d25[1], r1
	ldrb	r1, [r7, r6]
	vmov.32	r6, d20[0]
	vmov.16	d24[1], r0
	vmov.32	r0, d28[0]
	vmov.16	d31[1], r1
	vmov.32	r1, d9[0]
	vmov.16	d25[2], r3
	ldrb	r2, [r7, r2]
	vmov.32	r3, d23[0]
	vmov.16	d5[0], r2
	ldrb	r2, [r7, r4]
	vmov.32	r4, d9[1]
	ldrb	r5, [r7, r5]
	vmov.16	d3[2], r2
	vmov.16	d30[2], r5
	vmov.32	r5, d26[1]
	ldrb	r6, [r7, r6]
	ldrb	r0, [r7, r0]
	vmov.16	d2[0], r6
	ldrb	r1, [r7, r1]
	vmov.32	r6, d20[1]
	vmov.16	d4[0], r0
	vmov.32	r0, d11[1]
	vmov.16	d31[2], r1
	ldrb	r1, [r7, r3]
	adr	r2, .LCPI1_31
	vld1.64	{d8, d9}, [r2:128]
	vmov.16	d24[2], r1
	ldrb	r1, [r7, r4]
	vmov.32	r4, d7[1]
	vadd.i32	q3, q9, q4
	vmov.32	r2, d29[0]
	vmov.16	d31[3], r1
	ldrb	r3, [r7, r5]
	vmov.32	r5, d27[1]
	vmov.16	d5[1], r3
	ldrb	r3, [r7, r6]
	vmov.32	r6, d28[1]
	ldrb	r1, [r7, r0]
	adr	r0, .LCPI1_32
	vmov.16	d2[1], r3
	vld1.64	{d10, d11}, [r0:128]
	vmov.16	d25[3], r1
	ldrb	r1, [r7, r4]
	ldrb	r12, [r7, r2]
	vmov.32	r2, d21[0]
	ldrb	r3, [r7, r5]
	adr	r4, .LCPI1_33
	vmov.32	r5, d27[0]
	vld1.64	{d12, d13}, [r4:128]
	vmov.32	r4, d1[1]
	vmov.16	d30[3], r1
	ldrb	r6, [r7, r6]
	vmov.16	d4[1], r6
	vmov.32	r6, d29[1]
	vmov.16	d4[2], r12
	ldrb	r2, [r7, r2]
	ldrb	r5, [r7, r5]
	adr	r1, .LCPI1_34
	vld1.64	{d28, d29}, [r1:128]
	vadd.i32	q14, q9, q14
	ldrb	r1, [r7, r4]
	vmov.32	r4, d23[1]
	vmov.16	d5[2], r5
	vmov.16	d2[2], r2
	vmov.16	d3[3], r1
	vmov.16	d5[3], r3
	ldrb	r1, [r7, r6]
	vmov.32	r3, d28[0]
	vmov.16	d4[3], r1
	ldrb	r6, [r7, r4]
	vmov.32	r4, d21[1]
	vshl.i16	d20, d3, #1
	vand	d21, d30, d16
	adr	r0, .LCPI1_35
	vand	d20, d20, d17
	vld1.64	{d22, d23}, [r0:128]
	vadd.i32	q13, q9, q11
	vorr	d20, d20, d21
	vmov.16	d24[3], r6
	vand	d21, d25, d16
	sub	r6, r9, #4
	vand	d24, d24, d16
	vshl.i16	d20, d20, #1
	vadd.i32	q11, q9, q6
	vorr	d20, d20, d21
	vand	d21, d31, d16
	vshl.i16	d20, d20, #1
	vorr	d25, d20, d21
	vadd.i32	q10, q9, q5
	vshl.i16	d25, d25, #1
	ldrb	r0, [r7, r4]
	adr	r2, .LCPI1_36
	vorr	d24, d25, d24
	vld1.64	{d30, d31}, [r2:128]
	vand	d25, d5, d16
	vadd.i32	q15, q9, q15
	vshl.i16	d24, d24, #1
	vmov.16	d2[3], r0
	adr	r0, .LCPI1_37
	vmov.32	r1, d30[0]
	vorr	d24, d24, d25
	vand	d25, d4, d16
	vld1.64	{d0, d1}, [r0:128]
	vadd.i32	q2, q9, q0
	vshl.i16	d24, d24, #1
	vmov.32	r5, d30[1]
	vorr	d24, d24, d25
	vmov.32	r0, d4[0]
	vand	d25, d2, d16
	vmov.32	r2, d4[1]
	vshl.i16	d24, d24, #1
	vorr	d24, d24, d25
	vuzp.8	d24, d25
	vst1.32	{d24[0]}, [r6]
	vmov.32	r6, d6[0]
	ldrb	r1, [r7, r1]
	vmov.16	d0[0], r1
	ldrb	r1, [r7, r3]
	ldrb	r0, [r7, r0]
	vmov.32	r3, d26[0]
	vmov.16	d1[0], r1
	vmov.16	d24[0], r0
	ldrb	r0, [r7, r2]
	vmov.32	r2, d28[1]
	vmov.16	d24[1], r0
	ldrb	r1, [r7, r6]
	vmov.32	r6, d6[1]
	vmov.16	d25[0], r1
	ldrb	r1, [r7, r5]
	vmov.32	r5, d26[1]
	vmov.16	d0[1], r1
	ldrb	r1, [r7, r3]
	vmov.32	r3, d5[0]
	vmov.16	d2[0], r1
	ldrb	r0, [r7, r2]
	vmov.32	r2, d7[0]
	ldrb	r1, [r7, r6]
	vmov.32	r6, d20[0]
	vmov.16	d1[1], r0
	vmov.32	r0, d20[1]
	vmov.16	d25[1], r1
	ldrb	r1, [r7, r5]
	adr	r5, .LCPI1_15
	vld1.64	{d8, d9}, [r5:128]
	vmov.32	r5, d7[1]
	vadd.i32	q3, q9, q4
	vmov.16	d2[1], r1
	ldrb	r1, [r7, r3]
	vmov.32	r3, d31[0]
	vmov.16	d24[2], r1
	ldrb	r2, [r7, r2]
	ldrb	r1, [r7, r6]
	vmov.16	d25[2], r2
	vmov.32	r2, d29[0]
	vmov.32	r6, d22[0]
	ldrb	r0, [r7, r0]
	vmov.16	d3[0], r1
	vmov.16	d3[1], r0
	ldrb	r5, [r7, r5]
	ldrb	r1, [r7, r3]
	vmov.32	r3, d27[0]
	vmov.16	d25[3], r5
	vmov.32	r5, d23[0]
	vmov.16	d0[2], r1
	ldrb	r0, [r7, r2]
	vmov.32	r2, d6[0]
	ldrb	r1, [r7, r6]
	vmov.32	r6, d27[1]
	vmov.16	d1[2], r0
	vmov.16	d18[0], r1
	vmov.32	r0, d29[1]
	ldrb	r1, [r7, r3]
	vmov.32	r3, d5[1]
	vmov.16	d2[2], r1
	ldrb	r1, [r7, r2]
	ldrb	r2, [r7, r6]
	vmov.32	r6, d21[0]
	vmov.16	d2[3], r2
	ldrb	r0, [r7, r0]
	ldrb	r2, [r7, r3]
	vmov.32	r3, d31[1]
	vmov.16	d1[3], r0
	vmov.16	d24[3], r2
	b	.LBB1_34
	.p2align	4
.LCPI1_23:
	.long	4294963653
	.long	4294963661
	.long	4294963669
	.long	4294963677
	.p2align	4
.LCPI1_24:
	.long	4294963651
	.long	4294963659
	.long	4294963667
	.long	4294963675
	.p2align	4
.LCPI1_25:
	.long	4294963652
	.long	4294963660
	.long	4294963668
	.long	4294963676
	.p2align	4
.LCPI1_26:
	.long	4294963654
	.long	4294963662
	.long	4294963670
	.long	4294963678
	.p2align	4
.LCPI1_27:
	.long	4294963650
	.long	4294963658
	.long	4294963666
	.long	4294963674
	.p2align	4
.LCPI1_28:
	.long	4294963655
	.long	4294963663
	.long	4294963671
	.long	4294963679
	.p2align	4
.LCPI1_29:
	.long	4294963649
	.long	4294963657
	.long	4294963665
	.long	4294963673
	.p2align	4
.LCPI1_30:
	.long	4294963648
	.long	4294963656
	.long	4294963664
	.long	4294963672
	.p2align	4
.LCPI1_31:
	.long	4294963683
	.long	4294963691
	.long	4294963699
	.long	4294963707
	.p2align	4
.LCPI1_32:
	.long	4294963685
	.long	4294963693
	.long	4294963701
	.long	4294963709
	.p2align	4
.LCPI1_33:
	.long	4294963686
	.long	4294963694
	.long	4294963702
	.long	4294963710
	.p2align	4
.LCPI1_34:
	.long	4294963681
	.long	4294963689
	.long	4294963697
	.long	4294963705
	.p2align	4
.LCPI1_35:
	.long	4294963682
	.long	4294963690
	.long	4294963698
	.long	4294963706
	.p2align	4
.LCPI1_36:
	.long	4294963680
	.long	4294963688
	.long	4294963696
	.long	4294963704
	.p2align	4
.LCPI1_37:
	.long	4294963684
	.long	4294963692
	.long	4294963700
	.long	4294963708
	.p2align	2
.LBB1_34:
	ldrb	r2, [r7, r6]
	vmov.32	r6, d22[1]
	vmov.16	d3[2], r2
	ldrb	r2, [r7, r3]
	vmov.32	r3, d6[1]
	vmov.16	d0[3], r2
	ldrb	r2, [r7, r5]
	vmov.32	r5, d21[1]
	vand	d20, d1, d16
	vshl.i16	d19, d0, #1
	vmov.16	d21[0], r1
	vand	d19, d19, d17
	vorr	d19, d19, d20
	ldrb	r6, [r7, r6]
	vand	d20, d2, d16
	vshl.i16	d19, d19, #1
	vmov.16	d18[1], r6
	vorr	d19, d19, d20
	vand	d20, d25, d16
	vmov.32	r6, d23[1]
	vshl.i16	d19, d19, #1
	vmov.16	d18[2], r2
	ldrb	r0, [r7, r3]
	vmov.32	r3, d7[0]
	vorr	d19, d19, d20
	vand	d20, d24, d16
	vmov.16	d21[1], r0
	vshl.i16	d19, d19, #1
	ldrb	r1, [r7, r5]
	vmov.32	r5, d7[1]
	vorr	d19, d19, d20
	vmov.16	d3[3], r1
	vshl.i16	d19, d19, #1
	vand	d20, d3, d16
	vorr	d19, d19, d20
	vshl.i16	d19, d19, #1
	ldrb	r1, [r7, r6]
	ldrb	r2, [r7, r3]
	vmov.16	d18[3], r1
	vand	d18, d18, d16
	vmov.16	d21[2], r2
	ldrb	r0, [r7, r5]
	vorr	d18, d19, d18
	vshl.i16	d18, d18, #1
	vmov.16	d21[3], r0
	vand	d19, d21, d16
	vorr	d18, d18, d19
	vuzp.8	d18, d19
	vst1.32	{d18[0]}, [r9]
.LBB1_35:
	add	lr, lr, #1
	add	r9, r9, #8
	add	r10, r10, #8
	cmp	lr, #57
	bne	.LBB1_17
	ldr	r9, [r11, #-192]
	ldr	r10, [r11, #-188]
	ldr	r0, [r11, #-196]
	add	r9, r9, #464
	add	r10, r10, #448
	add	r0, r0, #1
	cmp	r0, #58
	bne	.LBB1_16
	ldr	r0, .LCPI1_19
	mov	r2, sp
	sub	r1, r2, #8
.LPC1_3:
	add	r0, pc, r0
	mov	sp, r1
	ldr	r6, [r11, #-224]
	ldr	r8, [r11, #-212]
	str	r6, [r2, #-8]
	str	r8, [r2, #-4]
	mov	r2, #0
	ldr	r4, .LCPI1_20
.LPC1_4:
	ldr	r4, [pc, r4]
	ldr	r3, [r4]
	mov	lr, pc
	mov	pc, r3
	cmp	r0, #0
	bne	.LBB1_40
	ldr	r0, .LCPI1_21
	mov	r2, sp
	sub	r1, r2, #16
.LPC1_5:
	add	r0, pc, r0
	mov	sp, r1
	ldr	r7, [r11, #-216]
	ldr	r3, [r11, #-220]
	str	r7, [r2, #-16]
	str	r6, [r2, #-12]
	str	r3, [r2, #-8]
	mov	r2, #0
	ldr	r3, [r4]
	mov	lr, pc
	mov	pc, r3
	cmp	r0, #0
	bne	.LBB1_40
	ldr	r4, .LCPI1_22
	mov	r0, #1
	mov	r2, r6
.LPC1_6:
	ldr	r4, [pc, r4]
	ldr	r5, [r11, #-228]
	ldr	r3, [r4]
	mov	r1, r5
	mov	lr, pc
	mov	pc, r3
	ldr	r3, [r4]
	mov	r0, #1
	mov	r1, r5
	mov	r2, r7
	mov	lr, pc
	mov	pc, r3
	ldr	r3, [r4]
	mov	r0, #1
	mov	r1, r5
	mov	r2, r8
	mov	lr, pc
	mov	pc, r3
	mov	r0, #0
.LBB1_40:
	sub	sp, r11, #80
	vpop	{d8, d9, d10, d11, d12, d13}
	add	sp, sp, #4
	pop	{r4, r5, r6, r7, r8, r9, r10, r11, lr}
	mov	pc, lr
	.p2align	4
.LCPI1_15:
	.long	4294963687
	.long	4294963695
	.long	4294963703
	.long	4294963711
.LCPI1_16:
.Ltmp1:
	.long	__TVMBackendAllocWorkspace(GOT_PREL)-((.LPC1_0+8)-.Ltmp1)
.LCPI1_17:
	.long	.L__tvm_parallel_lambda-(.LPC1_1+8)
.LCPI1_18:
.Ltmp2:
	.long	__TVMBackendParallelLaunch(GOT_PREL)-((.LPC1_2+8)-.Ltmp2)
.LCPI1_19:
	.long	.L__tvm_parallel_lambda.34-(.LPC1_3+8)
.LCPI1_20:
.Ltmp3:
	.long	__TVMBackendParallelLaunch(GOT_PREL)-((.LPC1_4+8)-.Ltmp3)
.LCPI1_21:
	.long	.L__tvm_parallel_lambda.35-(.LPC1_5+8)
.LCPI1_22:
.Ltmp4:
	.long	__TVMBackendFreeWorkspace(GOT_PREL)-((.LPC1_6+8)-.Ltmp4)
.Lfunc_end1:
	.size	.Ldefault_function_compute_, .Lfunc_end1-.Ldefault_function_compute_
	.fnend

	.p2align	2
	.type	.L__tvm_parallel_lambda,%function
	.code	32
.L__tvm_parallel_lambda:
	.fnstart
	push	{r4, r5, r6, r7, r8, lr}
	ldr	r1, [r1, #4]
	mov	r5, r0
	mov	r4, r2
	add	r0, r1, #7
	bl	__divsi3
	add	r2, r5, #1
	mul	r1, r0, r5
	mov	lr, #8
	mul	r3, r0, r2
	mov	r12, #8
	cmp	r1, #8
	movlt	lr, r1
	cmp	r3, #8
	movlt	r12, r3
	cmp	lr, r12
	bge	.LBB2_7
	mvn	r1, r1
	cmn	r1, #9
	ldm	r4, {r0, r2}
	mvnle	r1, #8
	mov	r4, #248
	sub	r4, r4, r1, lsl #3
	ldr	r3, .LCPI2_0
	add	r1, r1, r1, lsl #3
	add	r8, r2, r4
	sub	r1, r3, r1, lsl #6
	add	r0, r0, r1
.LBB2_2:
	mov	r5, #0
	mov	r2, r8
	mov	r3, r0
.LBB2_3:
	mov	r1, #55
	mov	r6, r2
.LBB2_4:
	add	r7, r3, r1
	ldrb	r4, [r6, #-256]
	strb	r4, [r7, #-55]
	ldrb	r4, [r6, #-192]
	strb	r4, [r7, #-54]
	ldrb	r4, [r6, #-128]
	strb	r4, [r7, #-53]
	ldrb	r4, [r6, #-64]
	strb	r4, [r7, #-52]
	ldrb	r4, [r6]
	strb	r4, [r7, #-51]
	ldrb	r4, [r6, #64]
	strb	r4, [r7, #-50]
	ldrb	r4, [r6, #128]
	strb	r4, [r7, #-49]
	ldrb	r4, [r6, #192]
	strb	r4, [r7, #-48]
	ldrb	r4, [r6, #-255]
	strb	r4, [r7, #-47]
	ldrb	r4, [r6, #-191]
	strb	r4, [r7, #-46]
	ldrb	r4, [r6, #-127]
	strb	r4, [r7, #-45]
	ldrb	r4, [r6, #-63]
	strb	r4, [r7, #-44]
	ldrb	r4, [r6, #1]
	strb	r4, [r7, #-43]
	ldrb	r4, [r6, #65]
	strb	r4, [r7, #-42]
	ldrb	r4, [r6, #129]
	strb	r4, [r7, #-41]
	ldrb	r4, [r6, #193]
	strb	r4, [r7, #-40]
	ldrb	r4, [r6, #-254]
	strb	r4, [r7, #-39]
	ldrb	r4, [r6, #-190]
	strb	r4, [r7, #-38]
	ldrb	r4, [r6, #-126]
	strb	r4, [r7, #-37]
	ldrb	r4, [r6, #-62]
	strb	r4, [r7, #-36]
	ldrb	r4, [r6, #2]
	strb	r4, [r7, #-35]
	ldrb	r4, [r6, #66]
	strb	r4, [r7, #-34]
	ldrb	r4, [r6, #130]
	strb	r4, [r7, #-33]
	ldrb	r4, [r6, #194]
	strb	r4, [r7, #-32]
	ldrb	r4, [r6, #-253]
	strb	r4, [r7, #-31]
	ldrb	r4, [r6, #-189]
	strb	r4, [r7, #-30]
	ldrb	r4, [r6, #-125]
	strb	r4, [r7, #-29]
	ldrb	r4, [r6, #-61]
	strb	r4, [r7, #-28]
	ldrb	r4, [r6, #3]
	strb	r4, [r7, #-27]
	ldrb	r4, [r6, #67]
	strb	r4, [r7, #-26]
	ldrb	r4, [r6, #131]
	strb	r4, [r7, #-25]
	ldrb	r4, [r6, #195]
	strb	r4, [r7, #-24]
	ldrb	r4, [r6, #-252]
	strb	r4, [r7, #-23]
	ldrb	r4, [r6, #-188]
	strb	r4, [r7, #-22]
	ldrb	r4, [r6, #-124]
	strb	r4, [r7, #-21]
	ldrb	r4, [r6, #-60]
	strb	r4, [r7, #-20]
	ldrb	r4, [r6, #4]
	strb	r4, [r7, #-19]
	ldrb	r4, [r6, #68]
	strb	r4, [r7, #-18]
	ldrb	r4, [r6, #132]
	strb	r4, [r7, #-17]
	ldrb	r4, [r6, #196]
	strb	r4, [r7, #-16]
	ldrb	r4, [r6, #-251]
	strb	r4, [r7, #-15]
	ldrb	r4, [r6, #-187]
	strb	r4, [r7, #-14]
	ldrb	r4, [r6, #-123]
	strb	r4, [r7, #-13]
	ldrb	r4, [r6, #-59]
	strb	r4, [r7, #-12]
	ldrb	r4, [r6, #5]
	strb	r4, [r7, #-11]
	ldrb	r4, [r6, #69]
	strb	r4, [r7, #-10]
	ldrb	r4, [r6, #133]
	strb	r4, [r7, #-9]
	ldrb	r4, [r6, #197]
	strb	r4, [r7, #-8]
	ldrb	r4, [r6, #-250]
	strb	r4, [r7, #-7]
	ldrb	r4, [r6, #-186]
	strb	r4, [r7, #-6]
	ldrb	r4, [r6, #-122]
	strb	r4, [r7, #-5]
	ldrb	r4, [r6, #-58]
	strb	r4, [r7, #-4]
	ldrb	r4, [r6, #6]
	strb	r4, [r7, #-3]
	ldrb	r4, [r6, #70]
	strb	r4, [r7, #-2]
	ldrb	r4, [r6, #134]
	strb	r4, [r7, #-1]
	ldrb	r4, [r6, #198]
	strb	r4, [r3, r1]
	add	r1, r1, #64
	cmp	r1, #247
	ldrb	r4, [r6, #-249]
	strb	r4, [r7, #1]
	ldrb	r4, [r6, #-185]
	strb	r4, [r7, #2]
	ldrb	r4, [r6, #-121]
	strb	r4, [r7, #3]
	ldrb	r4, [r6, #-57]
	strb	r4, [r7, #4]
	ldrb	r4, [r6, #7]
	strb	r4, [r7, #5]
	ldrb	r4, [r6, #71]
	strb	r4, [r7, #6]
	ldrb	r4, [r6, #135]
	strb	r4, [r7, #7]
	ldrb	r4, [r6, #199]
	add	r6, r6, #512
	strb	r4, [r7, #8]
	bne	.LBB2_4
	add	r5, r5, #1
	add	r2, r2, #1536
	add	r3, r3, #192
	cmp	r5, #3
	bne	.LBB2_3
	add	lr, lr, #1
	add	r8, r8, #8
	add	r0, r0, #576
	cmp	lr, r12
	blt	.LBB2_2
.LBB2_7:
	mov	r0, #0
	pop	{r4, r5, r6, r7, r8, lr}
	mov	pc, lr
	.p2align	2
.LCPI2_0:
	.long	4294966720
.Lfunc_end2:
	.size	.L__tvm_parallel_lambda, .Lfunc_end2-.L__tvm_parallel_lambda
	.cantunwind
	.fnend

	.p2align	2
	.type	.L__tvm_parallel_lambda.34,%function
	.code	32
.L__tvm_parallel_lambda.34:
	.fnstart
	push	{r4, r5, r6, r7, r8, r9, r10, r11, lr}
	sub	sp, sp, #28
	ldr	r1, [r1, #4]
	mov	r5, r0
	mov	r4, r2
	add	r0, r1, #55
	bl	__divsi3
	add	r1, r5, #1
	mul	r3, r0, r5
	mul	r2, r0, r1
	mov	r1, #56
	mov	r0, #56
	cmp	r3, #56
	movlt	r1, r3
	cmp	r2, #56
	movlt	r0, r2
	cmp	r1, r0
	str	r0, [sp, #4]
	bge	.LBB3_15
	mvn	r3, r3
	cmn	r3, #57
	mvnle	r3, #56
	mov	r6, #464
	mul	r5, r3, r6
	sub	r6, r3, r3, lsl #6
	ldr	r10, .LCPI3_0
	ldm	r4, {r0, r7}
	rsb	r3, r3, r3, lsl #6
	vmov.i32	q8, #0x0
	add	r4, r0, r6, lsl #6
	add	r6, r10, #16
	sub	r6, r6, r3, lsl #6
	sub	r3, r10, r3, lsl #6
	add	r2, r0, r6
	add	r6, r0, r3
	add	r0, r10, #3568
	mov	r3, #0
	sub	r0, r0, r5
	mov	r8, #20
	add	r11, r7, r0
.LBB3_2:
	mov	r5, r2
	mov	r0, r6
	mov	r7, #0
	str	r11, [sp, #8]
	str	r2, [sp, #16]
	str	r4, [sp, #20]
	str	r6, [sp, #12]
.LBB3_3:
	mov	r12, #0
	mov	lr, #57
	str	r11, [sp, #24]
.LBB3_4:
	sub	r6, lr, #56
	cmp	r6, r1
	cmple	r1, lr
	blt	.LBB3_6
	add	r6, r0, r12
	str	r3, [r6, #16]
	vst1.8	{d16, d17}, [r6], r8
	str	r3, [r6]
	b	.LBB3_12
.LBB3_6:
	cmp	r7, #0
	beq	.LBB3_8
	ldrb	r6, [r11]
	strb	r6, [r0, r12]
	add	r6, r4, r12
	add	r6, r6, r10
	ldrb	r2, [r11, #1]
	strb	r2, [r6, #1]
	ldrb	r2, [r11, #2]
	strb	r2, [r6, #2]
	ldrb	r2, [r11, #3]
	strb	r2, [r6, #3]
	ldrb	r2, [r11, #4]
	strb	r2, [r6, #4]
	ldrb	r2, [r11, #5]
	strb	r2, [r6, #5]
	ldrb	r2, [r11, #6]
	strb	r2, [r6, #6]
	ldrb	r2, [r11, #7]
	strb	r2, [r6, #7]
	b	.LBB3_9
.LBB3_8:
	mov	r6, r0
	str	r3, [r6, r12]!
	str	r3, [r6, #4]
.LBB3_9:
	add	r2, r4, r12
	ldrb	r6, [r11, #8]
	add	r9, r2, r10
	cmp	r7, #55
	strb	r6, [r9, #8]
	ldrb	r6, [r11, #9]
	strb	r6, [r9, #9]
	ldrb	r6, [r11, #10]
	strb	r6, [r9, #10]
	ldrb	r6, [r11, #11]
	strb	r6, [r9, #11]
	ldrb	r6, [r11, #12]
	strb	r6, [r9, #12]
	ldrb	r6, [r11, #13]
	strb	r6, [r9, #13]
	ldrb	r6, [r11, #14]
	strb	r6, [r9, #14]
	ldrb	r6, [r11, #15]
	strb	r6, [r2, #-4017]
	bhs	.LBB3_11
	ldrb	r2, [r11, #16]
	strb	r2, [r5, r12]
	ldrb	r2, [r11, #17]
	strb	r2, [r9, #17]
	ldrb	r2, [r11, #18]
	strb	r2, [r9, #18]
	ldrb	r2, [r11, #19]
	strb	r2, [r9, #19]
	ldrb	r2, [r11, #20]
	strb	r2, [r9, #20]
	ldrb	r2, [r11, #21]
	strb	r2, [r9, #21]
	ldrb	r2, [r11, #22]
	strb	r2, [r9, #22]
	ldrb	r2, [r11, #23]
	strb	r2, [r9, #23]
	b	.LBB3_12
.LBB3_11:
	mov	r2, r5
	str	r3, [r2, r12]!
	str	r3, [r2, #4]
.LBB3_12:
	add	r12, r12, #24
	sub	lr, lr, #1
	add	r11, r11, #464
	cmp	r12, #72
	bne	.LBB3_4
	ldr	r11, [sp, #24]
	add	r7, r7, #1
	add	r5, r5, #72
	add	r4, r4, #72
	add	r11, r11, #8
	add	r0, r0, #72
	cmp	r7, #56
	bne	.LBB3_3
	ldr	r11, [sp, #8]
	add	r1, r1, #1
	ldr	r2, [sp, #16]
	ldr	r4, [sp, #20]
	add	r11, r11, #464
	ldr	r6, [sp, #12]
	add	r2, r2, #4032
	ldr	r0, [sp, #4]
	add	r4, r4, #4032
	add	r6, r6, #4032
	cmp	r1, r0
	blt	.LBB3_2
.LBB3_15:
	mov	r0, #0
	add	sp, sp, #28
	pop	{r4, r5, r6, r7, r8, r9, r10, r11, lr}
	mov	pc, lr
	.p2align	2
.LCPI3_0:
	.long	4294963264
.Lfunc_end3:
	.size	.L__tvm_parallel_lambda.34, .Lfunc_end3-.L__tvm_parallel_lambda.34
	.cantunwind
	.fnend

	.p2align	2
	.type	.L__tvm_parallel_lambda.35,%function
	.code	32
.L__tvm_parallel_lambda.35:
	.fnstart
	push	{r4, r5, r6, r7, r8, r9, r10, r11, lr}
	add	r11, sp, #28
	sub	sp, sp, #92
	bic	sp, sp, #15
	ldr	r1, [r1, #4]
	mov	r5, r0
	mov	r6, sp
	mov	r4, r2
	add	r0, r1, #55
	bl	__divsi3
	add	r2, r5, #1
	mul	r1, r0, r5
	mov	r5, #56
	mul	r3, r0, r2
	mov	r0, #56
	cmp	r1, #56
	movlt	r5, r1
	cmp	r3, #56
	movlt	r0, r3
	cmp	r5, r0
	str	r0, [r6, #20]
	bge	.LBB4_9
	ldr	r0, [r4]
	mvn	r1, r1
	str	r0, [r6, #40]
	cmn	r1, #57
	ldmib	r4, {r0, r2}
	mvnle	r1, #56
	str	r2, [r6, #36]
	ldr	r2, .LCPI4_0
	add	r3, r2, #432
	mul	r7, r1, r3
	rsb	r1, r1, r1, lsl #6
	sub	r1, r2, r1, lsl #6
	mov	r2, #0
	add	r10, r0, r1
	sub	r0, r7, #3584
	str	r0, [r6, #16]
	mov	r0, #184
	mov	r7, #24
.LBB4_2:
	str	r2, [r6, #24]
	rsb	r1, r2, r2, lsl #3
	ldr	r2, [r6, #16]
	str	r5, [r6, #32]
	add	r1, r2, r1, lsl #9
	str	r1, [r6, #72]
	sub	r1, sp, #128
	bic	r4, r1, #15
	add	r1, r4, #112
	str	r1, [r6, #68]
	add	r1, r4, #96
	str	r1, [r6, #64]
	add	r1, r4, #80
	str	r1, [r6, #60]
	add	r1, r4, #64
	str	r1, [r6, #56]
	add	r1, r4, #48
	str	r1, [r6, #52]
	add	r1, r4, #32
	str	r1, [r6, #48]
	add	r1, r4, #16
	str	r1, [r6, #44]
	mov	sp, r4
	mov	r2, #0
	str	r10, [r6, #28]
.LBB4_3:
	ldr	r1, [r6, #72]
	mov	lr, #0
	ldr	r8, [r6, #40]
	add	r12, r1, r2, lsl #6
	str	r2, [r6, #76]
.LBB4_4:
	vmov.i32	q8, #0x0
	add	r9, r4, lr, lsl #4
	mov	r1, r10
	mov	r2, #0
	vst1.64	{d16, d17}, [r9:128]
.LBB4_5:
	add	r3, r8, r2
	vldr	d18, [r1, #-16]
	vldr	d19, [r1, #-8]
	add	r2, r2, #192
	mov	r5, r3
	vldr	d30, [r3, #32]
	vldr	d31, [r3, #40]
	vand	d30, d30, d18
	vldr	d0, [r3, #48]
	vand	d31, d31, d18
	vldr	d1, [r3, #56]
	vand	d0, d0, d18
	vld1.8	{d20}, [r5:64], r0
	vand	d1, d1, d18
	vand	d20, d20, d18
	cmp	r2, #576
	vldr	d27, [r3, #8]
	vcnt.8	d0, d0
	vldr	d29, [r3, #24]
	vand	d27, d27, d18
	vldr	d28, [r3, #16]
	vand	d29, d29, d18
	vand	d18, d28, d18
	vldr	d23, [r3, #96]
	vcnt.8	d28, d1
	vldr	d24, [r3, #104]
	vcnt.8	d31, d31
	vldr	d25, [r3, #112]
	vcnt.8	d30, d30
	vldr	d26, [r3, #120]
	vcnt.8	d27, d27
	vldr	d21, [r3, #80]
	vcnt.8	d20, d20
	vldr	d22, [r3, #88]
	vcnt.8	d29, d29
	vldr	d3, [r3, #72]
	vcnt.8	d18, d18
	vldr	d2, [r3, #64]
	vand	d23, d23, d19
	vldr	d4, [r3, #144]
	vand	d24, d24, d19
	vand	d25, d25, d19
	vand	d26, d26, d19
	vpadd.i8	d28, d0, d28
	vldr	d0, [r3, #168]
	vpadd.i8	d30, d30, d31
	vldr	d31, [r3, #160]
	vand	d21, d21, d19
	vand	d22, d22, d19
	vand	d3, d3, d19
	vand	d19, d2, d19
	vld1.8	{d2}, [r1:64], r7
	vpadd.i8	d18, d18, d29
	vpadd.i8	d20, d20, d27
	vldr	d29, [r3, #136]
	vcnt.8	d26, d26
	vldr	d27, [r3, #128]
	vcnt.8	d25, d25
	vcnt.8	d24, d24
	vcnt.8	d23, d23
	vpadd.i8	d7, d30, d28
	vldr	d28, [r3, #176]
	vcnt.8	d1, d3
	vldr	d3, [r3, #152]
	vcnt.8	d19, d19
	vcnt.8	d22, d22
	vcnt.8	d21, d21
	vpadd.i8	d6, d20, d18
	vldr	d18, [r5]
	vand	d27, d27, d2
	vand	d3, d3, d2
	vand	d4, d4, d2
	vpadd.i8	d23, d23, d24
	vand	d29, d29, d2
	vpadd.i8	d25, d25, d26
	vand	d20, d28, d2
	vand	d18, d18, d2
	vand	d0, d0, d2
	vpadd.i8	d21, d21, d22
	vand	d31, d31, d2
	vpadd.i8	d19, d19, d1
	vcnt.8	d26, d29
	vcnt.8	d22, d3
	vcnt.8	d24, d4
	vcnt.8	d27, d27
	vpadal.u8	q8, q3
	vcnt.8	d29, d31
	vpadd.i8	d31, d23, d25
	vcnt.8	d28, d0
	vcnt.8	d20, d20
	vcnt.8	d18, d18
	vpadd.i8	d30, d19, d21
	vpadd.i8	d19, d24, d22
	vpadd.i8	d21, d27, d26
	vpadd.i8	d18, d20, d18
	vpadd.i8	d22, d29, d28
	vpadal.u8	q8, q15
	vpadd.i8	d20, d21, d19
	vpadd.i8	d21, d22, d18
	vpadal.u8	q8, q10
	bne	.LBB4_5
	add	lr, lr, #1
	add	r8, r8, #576
	cmp	lr, #8
	vst1.64	{d16, d17}, [r9:128]
	bne	.LBB4_4
	ldr	r2, [r6, #36]
	add	r10, r10, #72
	vld1.16	{d16, d17}, [r4]
	add	r1, r2, r12, lsl #1
	ldr	r3, [r6, #44]
	vst1.16	{d16, d17}, [r1]
	orr	r1, r12, #8
	vld1.16	{d16, d17}, [r3]
	add	r1, r2, r1, lsl #1
	vst1.16	{d16, d17}, [r1]
	orr	r1, r12, #16
	ldr	r3, [r6, #48]
	add	r1, r2, r1, lsl #1
	vld1.16	{d16, d17}, [r3]
	vst1.16	{d16, d17}, [r1]
	orr	r1, r12, #24
	ldr	r3, [r6, #52]
	add	r1, r2, r1, lsl #1
	vld1.16	{d16, d17}, [r3]
	vst1.16	{d16, d17}, [r1]
	orr	r1, r12, #32
	ldr	r3, [r6, #56]
	add	r1, r2, r1, lsl #1
	vld1.16	{d16, d17}, [r3]
	vst1.16	{d16, d17}, [r1]
	orr	r1, r12, #40
	ldr	r3, [r6, #60]
	add	r1, r2, r1, lsl #1
	vld1.16	{d16, d17}, [r3]
	vst1.16	{d16, d17}, [r1]
	orr	r1, r12, #48
	ldr	r3, [r6, #64]
	add	r1, r2, r1, lsl #1
	vld1.16	{d16, d17}, [r3]
	vst1.16	{d16, d17}, [r1]
	orr	r1, r12, #56
	ldr	r3, [r6, #68]
	add	r1, r2, r1, lsl #1
	ldr	r2, [r6, #76]
	vld1.16	{d16, d17}, [r3]
	add	r2, r2, #1
	cmp	r2, #56
	vst1.16	{d16, d17}, [r1]
	bne	.LBB4_3
	ldr	r10, [r6, #28]
	ldr	r2, [r6, #24]
	ldr	r5, [r6, #32]
	add	r10, r10, #4032
	ldr	r1, [r6, #20]
	add	r2, r2, #1
	add	r5, r5, #1
	cmp	r5, r1
	blt	.LBB4_2
.LBB4_9:
	mov	r0, #0
	sub	sp, r11, #28
	pop	{r4, r5, r6, r7, r8, r9, r10, r11, lr}
	mov	pc, lr
	.p2align	2
.LCPI4_0:
	.long	4294963280
.Lfunc_end4:
	.size	.L__tvm_parallel_lambda.35, .Lfunc_end4-.L__tvm_parallel_lambda.35
	.cantunwind
	.fnend

	.type	__TVMAPISetLastError,%object
	.bss
	.weak	__TVMAPISetLastError
	.p2align	2
__TVMAPISetLastError:
	.long	0
	.size	__TVMAPISetLastError, 4

	.type	__TVMBackendParallelLaunch,%object
	.weak	__TVMBackendParallelLaunch
	.p2align	2
__TVMBackendParallelLaunch:
	.long	0
	.size	__TVMBackendParallelLaunch, 4

	.type	.L.str,%object
	.section	.rodata,"a",%progbits
.L.str:
	.asciz	"Assert fail: (num_args == 3), default_function: num_args should be 3"
	.size	.L.str, 69

	.type	.L.str.1,%object
.L.str.1:
	.asciz	"Assert fail: ((((1 == int32(arg0.strides[3])) && (64 == int32(arg0.strides[2]))) && (3584 == int32(arg0.strides[1]))) && (200704 == int32(arg0.strides[0]))), arg0.strides: expected to be compact array"
	.size	.L.str.1, 201

	.type	.L.str.2,%object
.L.str.2:
	.asciz	"Assert fail: ((((1 == int32(arg1.strides[3])) && (64 == int32(arg1.strides[2]))) && (4096 == int32(arg1.strides[1]))) && (12288 == int32(arg1.strides[0]))), arg1.strides: expected to be compact array"
	.size	.L.str.2, 200

	.type	.L.str.3,%object
.L.str.3:
	.asciz	"Assert fail: ((((1 == int32(arg2.strides[3])) && (64 == int32(arg2.strides[2]))) && (3584 == int32(arg2.strides[1]))) && (200704 == int32(arg2.strides[0]))), arg2.strides: expected to be compact array"
	.size	.L.str.3, 201

	.type	.L.str.4,%object
.L.str.4:
	.asciz	"Assert fail: ((((arg0.code == 3) || (arg0.code == 13)) || (arg0.code == 7)) || (arg0.code == 4)), default_function: Expect arg[0] to be pointer"
	.size	.L.str.4, 144

	.type	.L.str.5,%object
.L.str.5:
	.asciz	"Assert fail: ((((arg1.code == 3) || (arg1.code == 13)) || (arg1.code == 7)) || (arg1.code == 4)), default_function: Expect arg[1] to be pointer"
	.size	.L.str.5, 144

	.type	.L.str.6,%object
.L.str.6:
	.asciz	"Assert fail: ((((arg2.code == 3) || (arg2.code == 13)) || (arg2.code == 7)) || (arg2.code == 4)), default_function: Expect arg[2] to be pointer"
	.size	.L.str.6, 144

	.type	.L.str.7,%object
.L.str.7:
	.asciz	"Assert fail: (dev_type == 1), device_type need to be 1"
	.size	.L.str.7, 55

	.type	.L.str.8,%object
.L.str.8:
	.asciz	"Assert fail: (4 == tvm_struct_get(arg0, 0, 4)), arg0.ndim is expected to equal 4"
	.size	.L.str.8, 81

	.type	.L.str.9,%object
.L.str.9:
	.asciz	"Assert fail: (((tvm_struct_get(arg0, 0, 5) == (uint8)1) && (tvm_struct_get(arg0, 0, 6) == (uint8)8)) && (tvm_struct_get(arg0, 0, 7) == (uint16)1)), arg0.dtype is expected to be uint8"
	.size	.L.str.9, 183

	.type	.L.str.10,%object
.L.str.10:
	.asciz	"Assert fail: (int32(arg0.shape[0]) == 1), Argument arg0.shape[0] has an unsatisfied constraint"
	.size	.L.str.10, 95

	.type	.L.str.11,%object
.L.str.11:
	.asciz	"Assert fail: (int32(arg0.shape[1]) == 56), Argument arg0.shape[1] has an unsatisfied constraint"
	.size	.L.str.11, 96

	.type	.L.str.12,%object
.L.str.12:
	.asciz	"Assert fail: (int32(arg0.shape[2]) == 56), Argument arg0.shape[2] has an unsatisfied constraint"
	.size	.L.str.12, 96

	.type	.L.str.13,%object
.L.str.13:
	.asciz	"Assert fail: (int32(arg0.shape[3]) == 64), Argument arg0.shape[3] has an unsatisfied constraint"
	.size	.L.str.13, 96

	.type	.L.str.14,%object
.L.str.14:
	.asciz	"Assert fail: (tvm_struct_get(arg0, 0, 8) == (uint64)0), Argument arg0.byte_offset has an unsatisfied constraint"
	.size	.L.str.14, 112

	.type	.L.str.15,%object
.L.str.15:
	.asciz	"Assert fail: (4 == tvm_struct_get(arg1, 0, 4)), arg1.ndim is expected to equal 4"
	.size	.L.str.15, 81

	.type	.L.str.16,%object
.L.str.16:
	.asciz	"Assert fail: (((tvm_struct_get(arg1, 0, 5) == (uint8)1) && (tvm_struct_get(arg1, 0, 6) == (uint8)8)) && (tvm_struct_get(arg1, 0, 7) == (uint16)1)), arg1.dtype is expected to be uint8"
	.size	.L.str.16, 183

	.type	.L.str.17,%object
.L.str.17:
	.asciz	"Assert fail: (int32(arg1.shape[0]) == 3), Argument arg1.shape[0] has an unsatisfied constraint"
	.size	.L.str.17, 95

	.type	.L.str.18,%object
.L.str.18:
	.asciz	"Assert fail: (int32(arg1.shape[1]) == 3), Argument arg1.shape[1] has an unsatisfied constraint"
	.size	.L.str.18, 95

	.type	.L.str.19,%object
.L.str.19:
	.asciz	"Assert fail: (int32(arg1.shape[2]) == 64), Argument arg1.shape[2] has an unsatisfied constraint"
	.size	.L.str.19, 96

	.type	.L.str.20,%object
.L.str.20:
	.asciz	"Assert fail: (int32(arg1.shape[3]) == 64), Argument arg1.shape[3] has an unsatisfied constraint"
	.size	.L.str.20, 96

	.type	.L.str.21,%object
.L.str.21:
	.asciz	"Assert fail: (tvm_struct_get(arg1, 0, 8) == (uint64)0), Argument arg1.byte_offset has an unsatisfied constraint"
	.size	.L.str.21, 112

	.type	.L.str.22,%object
.L.str.22:
	.asciz	"Assert fail: (1 == tvm_struct_get(arg1, 0, 10)), Argument arg1.device_type has an unsatisfied constraint"
	.size	.L.str.22, 105

	.type	.L.str.23,%object
.L.str.23:
	.asciz	"Assert fail: (dev_id == tvm_struct_get(arg1, 0, 9)), Argument arg1.device_id has an unsatisfied constraint"
	.size	.L.str.23, 107

	.type	.L.str.24,%object
.L.str.24:
	.asciz	"Assert fail: (4 == tvm_struct_get(arg2, 0, 4)), arg2.ndim is expected to equal 4"
	.size	.L.str.24, 81

	.type	.L.str.25,%object
.L.str.25:
	.asciz	"Assert fail: (((tvm_struct_get(arg2, 0, 5) == (uint8)1) && (tvm_struct_get(arg2, 0, 6) == (uint8)16)) && (tvm_struct_get(arg2, 0, 7) == (uint16)1)), arg2.dtype is expected to be uint16"
	.size	.L.str.25, 185

	.type	.L.str.26,%object
.L.str.26:
	.asciz	"Assert fail: (int32(arg2.shape[0]) == 1), Argument arg2.shape[0] has an unsatisfied constraint"
	.size	.L.str.26, 95

	.type	.L.str.27,%object
.L.str.27:
	.asciz	"Assert fail: (int32(arg2.shape[1]) == 56), Argument arg2.shape[1] has an unsatisfied constraint"
	.size	.L.str.27, 96

	.type	.L.str.28,%object
.L.str.28:
	.asciz	"Assert fail: (int32(arg2.shape[2]) == 56), Argument arg2.shape[2] has an unsatisfied constraint"
	.size	.L.str.28, 96

	.type	.L.str.29,%object
.L.str.29:
	.asciz	"Assert fail: (int32(arg2.shape[3]) == 64), Argument arg2.shape[3] has an unsatisfied constraint"
	.size	.L.str.29, 96

	.type	.L.str.30,%object
.L.str.30:
	.asciz	"Assert fail: (tvm_struct_get(arg2, 0, 8) == (uint64)0), Argument arg2.byte_offset has an unsatisfied constraint"
	.size	.L.str.30, 112

	.type	.L.str.31,%object
.L.str.31:
	.asciz	"Assert fail: (1 == tvm_struct_get(arg2, 0, 10)), Argument arg2.device_type has an unsatisfied constraint"
	.size	.L.str.31, 105

	.type	.L.str.32,%object
.L.str.32:
	.asciz	"Assert fail: (dev_id == tvm_struct_get(arg2, 0, 9)), Argument arg2.device_id has an unsatisfied constraint"
	.size	.L.str.32, 107

	.type	__TVMBackendAllocWorkspace,%object
	.bss
	.weak	__TVMBackendAllocWorkspace
	.p2align	2
__TVMBackendAllocWorkspace:
	.long	0
	.size	__TVMBackendAllocWorkspace, 4

	.type	__TVMBackendFreeWorkspace,%object
	.weak	__TVMBackendFreeWorkspace
	.p2align	2
__TVMBackendFreeWorkspace:
	.long	0
	.size	__TVMBackendFreeWorkspace, 4

	.type	__tvm_main__,%object
	.section	.rodata,"a",%progbits
	.weak	__tvm_main__
__tvm_main__:
	.asciz	"default_function"
	.size	__tvm_main__, 17


	.section	".note.GNU-stack","",%progbits
