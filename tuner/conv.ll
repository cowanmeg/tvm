; ModuleID = 'default_function'
source_filename = "default_function"
target datalayout = "e-m:e-p:32:32-i64:64-v128:64:128-a:0:32-n32-S64"
target triple = "arm-linux-gnueabihf"

%0 = type { i32*, i32 }
%1 = type { i8*, %2, i32, %3, i64*, i64*, i64 }
%2 = type { i32, i32 }
%3 = type { i8, i8, i16 }
%4 = type { i8*, i8* }
%5 = type { i8*, i8* }
%6 = type { i8*, i8*, i8* }

@__TVMAPISetLastError = linkonce dllexport local_unnamed_addr global void (i8*)* null, align 4
@__TVMBackendParallelLaunch = linkonce dllexport local_unnamed_addr global i32 (i32 (i32, %0*, i8*)*, i8*, i32)* null, align 4
@.str = private constant [69 x i8] c"Assert fail: (num_args == 3), default_function: num_args should be 3\00", align 1
@.str.1 = private constant [201 x i8] c"Assert fail: ((((1 == int32(arg0.strides[3])) && (64 == int32(arg0.strides[2]))) && (3584 == int32(arg0.strides[1]))) && (200704 == int32(arg0.strides[0]))), arg0.strides: expected to be compact array\00", align 1
@.str.2 = private constant [200 x i8] c"Assert fail: ((((1 == int32(arg1.strides[3])) && (64 == int32(arg1.strides[2]))) && (4096 == int32(arg1.strides[1]))) && (12288 == int32(arg1.strides[0]))), arg1.strides: expected to be compact array\00", align 1
@.str.3 = private constant [201 x i8] c"Assert fail: ((((1 == int32(arg2.strides[3])) && (64 == int32(arg2.strides[2]))) && (3584 == int32(arg2.strides[1]))) && (200704 == int32(arg2.strides[0]))), arg2.strides: expected to be compact array\00", align 1
@.str.4 = private constant [144 x i8] c"Assert fail: ((((arg0.code == 3) || (arg0.code == 13)) || (arg0.code == 7)) || (arg0.code == 4)), default_function: Expect arg[0] to be pointer\00", align 1
@.str.5 = private constant [144 x i8] c"Assert fail: ((((arg1.code == 3) || (arg1.code == 13)) || (arg1.code == 7)) || (arg1.code == 4)), default_function: Expect arg[1] to be pointer\00", align 1
@.str.6 = private constant [144 x i8] c"Assert fail: ((((arg2.code == 3) || (arg2.code == 13)) || (arg2.code == 7)) || (arg2.code == 4)), default_function: Expect arg[2] to be pointer\00", align 1
@.str.7 = private constant [55 x i8] c"Assert fail: (dev_type == 1), device_type need to be 1\00", align 1
@.str.8 = private constant [81 x i8] c"Assert fail: (4 == tvm_struct_get(arg0, 0, 4)), arg0.ndim is expected to equal 4\00", align 1
@.str.9 = private constant [183 x i8] c"Assert fail: (((tvm_struct_get(arg0, 0, 5) == (uint8)1) && (tvm_struct_get(arg0, 0, 6) == (uint8)8)) && (tvm_struct_get(arg0, 0, 7) == (uint16)1)), arg0.dtype is expected to be uint8\00", align 1
@.str.10 = private constant [95 x i8] c"Assert fail: (int32(arg0.shape[0]) == 1), Argument arg0.shape[0] has an unsatisfied constraint\00", align 1
@.str.11 = private constant [96 x i8] c"Assert fail: (int32(arg0.shape[1]) == 56), Argument arg0.shape[1] has an unsatisfied constraint\00", align 1
@.str.12 = private constant [96 x i8] c"Assert fail: (int32(arg0.shape[2]) == 56), Argument arg0.shape[2] has an unsatisfied constraint\00", align 1
@.str.13 = private constant [96 x i8] c"Assert fail: (int32(arg0.shape[3]) == 64), Argument arg0.shape[3] has an unsatisfied constraint\00", align 1
@.str.14 = private constant [112 x i8] c"Assert fail: (tvm_struct_get(arg0, 0, 8) == (uint64)0), Argument arg0.byte_offset has an unsatisfied constraint\00", align 1
@.str.15 = private constant [81 x i8] c"Assert fail: (4 == tvm_struct_get(arg1, 0, 4)), arg1.ndim is expected to equal 4\00", align 1
@.str.16 = private constant [183 x i8] c"Assert fail: (((tvm_struct_get(arg1, 0, 5) == (uint8)1) && (tvm_struct_get(arg1, 0, 6) == (uint8)8)) && (tvm_struct_get(arg1, 0, 7) == (uint16)1)), arg1.dtype is expected to be uint8\00", align 1
@.str.17 = private constant [95 x i8] c"Assert fail: (int32(arg1.shape[0]) == 3), Argument arg1.shape[0] has an unsatisfied constraint\00", align 1
@.str.18 = private constant [95 x i8] c"Assert fail: (int32(arg1.shape[1]) == 3), Argument arg1.shape[1] has an unsatisfied constraint\00", align 1
@.str.19 = private constant [96 x i8] c"Assert fail: (int32(arg1.shape[2]) == 64), Argument arg1.shape[2] has an unsatisfied constraint\00", align 1
@.str.20 = private constant [96 x i8] c"Assert fail: (int32(arg1.shape[3]) == 64), Argument arg1.shape[3] has an unsatisfied constraint\00", align 1
@.str.21 = private constant [112 x i8] c"Assert fail: (tvm_struct_get(arg1, 0, 8) == (uint64)0), Argument arg1.byte_offset has an unsatisfied constraint\00", align 1
@.str.22 = private constant [105 x i8] c"Assert fail: (1 == tvm_struct_get(arg1, 0, 10)), Argument arg1.device_type has an unsatisfied constraint\00", align 1
@.str.23 = private constant [107 x i8] c"Assert fail: (dev_id == tvm_struct_get(arg1, 0, 9)), Argument arg1.device_id has an unsatisfied constraint\00", align 1
@.str.24 = private constant [81 x i8] c"Assert fail: (4 == tvm_struct_get(arg2, 0, 4)), arg2.ndim is expected to equal 4\00", align 1
@.str.25 = private constant [185 x i8] c"Assert fail: (((tvm_struct_get(arg2, 0, 5) == (uint8)1) && (tvm_struct_get(arg2, 0, 6) == (uint8)16)) && (tvm_struct_get(arg2, 0, 7) == (uint16)1)), arg2.dtype is expected to be uint16\00", align 1
@.str.26 = private constant [95 x i8] c"Assert fail: (int32(arg2.shape[0]) == 1), Argument arg2.shape[0] has an unsatisfied constraint\00", align 1
@.str.27 = private constant [96 x i8] c"Assert fail: (int32(arg2.shape[1]) == 56), Argument arg2.shape[1] has an unsatisfied constraint\00", align 1
@.str.28 = private constant [96 x i8] c"Assert fail: (int32(arg2.shape[2]) == 56), Argument arg2.shape[2] has an unsatisfied constraint\00", align 1
@.str.29 = private constant [96 x i8] c"Assert fail: (int32(arg2.shape[3]) == 64), Argument arg2.shape[3] has an unsatisfied constraint\00", align 1
@.str.30 = private constant [112 x i8] c"Assert fail: (tvm_struct_get(arg2, 0, 8) == (uint64)0), Argument arg2.byte_offset has an unsatisfied constraint\00", align 1
@.str.31 = private constant [105 x i8] c"Assert fail: (1 == tvm_struct_get(arg2, 0, 10)), Argument arg2.device_type has an unsatisfied constraint\00", align 1
@.str.32 = private constant [107 x i8] c"Assert fail: (dev_id == tvm_struct_get(arg2, 0, 9)), Argument arg2.device_id has an unsatisfied constraint\00", align 1
@__TVMBackendAllocWorkspace = linkonce dllexport local_unnamed_addr global i8* (i32, i32, i64, i32, i32)* null, align 4
@__TVMBackendFreeWorkspace = linkonce dllexport local_unnamed_addr global i32 (i32, i32, i8*)* null, align 4
@__tvm_main__ = weak local_unnamed_addr constant [17 x i8] c"default_function\00", align 1

define dllexport i32 @default_function(i8* noalias nocapture readonly, i8* noalias nocapture readonly, i32) local_unnamed_addr {
entry:
  %3 = icmp eq i32 %2, 3
  br i1 %3, label %assert_end, label %assert_fail, !prof !1

assert_fail:                                      ; preds = %entry
  %4 = load void (i8*)*, void (i8*)** @__TVMAPISetLastError, align 4, !tbaa !2
  tail call void %4(i8* getelementptr inbounds ([69 x i8], [69 x i8]* @.str, i32 0, i32 0))
  ret i32 -1

assert_end:                                       ; preds = %entry
  %5 = bitcast i8* %0 to %1**
  %6 = load %1*, %1** %5, align 4
  %7 = bitcast i8* %1 to i32*
  %8 = load i32, i32* %7, align 4, !tbaa !5
  %9 = getelementptr inbounds i8, i8* %0, i32 8
  %10 = bitcast i8* %9 to %1**
  %11 = load %1*, %1** %10, align 4
  %12 = getelementptr inbounds i8, i8* %1, i32 4
  %13 = bitcast i8* %12 to i32*
  %14 = load i32, i32* %13, align 4, !tbaa !19
  %15 = getelementptr inbounds i8, i8* %0, i32 16
  %16 = bitcast i8* %15 to %1**
  %17 = load %1*, %1** %16, align 4
  %18 = getelementptr inbounds i8, i8* %1, i32 8
  %19 = bitcast i8* %18 to i32*
  %20 = load i32, i32* %19, align 4, !tbaa !21
  %21 = getelementptr inbounds %1, %1* %6, i32 0, i32 0
  %22 = load i8*, i8** %21, align 4
  %23 = getelementptr inbounds %1, %1* %6, i32 0, i32 4
  %24 = load i64*, i64** %23, align 4
  %25 = getelementptr inbounds %1, %1* %6, i32 0, i32 5
  %26 = load i64*, i64** %25, align 4
  %27 = icmp eq i64* %26, null
  br i1 %27, label %if_end, label %if_then, !prof !24

if_then:                                          ; preds = %assert_end
  %28 = load i64, i64* %26, align 8, !tbaa !25
  %29 = trunc i64 %28 to i32
  %30 = icmp eq i32 %29, 200704
  %31 = getelementptr inbounds i64, i64* %26, i32 1
  %32 = load i64, i64* %31, align 8, !tbaa !39
  %33 = trunc i64 %32 to i32
  %34 = icmp eq i32 %33, 3584
  %35 = getelementptr inbounds i64, i64* %26, i32 2
  %36 = bitcast i64* %35 to <2 x i64>*
  %37 = load <2 x i64>, <2 x i64>* %36, align 8, !tbaa !41
  %38 = trunc <2 x i64> %37 to <2 x i32>
  %39 = icmp eq <2 x i32> %38, <i32 64, i32 1>
  %40 = extractelement <2 x i1> %39, i32 0
  %41 = extractelement <2 x i1> %39, i32 1
  %42 = and i1 %40, %41
  %43 = and i1 %34, %42
  %44 = and i1 %30, %43
  br i1 %44, label %if_end, label %assert_fail1, !prof !1

if_end:                                           ; preds = %assert_end, %if_then
  %45 = getelementptr inbounds %1, %1* %6, i32 0, i32 1, i32 0
  %46 = load i32, i32* %45, align 4
  %47 = getelementptr inbounds %1, %1* %6, i32 0, i32 1, i32 1
  %48 = load i32, i32* %47, align 4
  %49 = getelementptr inbounds %1, %1* %11, i32 0, i32 0
  %50 = load i8*, i8** %49, align 4
  %51 = getelementptr inbounds %1, %1* %11, i32 0, i32 4
  %52 = load i64*, i64** %51, align 4
  %53 = getelementptr inbounds %1, %1* %11, i32 0, i32 5
  %54 = load i64*, i64** %53, align 4
  %55 = icmp eq i64* %54, null
  br i1 %55, label %if_end4, label %if_then3, !prof !24

assert_fail1:                                     ; preds = %if_then
  %56 = load void (i8*)*, void (i8*)** @__TVMAPISetLastError, align 4, !tbaa !2
  tail call void %56(i8* getelementptr inbounds ([201 x i8], [201 x i8]* @.str.1, i32 0, i32 0))
  ret i32 -1

if_then3:                                         ; preds = %if_end
  %57 = load i64, i64* %54, align 8, !tbaa !43
  %58 = trunc i64 %57 to i32
  %59 = icmp eq i32 %58, 12288
  %60 = getelementptr inbounds i64, i64* %54, i32 1
  %61 = load i64, i64* %60, align 8, !tbaa !57
  %62 = trunc i64 %61 to i32
  %63 = icmp eq i32 %62, 4096
  %64 = getelementptr inbounds i64, i64* %54, i32 2
  %65 = bitcast i64* %64 to <2 x i64>*
  %66 = load <2 x i64>, <2 x i64>* %65, align 8, !tbaa !59
  %67 = trunc <2 x i64> %66 to <2 x i32>
  %68 = icmp eq <2 x i32> %67, <i32 64, i32 1>
  %69 = extractelement <2 x i1> %68, i32 0
  %70 = extractelement <2 x i1> %68, i32 1
  %71 = and i1 %69, %70
  %72 = and i1 %63, %71
  %73 = and i1 %59, %72
  br i1 %73, label %if_end4, label %assert_fail5, !prof !1

if_end4:                                          ; preds = %if_end, %if_then3
  %74 = getelementptr inbounds %1, %1* %17, i32 0, i32 0
  %75 = load i8*, i8** %74, align 4
  %76 = getelementptr inbounds %1, %1* %17, i32 0, i32 4
  %77 = load i64*, i64** %76, align 4
  %78 = getelementptr inbounds %1, %1* %17, i32 0, i32 5
  %79 = load i64*, i64** %78, align 4
  %80 = icmp eq i64* %79, null
  br i1 %80, label %if_end8, label %if_then7, !prof !24

assert_fail5:                                     ; preds = %if_then3
  %81 = load void (i8*)*, void (i8*)** @__TVMAPISetLastError, align 4, !tbaa !2
  tail call void %81(i8* getelementptr inbounds ([200 x i8], [200 x i8]* @.str.2, i32 0, i32 0))
  ret i32 -1

if_then7:                                         ; preds = %if_end4
  %82 = load i64, i64* %79, align 8, !tbaa !61
  %83 = trunc i64 %82 to i32
  %84 = icmp eq i32 %83, 200704
  %85 = getelementptr inbounds i64, i64* %79, i32 1
  %86 = load i64, i64* %85, align 8, !tbaa !75
  %87 = trunc i64 %86 to i32
  %88 = icmp eq i32 %87, 3584
  %89 = getelementptr inbounds i64, i64* %79, i32 2
  %90 = bitcast i64* %89 to <2 x i64>*
  %91 = load <2 x i64>, <2 x i64>* %90, align 8, !tbaa !77
  %92 = trunc <2 x i64> %91 to <2 x i32>
  %93 = icmp eq <2 x i32> %92, <i32 64, i32 1>
  %94 = extractelement <2 x i1> %93, i32 0
  %95 = extractelement <2 x i1> %93, i32 1
  %96 = and i1 %94, %95
  %97 = and i1 %88, %96
  %98 = and i1 %84, %97
  br i1 %98, label %if_end8, label %assert_fail9, !prof !1

if_end8:                                          ; preds = %if_end4, %if_then7
  switch i32 %8, label %assert_fail11 [
    i32 13, label %assert_end12
    i32 7, label %assert_end12
    i32 4, label %assert_end12
    i32 3, label %assert_end12
  ]

assert_fail9:                                     ; preds = %if_then7
  %99 = load void (i8*)*, void (i8*)** @__TVMAPISetLastError, align 4, !tbaa !2
  tail call void %99(i8* getelementptr inbounds ([201 x i8], [201 x i8]* @.str.3, i32 0, i32 0))
  ret i32 -1

assert_fail11:                                    ; preds = %if_end8
  %100 = load void (i8*)*, void (i8*)** @__TVMAPISetLastError, align 4, !tbaa !2
  tail call void %100(i8* getelementptr inbounds ([144 x i8], [144 x i8]* @.str.4, i32 0, i32 0))
  ret i32 -1

assert_end12:                                     ; preds = %if_end8, %if_end8, %if_end8, %if_end8
  switch i32 %14, label %assert_fail13 [
    i32 13, label %assert_end14
    i32 7, label %assert_end14
    i32 4, label %assert_end14
    i32 3, label %assert_end14
  ]

assert_fail13:                                    ; preds = %assert_end12
  %101 = load void (i8*)*, void (i8*)** @__TVMAPISetLastError, align 4, !tbaa !2
  tail call void %101(i8* getelementptr inbounds ([144 x i8], [144 x i8]* @.str.5, i32 0, i32 0))
  ret i32 -1

assert_end14:                                     ; preds = %assert_end12, %assert_end12, %assert_end12, %assert_end12
  switch i32 %20, label %assert_fail15 [
    i32 13, label %assert_end16
    i32 7, label %assert_end16
    i32 4, label %assert_end16
    i32 3, label %assert_end16
  ]

assert_fail15:                                    ; preds = %assert_end14
  %102 = load void (i8*)*, void (i8*)** @__TVMAPISetLastError, align 4, !tbaa !2
  tail call void %102(i8* getelementptr inbounds ([144 x i8], [144 x i8]* @.str.6, i32 0, i32 0))
  ret i32 -1

assert_end16:                                     ; preds = %assert_end14, %assert_end14, %assert_end14, %assert_end14
  %103 = icmp eq i32 %46, 1
  br i1 %103, label %assert_end18, label %assert_fail17, !prof !1

assert_fail17:                                    ; preds = %assert_end16
  %104 = load void (i8*)*, void (i8*)** @__TVMAPISetLastError, align 4, !tbaa !2
  tail call void %104(i8* getelementptr inbounds ([55 x i8], [55 x i8]* @.str.7, i32 0, i32 0))
  ret i32 -1

assert_end18:                                     ; preds = %assert_end16
  %105 = getelementptr inbounds %1, %1* %6, i32 0, i32 2
  %106 = load i32, i32* %105, align 4
  %107 = icmp eq i32 %106, 4
  br i1 %107, label %assert_end20, label %assert_fail19, !prof !1

assert_fail19:                                    ; preds = %assert_end18
  %108 = load void (i8*)*, void (i8*)** @__TVMAPISetLastError, align 4, !tbaa !2
  tail call void %108(i8* getelementptr inbounds ([81 x i8], [81 x i8]* @.str.8, i32 0, i32 0))
  ret i32 -1

assert_end20:                                     ; preds = %assert_end18
  %109 = getelementptr inbounds %1, %1* %6, i32 0, i32 3, i32 2
  %110 = load i16, i16* %109, align 2
  %111 = icmp eq i16 %110, 1
  %112 = getelementptr inbounds %1, %1* %6, i32 0, i32 3, i32 1
  %113 = load i8, i8* %112, align 1
  %114 = icmp eq i8 %113, 8
  %115 = getelementptr inbounds %1, %1* %6, i32 0, i32 3, i32 0
  %116 = load i8, i8* %115, align 1
  %117 = icmp eq i8 %116, 1
  %118 = and i1 %114, %117
  %119 = and i1 %111, %118
  br i1 %119, label %assert_end22, label %assert_fail21, !prof !1

assert_fail21:                                    ; preds = %assert_end20
  %120 = load void (i8*)*, void (i8*)** @__TVMAPISetLastError, align 4, !tbaa !2
  tail call void %120(i8* getelementptr inbounds ([183 x i8], [183 x i8]* @.str.9, i32 0, i32 0))
  ret i32 -1

assert_end22:                                     ; preds = %assert_end20
  %121 = load i64, i64* %24, align 8, !tbaa !79
  %122 = trunc i64 %121 to i32
  %123 = icmp eq i32 %122, 1
  br i1 %123, label %assert_end24, label %assert_fail23, !prof !1

assert_fail23:                                    ; preds = %assert_end22
  %124 = load void (i8*)*, void (i8*)** @__TVMAPISetLastError, align 4, !tbaa !2
  tail call void %124(i8* getelementptr inbounds ([95 x i8], [95 x i8]* @.str.10, i32 0, i32 0))
  ret i32 -1

assert_end24:                                     ; preds = %assert_end22
  %125 = getelementptr inbounds i64, i64* %24, i32 1
  %126 = load i64, i64* %125, align 8, !tbaa !93
  %127 = trunc i64 %126 to i32
  %128 = icmp eq i32 %127, 56
  br i1 %128, label %assert_end26, label %assert_fail25, !prof !1

assert_fail25:                                    ; preds = %assert_end24
  %129 = load void (i8*)*, void (i8*)** @__TVMAPISetLastError, align 4, !tbaa !2
  tail call void %129(i8* getelementptr inbounds ([96 x i8], [96 x i8]* @.str.11, i32 0, i32 0))
  ret i32 -1

assert_end26:                                     ; preds = %assert_end24
  %130 = getelementptr inbounds i64, i64* %24, i32 2
  %131 = load i64, i64* %130, align 8, !tbaa !95
  %132 = trunc i64 %131 to i32
  %133 = icmp eq i32 %132, 56
  br i1 %133, label %assert_end28, label %assert_fail27, !prof !1

assert_fail27:                                    ; preds = %assert_end26
  %134 = load void (i8*)*, void (i8*)** @__TVMAPISetLastError, align 4, !tbaa !2
  tail call void %134(i8* getelementptr inbounds ([96 x i8], [96 x i8]* @.str.12, i32 0, i32 0))
  ret i32 -1

assert_end28:                                     ; preds = %assert_end26
  %135 = getelementptr inbounds i64, i64* %24, i32 3
  %136 = load i64, i64* %135, align 8, !tbaa !98
  %137 = trunc i64 %136 to i32
  %138 = icmp eq i32 %137, 64
  br i1 %138, label %assert_end30, label %assert_fail29, !prof !1

assert_fail29:                                    ; preds = %assert_end28
  %139 = load void (i8*)*, void (i8*)** @__TVMAPISetLastError, align 4, !tbaa !2
  tail call void %139(i8* getelementptr inbounds ([96 x i8], [96 x i8]* @.str.13, i32 0, i32 0))
  ret i32 -1

assert_end30:                                     ; preds = %assert_end28
  %140 = getelementptr inbounds %1, %1* %6, i32 0, i32 6
  %141 = load i64, i64* %140, align 8
  %142 = icmp eq i64 %141, 0
  br i1 %142, label %assert_end32, label %assert_fail31, !prof !1

assert_fail31:                                    ; preds = %assert_end30
  %143 = load void (i8*)*, void (i8*)** @__TVMAPISetLastError, align 4, !tbaa !2
  tail call void %143(i8* getelementptr inbounds ([112 x i8], [112 x i8]* @.str.14, i32 0, i32 0))
  ret i32 -1

assert_end32:                                     ; preds = %assert_end30
  %144 = getelementptr inbounds %1, %1* %11, i32 0, i32 2
  %145 = load i32, i32* %144, align 4
  %146 = icmp eq i32 %145, 4
  br i1 %146, label %assert_end34, label %assert_fail33, !prof !1

assert_fail33:                                    ; preds = %assert_end32
  %147 = load void (i8*)*, void (i8*)** @__TVMAPISetLastError, align 4, !tbaa !2
  tail call void %147(i8* getelementptr inbounds ([81 x i8], [81 x i8]* @.str.15, i32 0, i32 0))
  ret i32 -1

assert_end34:                                     ; preds = %assert_end32
  %148 = getelementptr inbounds %1, %1* %11, i32 0, i32 3, i32 2
  %149 = load i16, i16* %148, align 2
  %150 = icmp eq i16 %149, 1
  %151 = getelementptr inbounds %1, %1* %11, i32 0, i32 3, i32 1
  %152 = load i8, i8* %151, align 1
  %153 = icmp eq i8 %152, 8
  %154 = getelementptr inbounds %1, %1* %11, i32 0, i32 3, i32 0
  %155 = load i8, i8* %154, align 1
  %156 = icmp eq i8 %155, 1
  %157 = and i1 %153, %156
  %158 = and i1 %150, %157
  br i1 %158, label %assert_end36, label %assert_fail35, !prof !1

assert_fail35:                                    ; preds = %assert_end34
  %159 = load void (i8*)*, void (i8*)** @__TVMAPISetLastError, align 4, !tbaa !2
  tail call void %159(i8* getelementptr inbounds ([183 x i8], [183 x i8]* @.str.16, i32 0, i32 0))
  ret i32 -1

assert_end36:                                     ; preds = %assert_end34
  %160 = load i64, i64* %52, align 8, !tbaa !100
  %161 = trunc i64 %160 to i32
  %162 = icmp eq i32 %161, 3
  br i1 %162, label %assert_end38, label %assert_fail37, !prof !1

assert_fail37:                                    ; preds = %assert_end36
  %163 = load void (i8*)*, void (i8*)** @__TVMAPISetLastError, align 4, !tbaa !2
  tail call void %163(i8* getelementptr inbounds ([95 x i8], [95 x i8]* @.str.17, i32 0, i32 0))
  ret i32 -1

assert_end38:                                     ; preds = %assert_end36
  %164 = getelementptr inbounds i64, i64* %52, i32 1
  %165 = load i64, i64* %164, align 8, !tbaa !114
  %166 = trunc i64 %165 to i32
  %167 = icmp eq i32 %166, 3
  br i1 %167, label %assert_end40, label %assert_fail39, !prof !1

assert_fail39:                                    ; preds = %assert_end38
  %168 = load void (i8*)*, void (i8*)** @__TVMAPISetLastError, align 4, !tbaa !2
  tail call void %168(i8* getelementptr inbounds ([95 x i8], [95 x i8]* @.str.18, i32 0, i32 0))
  ret i32 -1

assert_end40:                                     ; preds = %assert_end38
  %169 = getelementptr inbounds i64, i64* %52, i32 2
  %170 = load i64, i64* %169, align 8, !tbaa !116
  %171 = trunc i64 %170 to i32
  %172 = icmp eq i32 %171, 64
  br i1 %172, label %assert_end42, label %assert_fail41, !prof !1

assert_fail41:                                    ; preds = %assert_end40
  %173 = load void (i8*)*, void (i8*)** @__TVMAPISetLastError, align 4, !tbaa !2
  tail call void %173(i8* getelementptr inbounds ([96 x i8], [96 x i8]* @.str.19, i32 0, i32 0))
  ret i32 -1

assert_end42:                                     ; preds = %assert_end40
  %174 = getelementptr inbounds i64, i64* %52, i32 3
  %175 = load i64, i64* %174, align 8, !tbaa !119
  %176 = trunc i64 %175 to i32
  %177 = icmp eq i32 %176, 64
  br i1 %177, label %assert_end44, label %assert_fail43, !prof !1

assert_fail43:                                    ; preds = %assert_end42
  %178 = load void (i8*)*, void (i8*)** @__TVMAPISetLastError, align 4, !tbaa !2
  tail call void %178(i8* getelementptr inbounds ([96 x i8], [96 x i8]* @.str.20, i32 0, i32 0))
  ret i32 -1

assert_end44:                                     ; preds = %assert_end42
  %179 = getelementptr inbounds %1, %1* %11, i32 0, i32 6
  %180 = load i64, i64* %179, align 8
  %181 = icmp eq i64 %180, 0
  br i1 %181, label %assert_end46, label %assert_fail45, !prof !1

assert_fail45:                                    ; preds = %assert_end44
  %182 = load void (i8*)*, void (i8*)** @__TVMAPISetLastError, align 4, !tbaa !2
  tail call void %182(i8* getelementptr inbounds ([112 x i8], [112 x i8]* @.str.21, i32 0, i32 0))
  ret i32 -1

assert_end46:                                     ; preds = %assert_end44
  %183 = getelementptr inbounds %1, %1* %11, i32 0, i32 1, i32 0
  %184 = load i32, i32* %183, align 4
  %185 = icmp eq i32 %184, 1
  br i1 %185, label %assert_end48, label %assert_fail47, !prof !1

assert_fail47:                                    ; preds = %assert_end46
  %186 = load void (i8*)*, void (i8*)** @__TVMAPISetLastError, align 4, !tbaa !2
  tail call void %186(i8* getelementptr inbounds ([105 x i8], [105 x i8]* @.str.22, i32 0, i32 0))
  ret i32 -1

assert_end48:                                     ; preds = %assert_end46
  %187 = getelementptr inbounds %1, %1* %11, i32 0, i32 1, i32 1
  %188 = load i32, i32* %187, align 4
  %189 = icmp eq i32 %48, %188
  br i1 %189, label %assert_end50, label %assert_fail49, !prof !1

assert_fail49:                                    ; preds = %assert_end48
  %190 = load void (i8*)*, void (i8*)** @__TVMAPISetLastError, align 4, !tbaa !2
  tail call void %190(i8* getelementptr inbounds ([107 x i8], [107 x i8]* @.str.23, i32 0, i32 0))
  ret i32 -1

assert_end50:                                     ; preds = %assert_end48
  %191 = getelementptr inbounds %1, %1* %17, i32 0, i32 2
  %192 = load i32, i32* %191, align 4
  %193 = icmp eq i32 %192, 4
  br i1 %193, label %assert_end52, label %assert_fail51, !prof !1

assert_fail51:                                    ; preds = %assert_end50
  %194 = load void (i8*)*, void (i8*)** @__TVMAPISetLastError, align 4, !tbaa !2
  tail call void %194(i8* getelementptr inbounds ([81 x i8], [81 x i8]* @.str.24, i32 0, i32 0))
  ret i32 -1

assert_end52:                                     ; preds = %assert_end50
  %195 = getelementptr inbounds %1, %1* %17, i32 0, i32 3, i32 2
  %196 = load i16, i16* %195, align 2
  %197 = icmp eq i16 %196, 1
  %198 = getelementptr inbounds %1, %1* %17, i32 0, i32 3, i32 1
  %199 = load i8, i8* %198, align 1
  %200 = icmp eq i8 %199, 16
  %201 = getelementptr inbounds %1, %1* %17, i32 0, i32 3, i32 0
  %202 = load i8, i8* %201, align 1
  %203 = icmp eq i8 %202, 1
  %204 = and i1 %200, %203
  %205 = and i1 %197, %204
  br i1 %205, label %assert_end54, label %assert_fail53, !prof !1

assert_fail53:                                    ; preds = %assert_end52
  %206 = load void (i8*)*, void (i8*)** @__TVMAPISetLastError, align 4, !tbaa !2
  tail call void %206(i8* getelementptr inbounds ([185 x i8], [185 x i8]* @.str.25, i32 0, i32 0))
  ret i32 -1

assert_end54:                                     ; preds = %assert_end52
  %207 = load i64, i64* %77, align 8, !tbaa !121
  %208 = trunc i64 %207 to i32
  %209 = icmp eq i32 %208, 1
  br i1 %209, label %assert_end56, label %assert_fail55, !prof !1

assert_fail55:                                    ; preds = %assert_end54
  %210 = load void (i8*)*, void (i8*)** @__TVMAPISetLastError, align 4, !tbaa !2
  tail call void %210(i8* getelementptr inbounds ([95 x i8], [95 x i8]* @.str.26, i32 0, i32 0))
  ret i32 -1

assert_end56:                                     ; preds = %assert_end54
  %211 = getelementptr inbounds i64, i64* %77, i32 1
  %212 = load i64, i64* %211, align 8, !tbaa !135
  %213 = trunc i64 %212 to i32
  %214 = icmp eq i32 %213, 56
  br i1 %214, label %assert_end58, label %assert_fail57, !prof !1

assert_fail57:                                    ; preds = %assert_end56
  %215 = load void (i8*)*, void (i8*)** @__TVMAPISetLastError, align 4, !tbaa !2
  tail call void %215(i8* getelementptr inbounds ([96 x i8], [96 x i8]* @.str.27, i32 0, i32 0))
  ret i32 -1

assert_end58:                                     ; preds = %assert_end56
  %216 = getelementptr inbounds i64, i64* %77, i32 2
  %217 = load i64, i64* %216, align 8, !tbaa !137
  %218 = trunc i64 %217 to i32
  %219 = icmp eq i32 %218, 56
  br i1 %219, label %assert_end60, label %assert_fail59, !prof !1

assert_fail59:                                    ; preds = %assert_end58
  %220 = load void (i8*)*, void (i8*)** @__TVMAPISetLastError, align 4, !tbaa !2
  tail call void %220(i8* getelementptr inbounds ([96 x i8], [96 x i8]* @.str.28, i32 0, i32 0))
  ret i32 -1

assert_end60:                                     ; preds = %assert_end58
  %221 = getelementptr inbounds i64, i64* %77, i32 3
  %222 = load i64, i64* %221, align 8, !tbaa !140
  %223 = trunc i64 %222 to i32
  %224 = icmp eq i32 %223, 64
  br i1 %224, label %assert_end62, label %assert_fail61, !prof !1

assert_fail61:                                    ; preds = %assert_end60
  %225 = load void (i8*)*, void (i8*)** @__TVMAPISetLastError, align 4, !tbaa !2
  tail call void %225(i8* getelementptr inbounds ([96 x i8], [96 x i8]* @.str.29, i32 0, i32 0))
  ret i32 -1

assert_end62:                                     ; preds = %assert_end60
  %226 = getelementptr inbounds %1, %1* %17, i32 0, i32 6
  %227 = load i64, i64* %226, align 8
  %228 = icmp eq i64 %227, 0
  br i1 %228, label %assert_end64, label %assert_fail63, !prof !1

assert_fail63:                                    ; preds = %assert_end62
  %229 = load void (i8*)*, void (i8*)** @__TVMAPISetLastError, align 4, !tbaa !2
  tail call void %229(i8* getelementptr inbounds ([112 x i8], [112 x i8]* @.str.30, i32 0, i32 0))
  ret i32 -1

assert_end64:                                     ; preds = %assert_end62
  %230 = getelementptr inbounds %1, %1* %17, i32 0, i32 1, i32 0
  %231 = load i32, i32* %230, align 4
  %232 = icmp eq i32 %231, 1
  br i1 %232, label %assert_end66, label %assert_fail65, !prof !1

assert_fail65:                                    ; preds = %assert_end64
  %233 = load void (i8*)*, void (i8*)** @__TVMAPISetLastError, align 4, !tbaa !2
  tail call void %233(i8* getelementptr inbounds ([105 x i8], [105 x i8]* @.str.31, i32 0, i32 0))
  ret i32 -1

assert_end66:                                     ; preds = %assert_end64
  %234 = getelementptr inbounds %1, %1* %17, i32 0, i32 1, i32 1
  %235 = load i32, i32* %234, align 4
  %236 = icmp eq i32 %48, %235
  br i1 %236, label %if_end70, label %assert_fail67, !prof !1

assert_fail67:                                    ; preds = %assert_end66
  %237 = load void (i8*)*, void (i8*)** @__TVMAPISetLastError, align 4, !tbaa !2
  tail call void %237(i8* getelementptr inbounds ([107 x i8], [107 x i8]* @.str.32, i32 0, i32 0))
  ret i32 -1

if_end70:                                         ; preds = %assert_end66
  %238 = tail call fastcc i32 @default_function_compute_(i8* %50, i8* %22, i8* %75, i32 %48)
  ret i32 %238
}

; Function Attrs: noinline
define private fastcc i32 @default_function_compute_(i8* noalias nocapture readonly, i8* noalias nocapture readonly, i8* noalias, i32) unnamed_addr #0 {
entry:
  %4 = load i8* (i32, i32, i64, i32, i32)*, i8* (i32, i32, i64, i32, i32)** @__TVMBackendAllocWorkspace, align 4, !tbaa !2
  %5 = tail call i8* %4(i32 1, i32 %3, i64 26912, i32 1, i32 8)
  %6 = load i8* (i32, i32, i64, i32, i32)*, i8* (i32, i32, i64, i32, i32)** @__TVMBackendAllocWorkspace, align 4, !tbaa !2
  %7 = tail call i8* %6(i32 1, i32 %3, i64 4608, i32 1, i32 8)
  %8 = load i8* (i32, i32, i64, i32, i32)*, i8* (i32, i32, i64, i32, i32)** @__TVMBackendAllocWorkspace, align 4, !tbaa !2
  %9 = tail call i8* %8(i32 1, i32 %3, i64 225792, i32 1, i32 8)
  br label %for_begin5.preheader

for_begin5.preheader:                             ; preds = %for_end10.2, %entry
  %10 = phi i32 [ 0, %entry ], [ %547, %for_end10.2 ]
  %11 = mul i32 %10, 24
  br label %for_begin11.preheader

for_end:                                          ; preds = %for_end10.2
  %12 = alloca %4, align 4
  %13 = getelementptr inbounds %4, %4* %12, i32 0, i32 0
  store i8* %7, i8** %13, align 4
  %14 = getelementptr inbounds %4, %4* %12, i32 0, i32 1
  store i8* %5, i8** %14, align 4
  %15 = bitcast %4* %12 to i8*
  %16 = load i32 (i32 (i32, %0*, i8*)*, i8*, i32)*, i32 (i32 (i32, %0*, i8*)*, i8*, i32)** @__TVMBackendParallelLaunch, align 4, !tbaa !2
  %17 = call i32 %16(i32 (i32, %0*, i8*)* nonnull @__tvm_parallel_lambda, i8* nonnull %15, i32 0)
  %18 = icmp eq i32 %17, 0
  br i1 %18, label %for_begin17.preheader, label %call_fail, !prof !1

for_begin11.preheader:                            ; preds = %for_end13, %for_begin5.preheader
  %19 = phi i32 [ 0, %for_begin5.preheader ], [ %75, %for_end13 ]
  %20 = add nuw nsw i32 %19, %11
  %21 = shl i32 %20, 6
  %22 = shl nsw i32 %20, 9
  br label %vector.body77

vector.body77:                                    ; preds = %vector.body77, %for_begin11.preheader
  %index81 = phi i32 [ 0, %for_begin11.preheader ], [ %index.next82, %vector.body77 ]
  %23 = add nuw nsw i32 %index81, %21
  %24 = add nuw nsw i32 %index81, %22
  %25 = add nuw nsw i32 %24, 448
  %26 = getelementptr inbounds i8, i8* %0, i32 %25
  %27 = bitcast i8* %26 to <4 x i8>*
  %wide.load88 = load <4 x i8>, <4 x i8>* %27, align 1, !tbaa !142
  %28 = and <4 x i8> %wide.load88, <i8 1, i8 1, i8 1, i8 1>
  %29 = add nuw nsw i32 %24, 384
  %30 = getelementptr inbounds i8, i8* %0, i32 %29
  %31 = bitcast i8* %30 to <4 x i8>*
  %wide.load89 = load <4 x i8>, <4 x i8>* %31, align 1, !tbaa !142
  %32 = and <4 x i8> %wide.load89, <i8 1, i8 1, i8 1, i8 1>
  %33 = add nuw nsw i32 %24, 320
  %34 = getelementptr inbounds i8, i8* %0, i32 %33
  %35 = bitcast i8* %34 to <4 x i8>*
  %wide.load90 = load <4 x i8>, <4 x i8>* %35, align 1, !tbaa !142
  %36 = and <4 x i8> %wide.load90, <i8 1, i8 1, i8 1, i8 1>
  %37 = add nuw nsw i32 %24, 256
  %38 = getelementptr inbounds i8, i8* %0, i32 %37
  %39 = bitcast i8* %38 to <4 x i8>*
  %wide.load91 = load <4 x i8>, <4 x i8>* %39, align 1, !tbaa !142
  %40 = and <4 x i8> %wide.load91, <i8 1, i8 1, i8 1, i8 1>
  %41 = add nuw nsw i32 %24, 192
  %42 = getelementptr inbounds i8, i8* %0, i32 %41
  %43 = bitcast i8* %42 to <4 x i8>*
  %wide.load92 = load <4 x i8>, <4 x i8>* %43, align 1, !tbaa !142
  %44 = and <4 x i8> %wide.load92, <i8 1, i8 1, i8 1, i8 1>
  %45 = add nuw nsw i32 %24, 128
  %46 = getelementptr inbounds i8, i8* %0, i32 %45
  %47 = bitcast i8* %46 to <4 x i8>*
  %wide.load93 = load <4 x i8>, <4 x i8>* %47, align 1, !tbaa !142
  %48 = and <4 x i8> %wide.load93, <i8 1, i8 1, i8 1, i8 1>
  %49 = add nuw nsw i32 %24, 64
  %50 = getelementptr inbounds i8, i8* %0, i32 %49
  %51 = bitcast i8* %50 to <4 x i8>*
  %wide.load94 = load <4 x i8>, <4 x i8>* %51, align 1, !tbaa !142
  %52 = and <4 x i8> %wide.load94, <i8 1, i8 1, i8 1, i8 1>
  %53 = getelementptr inbounds i8, i8* %0, i32 %24
  %54 = bitcast i8* %53 to <4 x i8>*
  %wide.load95 = load <4 x i8>, <4 x i8>* %54, align 1, !tbaa !142
  %55 = shl <4 x i8> %wide.load95, <i8 1, i8 1, i8 1, i8 1>
  %56 = and <4 x i8> %55, <i8 2, i8 2, i8 2, i8 2>
  %57 = or <4 x i8> %56, %52
  %58 = shl nuw nsw <4 x i8> %57, <i8 1, i8 1, i8 1, i8 1>
  %59 = or <4 x i8> %58, %48
  %60 = shl nuw nsw <4 x i8> %59, <i8 1, i8 1, i8 1, i8 1>
  %61 = or <4 x i8> %60, %44
  %62 = shl nuw nsw <4 x i8> %61, <i8 1, i8 1, i8 1, i8 1>
  %63 = or <4 x i8> %62, %40
  %64 = shl <4 x i8> %63, <i8 1, i8 1, i8 1, i8 1>
  %65 = or <4 x i8> %64, %36
  %66 = shl <4 x i8> %65, <i8 1, i8 1, i8 1, i8 1>
  %67 = or <4 x i8> %66, %32
  %68 = shl <4 x i8> %67, <i8 1, i8 1, i8 1, i8 1>
  %69 = or <4 x i8> %68, %28
  %70 = getelementptr inbounds i8, i8* %5, i32 %23
  %71 = bitcast i8* %70 to <4 x i8>*
  store <4 x i8> %69, <4 x i8>* %71, align 1, !tbaa !145
  %index.next82 = add i32 %index81, 4
  %72 = icmp eq i32 %index.next82, 64
  br i1 %72, label %for_end13, label %vector.body77, !llvm.loop !148

for_end10:                                        ; preds = %for_end13
  %73 = mul i32 %10, 24
  %74 = add i32 %73, 8
  br label %for_begin11.preheader.1

for_end13:                                        ; preds = %vector.body77
  %75 = add nuw nsw i32 %19, 1
  %exitcond48 = icmp eq i32 %75, 8
  br i1 %exitcond48, label %for_end10, label %for_begin11.preheader, !prof !24

call_fail:                                        ; preds = %call_end32, %for_end16, %call_end34, %for_end
  %merge = phi i32 [ %17, %for_end ], [ 0, %call_end34 ], [ %85, %for_end16 ], [ %427, %call_end32 ]
  ret i32 %merge

for_begin17.preheader:                            ; preds = %for_end, %for_end19
  %76 = phi i32 [ %419, %for_end19 ], [ 0, %for_end ]
  %.off = add nsw i32 %76, -1
  %77 = icmp ult i32 %.off, 56
  %78 = mul nuw nsw i32 %76, 58
  %79 = mul nuw nsw i32 %76, 56
  br label %for_begin20.preheader

for_end16:                                        ; preds = %for_end19
  %80 = alloca %5, align 4
  %81 = getelementptr inbounds %5, %5* %80, i32 0, i32 0
  store i8* %9, i8** %81, align 4
  %82 = getelementptr inbounds %5, %5* %80, i32 0, i32 1
  store i8* %5, i8** %82, align 4
  %83 = bitcast %5* %80 to i8*
  %84 = load i32 (i32 (i32, %0*, i8*)*, i8*, i32)*, i32 (i32 (i32, %0*, i8*)*, i8*, i32)** @__TVMBackendParallelLaunch, align 4, !tbaa !2
  %85 = call i32 %84(i32 (i32, %0*, i8*)* nonnull @__tvm_parallel_lambda.34, i8* nonnull %83, i32 0)
  %86 = icmp eq i32 %85, 0
  br i1 %86, label %call_end32, label %call_fail, !prof !1

for_begin20.preheader:                            ; preds = %for_end22, %for_begin17.preheader
  %87 = phi i32 [ 0, %for_begin17.preheader ], [ %420, %for_end22 ]
  %.off43 = add nsw i32 %87, -1
  %88 = icmp ult i32 %.off43, 56
  %89 = and i1 %77, %88
  %90 = add nuw nsw i32 %87, %78
  %91 = shl i32 %90, 3
  br i1 %89, label %vector.ph99, label %for_end22, !prof !150

vector.ph99:                                      ; preds = %for_begin20.preheader
  %92 = add nuw nsw i32 %87, %79
  %93 = shl i32 %92, 3
  %broadcast.splatinsert104 = insertelement <4 x i32> undef, i32 %93, i32 0
  %broadcast.splat105 = shufflevector <4 x i32> %broadcast.splatinsert104, <4 x i32> undef, <4 x i32> zeroinitializer
  %94 = add nuw nsw <4 x i32> %broadcast.splat105, <i32 0, i32 1, i32 2, i32 3>
  %95 = shl nsw <4 x i32> %94, <i32 3, i32 3, i32 3, i32 3>
  %96 = add nsw <4 x i32> %95, <i32 -3641, i32 -3641, i32 -3641, i32 -3641>
  %97 = extractelement <4 x i32> %96, i32 0
  %98 = getelementptr inbounds i8, i8* %1, i32 %97
  %99 = extractelement <4 x i32> %96, i32 1
  %100 = getelementptr inbounds i8, i8* %1, i32 %99
  %101 = extractelement <4 x i32> %96, i32 2
  %102 = getelementptr inbounds i8, i8* %1, i32 %101
  %103 = extractelement <4 x i32> %96, i32 3
  %104 = getelementptr inbounds i8, i8* %1, i32 %103
  %105 = load i8, i8* %98, align 1, !tbaa !151
  %106 = load i8, i8* %100, align 1, !tbaa !151
  %107 = load i8, i8* %102, align 1, !tbaa !151
  %108 = load i8, i8* %104, align 1, !tbaa !151
  %109 = insertelement <4 x i8> undef, i8 %105, i32 0
  %110 = insertelement <4 x i8> %109, i8 %106, i32 1
  %111 = insertelement <4 x i8> %110, i8 %107, i32 2
  %112 = insertelement <4 x i8> %111, i8 %108, i32 3
  %113 = and <4 x i8> %112, <i8 1, i8 1, i8 1, i8 1>
  %114 = add nsw <4 x i32> %95, <i32 -3642, i32 -3642, i32 -3642, i32 -3642>
  %115 = extractelement <4 x i32> %114, i32 0
  %116 = getelementptr inbounds i8, i8* %1, i32 %115
  %117 = extractelement <4 x i32> %114, i32 1
  %118 = getelementptr inbounds i8, i8* %1, i32 %117
  %119 = extractelement <4 x i32> %114, i32 2
  %120 = getelementptr inbounds i8, i8* %1, i32 %119
  %121 = extractelement <4 x i32> %114, i32 3
  %122 = getelementptr inbounds i8, i8* %1, i32 %121
  %123 = load i8, i8* %116, align 2, !tbaa !151
  %124 = load i8, i8* %118, align 2, !tbaa !151
  %125 = load i8, i8* %120, align 2, !tbaa !151
  %126 = load i8, i8* %122, align 2, !tbaa !151
  %127 = insertelement <4 x i8> undef, i8 %123, i32 0
  %128 = insertelement <4 x i8> %127, i8 %124, i32 1
  %129 = insertelement <4 x i8> %128, i8 %125, i32 2
  %130 = insertelement <4 x i8> %129, i8 %126, i32 3
  %131 = and <4 x i8> %130, <i8 1, i8 1, i8 1, i8 1>
  %132 = add nsw <4 x i32> %95, <i32 -3643, i32 -3643, i32 -3643, i32 -3643>
  %133 = extractelement <4 x i32> %132, i32 0
  %134 = getelementptr inbounds i8, i8* %1, i32 %133
  %135 = extractelement <4 x i32> %132, i32 1
  %136 = getelementptr inbounds i8, i8* %1, i32 %135
  %137 = extractelement <4 x i32> %132, i32 2
  %138 = getelementptr inbounds i8, i8* %1, i32 %137
  %139 = extractelement <4 x i32> %132, i32 3
  %140 = getelementptr inbounds i8, i8* %1, i32 %139
  %141 = load i8, i8* %134, align 1, !tbaa !151
  %142 = load i8, i8* %136, align 1, !tbaa !151
  %143 = load i8, i8* %138, align 1, !tbaa !151
  %144 = load i8, i8* %140, align 1, !tbaa !151
  %145 = insertelement <4 x i8> undef, i8 %141, i32 0
  %146 = insertelement <4 x i8> %145, i8 %142, i32 1
  %147 = insertelement <4 x i8> %146, i8 %143, i32 2
  %148 = insertelement <4 x i8> %147, i8 %144, i32 3
  %149 = and <4 x i8> %148, <i8 1, i8 1, i8 1, i8 1>
  %150 = add nsw <4 x i32> %95, <i32 -3644, i32 -3644, i32 -3644, i32 -3644>
  %151 = extractelement <4 x i32> %150, i32 0
  %152 = getelementptr inbounds i8, i8* %1, i32 %151
  %153 = extractelement <4 x i32> %150, i32 1
  %154 = getelementptr inbounds i8, i8* %1, i32 %153
  %155 = extractelement <4 x i32> %150, i32 2
  %156 = getelementptr inbounds i8, i8* %1, i32 %155
  %157 = extractelement <4 x i32> %150, i32 3
  %158 = getelementptr inbounds i8, i8* %1, i32 %157
  %159 = load i8, i8* %152, align 4, !tbaa !151
  %160 = load i8, i8* %154, align 4, !tbaa !151
  %161 = load i8, i8* %156, align 4, !tbaa !151
  %162 = load i8, i8* %158, align 4, !tbaa !151
  %163 = insertelement <4 x i8> undef, i8 %159, i32 0
  %164 = insertelement <4 x i8> %163, i8 %160, i32 1
  %165 = insertelement <4 x i8> %164, i8 %161, i32 2
  %166 = insertelement <4 x i8> %165, i8 %162, i32 3
  %167 = and <4 x i8> %166, <i8 1, i8 1, i8 1, i8 1>
  %168 = add nsw <4 x i32> %95, <i32 -3645, i32 -3645, i32 -3645, i32 -3645>
  %169 = extractelement <4 x i32> %168, i32 0
  %170 = getelementptr inbounds i8, i8* %1, i32 %169
  %171 = extractelement <4 x i32> %168, i32 1
  %172 = getelementptr inbounds i8, i8* %1, i32 %171
  %173 = extractelement <4 x i32> %168, i32 2
  %174 = getelementptr inbounds i8, i8* %1, i32 %173
  %175 = extractelement <4 x i32> %168, i32 3
  %176 = getelementptr inbounds i8, i8* %1, i32 %175
  %177 = load i8, i8* %170, align 1, !tbaa !151
  %178 = load i8, i8* %172, align 1, !tbaa !151
  %179 = load i8, i8* %174, align 1, !tbaa !151
  %180 = load i8, i8* %176, align 1, !tbaa !151
  %181 = insertelement <4 x i8> undef, i8 %177, i32 0
  %182 = insertelement <4 x i8> %181, i8 %178, i32 1
  %183 = insertelement <4 x i8> %182, i8 %179, i32 2
  %184 = insertelement <4 x i8> %183, i8 %180, i32 3
  %185 = and <4 x i8> %184, <i8 1, i8 1, i8 1, i8 1>
  %186 = add nsw <4 x i32> %95, <i32 -3646, i32 -3646, i32 -3646, i32 -3646>
  %187 = extractelement <4 x i32> %186, i32 0
  %188 = getelementptr inbounds i8, i8* %1, i32 %187
  %189 = extractelement <4 x i32> %186, i32 1
  %190 = getelementptr inbounds i8, i8* %1, i32 %189
  %191 = extractelement <4 x i32> %186, i32 2
  %192 = getelementptr inbounds i8, i8* %1, i32 %191
  %193 = extractelement <4 x i32> %186, i32 3
  %194 = getelementptr inbounds i8, i8* %1, i32 %193
  %195 = load i8, i8* %188, align 2, !tbaa !151
  %196 = load i8, i8* %190, align 2, !tbaa !151
  %197 = load i8, i8* %192, align 2, !tbaa !151
  %198 = load i8, i8* %194, align 2, !tbaa !151
  %199 = insertelement <4 x i8> undef, i8 %195, i32 0
  %200 = insertelement <4 x i8> %199, i8 %196, i32 1
  %201 = insertelement <4 x i8> %200, i8 %197, i32 2
  %202 = insertelement <4 x i8> %201, i8 %198, i32 3
  %203 = and <4 x i8> %202, <i8 1, i8 1, i8 1, i8 1>
  %204 = add nsw <4 x i32> %95, <i32 -3647, i32 -3647, i32 -3647, i32 -3647>
  %205 = extractelement <4 x i32> %204, i32 0
  %206 = getelementptr inbounds i8, i8* %1, i32 %205
  %207 = extractelement <4 x i32> %204, i32 1
  %208 = getelementptr inbounds i8, i8* %1, i32 %207
  %209 = extractelement <4 x i32> %204, i32 2
  %210 = getelementptr inbounds i8, i8* %1, i32 %209
  %211 = extractelement <4 x i32> %204, i32 3
  %212 = getelementptr inbounds i8, i8* %1, i32 %211
  %213 = load i8, i8* %206, align 1, !tbaa !151
  %214 = load i8, i8* %208, align 1, !tbaa !151
  %215 = load i8, i8* %210, align 1, !tbaa !151
  %216 = load i8, i8* %212, align 1, !tbaa !151
  %217 = insertelement <4 x i8> undef, i8 %213, i32 0
  %218 = insertelement <4 x i8> %217, i8 %214, i32 1
  %219 = insertelement <4 x i8> %218, i8 %215, i32 2
  %220 = insertelement <4 x i8> %219, i8 %216, i32 3
  %221 = and <4 x i8> %220, <i8 1, i8 1, i8 1, i8 1>
  %222 = add nsw <4 x i32> %95, <i32 -3648, i32 -3648, i32 -3648, i32 -3648>
  %223 = extractelement <4 x i32> %222, i32 0
  %224 = getelementptr inbounds i8, i8* %1, i32 %223
  %225 = extractelement <4 x i32> %222, i32 1
  %226 = getelementptr inbounds i8, i8* %1, i32 %225
  %227 = extractelement <4 x i32> %222, i32 2
  %228 = getelementptr inbounds i8, i8* %1, i32 %227
  %229 = extractelement <4 x i32> %222, i32 3
  %230 = getelementptr inbounds i8, i8* %1, i32 %229
  %231 = load i8, i8* %224, align 8, !tbaa !151
  %232 = load i8, i8* %226, align 8, !tbaa !151
  %233 = load i8, i8* %228, align 8, !tbaa !151
  %234 = load i8, i8* %230, align 8, !tbaa !151
  %235 = insertelement <4 x i8> undef, i8 %231, i32 0
  %236 = insertelement <4 x i8> %235, i8 %232, i32 1
  %237 = insertelement <4 x i8> %236, i8 %233, i32 2
  %238 = insertelement <4 x i8> %237, i8 %234, i32 3
  %239 = shl <4 x i8> %238, <i8 1, i8 1, i8 1, i8 1>
  %240 = and <4 x i8> %239, <i8 2, i8 2, i8 2, i8 2>
  %241 = or <4 x i8> %240, %221
  %242 = shl nuw nsw <4 x i8> %241, <i8 1, i8 1, i8 1, i8 1>
  %243 = or <4 x i8> %242, %203
  %244 = shl nuw nsw <4 x i8> %243, <i8 1, i8 1, i8 1, i8 1>
  %245 = or <4 x i8> %244, %185
  %246 = shl nuw nsw <4 x i8> %245, <i8 1, i8 1, i8 1, i8 1>
  %247 = or <4 x i8> %246, %167
  %248 = shl <4 x i8> %247, <i8 1, i8 1, i8 1, i8 1>
  %249 = or <4 x i8> %248, %149
  %250 = shl <4 x i8> %249, <i8 1, i8 1, i8 1, i8 1>
  %251 = or <4 x i8> %250, %131
  %252 = shl <4 x i8> %251, <i8 1, i8 1, i8 1, i8 1>
  %253 = or <4 x i8> %252, %113
  %254 = getelementptr inbounds i8, i8* %5, i32 %91
  %255 = bitcast i8* %254 to <4 x i8>*
  store <4 x i8> %253, <4 x i8>* %255, align 1, !tbaa !145
  %256 = or i32 %91, 4
  %257 = add nuw nsw <4 x i32> %broadcast.splat105, <i32 4, i32 5, i32 6, i32 7>
  %258 = shl nsw <4 x i32> %257, <i32 3, i32 3, i32 3, i32 3>
  %259 = add nsw <4 x i32> %258, <i32 -3641, i32 -3641, i32 -3641, i32 -3641>
  %260 = extractelement <4 x i32> %259, i32 0
  %261 = getelementptr inbounds i8, i8* %1, i32 %260
  %262 = extractelement <4 x i32> %259, i32 1
  %263 = getelementptr inbounds i8, i8* %1, i32 %262
  %264 = extractelement <4 x i32> %259, i32 2
  %265 = getelementptr inbounds i8, i8* %1, i32 %264
  %266 = extractelement <4 x i32> %259, i32 3
  %267 = getelementptr inbounds i8, i8* %1, i32 %266
  %268 = load i8, i8* %261, align 1, !tbaa !151
  %269 = load i8, i8* %263, align 1, !tbaa !151
  %270 = load i8, i8* %265, align 1, !tbaa !151
  %271 = load i8, i8* %267, align 1, !tbaa !151
  %272 = insertelement <4 x i8> undef, i8 %268, i32 0
  %273 = insertelement <4 x i8> %272, i8 %269, i32 1
  %274 = insertelement <4 x i8> %273, i8 %270, i32 2
  %275 = insertelement <4 x i8> %274, i8 %271, i32 3
  %276 = and <4 x i8> %275, <i8 1, i8 1, i8 1, i8 1>
  %277 = add nsw <4 x i32> %258, <i32 -3642, i32 -3642, i32 -3642, i32 -3642>
  %278 = extractelement <4 x i32> %277, i32 0
  %279 = getelementptr inbounds i8, i8* %1, i32 %278
  %280 = extractelement <4 x i32> %277, i32 1
  %281 = getelementptr inbounds i8, i8* %1, i32 %280
  %282 = extractelement <4 x i32> %277, i32 2
  %283 = getelementptr inbounds i8, i8* %1, i32 %282
  %284 = extractelement <4 x i32> %277, i32 3
  %285 = getelementptr inbounds i8, i8* %1, i32 %284
  %286 = load i8, i8* %279, align 2, !tbaa !151
  %287 = load i8, i8* %281, align 2, !tbaa !151
  %288 = load i8, i8* %283, align 2, !tbaa !151
  %289 = load i8, i8* %285, align 2, !tbaa !151
  %290 = insertelement <4 x i8> undef, i8 %286, i32 0
  %291 = insertelement <4 x i8> %290, i8 %287, i32 1
  %292 = insertelement <4 x i8> %291, i8 %288, i32 2
  %293 = insertelement <4 x i8> %292, i8 %289, i32 3
  %294 = and <4 x i8> %293, <i8 1, i8 1, i8 1, i8 1>
  %295 = add nsw <4 x i32> %258, <i32 -3643, i32 -3643, i32 -3643, i32 -3643>
  %296 = extractelement <4 x i32> %295, i32 0
  %297 = getelementptr inbounds i8, i8* %1, i32 %296
  %298 = extractelement <4 x i32> %295, i32 1
  %299 = getelementptr inbounds i8, i8* %1, i32 %298
  %300 = extractelement <4 x i32> %295, i32 2
  %301 = getelementptr inbounds i8, i8* %1, i32 %300
  %302 = extractelement <4 x i32> %295, i32 3
  %303 = getelementptr inbounds i8, i8* %1, i32 %302
  %304 = load i8, i8* %297, align 1, !tbaa !151
  %305 = load i8, i8* %299, align 1, !tbaa !151
  %306 = load i8, i8* %301, align 1, !tbaa !151
  %307 = load i8, i8* %303, align 1, !tbaa !151
  %308 = insertelement <4 x i8> undef, i8 %304, i32 0
  %309 = insertelement <4 x i8> %308, i8 %305, i32 1
  %310 = insertelement <4 x i8> %309, i8 %306, i32 2
  %311 = insertelement <4 x i8> %310, i8 %307, i32 3
  %312 = and <4 x i8> %311, <i8 1, i8 1, i8 1, i8 1>
  %313 = add nsw <4 x i32> %258, <i32 -3644, i32 -3644, i32 -3644, i32 -3644>
  %314 = extractelement <4 x i32> %313, i32 0
  %315 = getelementptr inbounds i8, i8* %1, i32 %314
  %316 = extractelement <4 x i32> %313, i32 1
  %317 = getelementptr inbounds i8, i8* %1, i32 %316
  %318 = extractelement <4 x i32> %313, i32 2
  %319 = getelementptr inbounds i8, i8* %1, i32 %318
  %320 = extractelement <4 x i32> %313, i32 3
  %321 = getelementptr inbounds i8, i8* %1, i32 %320
  %322 = load i8, i8* %315, align 4, !tbaa !151
  %323 = load i8, i8* %317, align 4, !tbaa !151
  %324 = load i8, i8* %319, align 4, !tbaa !151
  %325 = load i8, i8* %321, align 4, !tbaa !151
  %326 = insertelement <4 x i8> undef, i8 %322, i32 0
  %327 = insertelement <4 x i8> %326, i8 %323, i32 1
  %328 = insertelement <4 x i8> %327, i8 %324, i32 2
  %329 = insertelement <4 x i8> %328, i8 %325, i32 3
  %330 = and <4 x i8> %329, <i8 1, i8 1, i8 1, i8 1>
  %331 = add nsw <4 x i32> %258, <i32 -3645, i32 -3645, i32 -3645, i32 -3645>
  %332 = extractelement <4 x i32> %331, i32 0
  %333 = getelementptr inbounds i8, i8* %1, i32 %332
  %334 = extractelement <4 x i32> %331, i32 1
  %335 = getelementptr inbounds i8, i8* %1, i32 %334
  %336 = extractelement <4 x i32> %331, i32 2
  %337 = getelementptr inbounds i8, i8* %1, i32 %336
  %338 = extractelement <4 x i32> %331, i32 3
  %339 = getelementptr inbounds i8, i8* %1, i32 %338
  %340 = load i8, i8* %333, align 1, !tbaa !151
  %341 = load i8, i8* %335, align 1, !tbaa !151
  %342 = load i8, i8* %337, align 1, !tbaa !151
  %343 = load i8, i8* %339, align 1, !tbaa !151
  %344 = insertelement <4 x i8> undef, i8 %340, i32 0
  %345 = insertelement <4 x i8> %344, i8 %341, i32 1
  %346 = insertelement <4 x i8> %345, i8 %342, i32 2
  %347 = insertelement <4 x i8> %346, i8 %343, i32 3
  %348 = and <4 x i8> %347, <i8 1, i8 1, i8 1, i8 1>
  %349 = add nsw <4 x i32> %258, <i32 -3646, i32 -3646, i32 -3646, i32 -3646>
  %350 = extractelement <4 x i32> %349, i32 0
  %351 = getelementptr inbounds i8, i8* %1, i32 %350
  %352 = extractelement <4 x i32> %349, i32 1
  %353 = getelementptr inbounds i8, i8* %1, i32 %352
  %354 = extractelement <4 x i32> %349, i32 2
  %355 = getelementptr inbounds i8, i8* %1, i32 %354
  %356 = extractelement <4 x i32> %349, i32 3
  %357 = getelementptr inbounds i8, i8* %1, i32 %356
  %358 = load i8, i8* %351, align 2, !tbaa !151
  %359 = load i8, i8* %353, align 2, !tbaa !151
  %360 = load i8, i8* %355, align 2, !tbaa !151
  %361 = load i8, i8* %357, align 2, !tbaa !151
  %362 = insertelement <4 x i8> undef, i8 %358, i32 0
  %363 = insertelement <4 x i8> %362, i8 %359, i32 1
  %364 = insertelement <4 x i8> %363, i8 %360, i32 2
  %365 = insertelement <4 x i8> %364, i8 %361, i32 3
  %366 = and <4 x i8> %365, <i8 1, i8 1, i8 1, i8 1>
  %367 = add nsw <4 x i32> %258, <i32 -3647, i32 -3647, i32 -3647, i32 -3647>
  %368 = extractelement <4 x i32> %367, i32 0
  %369 = getelementptr inbounds i8, i8* %1, i32 %368
  %370 = extractelement <4 x i32> %367, i32 1
  %371 = getelementptr inbounds i8, i8* %1, i32 %370
  %372 = extractelement <4 x i32> %367, i32 2
  %373 = getelementptr inbounds i8, i8* %1, i32 %372
  %374 = extractelement <4 x i32> %367, i32 3
  %375 = getelementptr inbounds i8, i8* %1, i32 %374
  %376 = load i8, i8* %369, align 1, !tbaa !151
  %377 = load i8, i8* %371, align 1, !tbaa !151
  %378 = load i8, i8* %373, align 1, !tbaa !151
  %379 = load i8, i8* %375, align 1, !tbaa !151
  %380 = insertelement <4 x i8> undef, i8 %376, i32 0
  %381 = insertelement <4 x i8> %380, i8 %377, i32 1
  %382 = insertelement <4 x i8> %381, i8 %378, i32 2
  %383 = insertelement <4 x i8> %382, i8 %379, i32 3
  %384 = and <4 x i8> %383, <i8 1, i8 1, i8 1, i8 1>
  %385 = add nsw <4 x i32> %258, <i32 -3648, i32 -3648, i32 -3648, i32 -3648>
  %386 = extractelement <4 x i32> %385, i32 0
  %387 = getelementptr inbounds i8, i8* %1, i32 %386
  %388 = extractelement <4 x i32> %385, i32 1
  %389 = getelementptr inbounds i8, i8* %1, i32 %388
  %390 = extractelement <4 x i32> %385, i32 2
  %391 = getelementptr inbounds i8, i8* %1, i32 %390
  %392 = extractelement <4 x i32> %385, i32 3
  %393 = getelementptr inbounds i8, i8* %1, i32 %392
  %394 = load i8, i8* %387, align 8, !tbaa !151
  %395 = load i8, i8* %389, align 8, !tbaa !151
  %396 = load i8, i8* %391, align 8, !tbaa !151
  %397 = load i8, i8* %393, align 8, !tbaa !151
  %398 = insertelement <4 x i8> undef, i8 %394, i32 0
  %399 = insertelement <4 x i8> %398, i8 %395, i32 1
  %400 = insertelement <4 x i8> %399, i8 %396, i32 2
  %401 = insertelement <4 x i8> %400, i8 %397, i32 3
  %402 = shl <4 x i8> %401, <i8 1, i8 1, i8 1, i8 1>
  %403 = and <4 x i8> %402, <i8 2, i8 2, i8 2, i8 2>
  %404 = or <4 x i8> %403, %384
  %405 = shl nuw nsw <4 x i8> %404, <i8 1, i8 1, i8 1, i8 1>
  %406 = or <4 x i8> %405, %366
  %407 = shl nuw nsw <4 x i8> %406, <i8 1, i8 1, i8 1, i8 1>
  %408 = or <4 x i8> %407, %348
  %409 = shl nuw nsw <4 x i8> %408, <i8 1, i8 1, i8 1, i8 1>
  %410 = or <4 x i8> %409, %330
  %411 = shl <4 x i8> %410, <i8 1, i8 1, i8 1, i8 1>
  %412 = or <4 x i8> %411, %312
  %413 = shl <4 x i8> %412, <i8 1, i8 1, i8 1, i8 1>
  %414 = or <4 x i8> %413, %294
  %415 = shl <4 x i8> %414, <i8 1, i8 1, i8 1, i8 1>
  %416 = or <4 x i8> %415, %276
  %417 = getelementptr inbounds i8, i8* %5, i32 %256
  %418 = bitcast i8* %417 to <4 x i8>*
  store <4 x i8> %416, <4 x i8>* %418, align 1, !tbaa !145
  br label %for_end22

for_end19:                                        ; preds = %for_end22
  %419 = add nuw nsw i32 %76, 1
  %exitcond46 = icmp eq i32 %419, 58
  br i1 %exitcond46, label %for_end16, label %for_begin17.preheader, !prof !24

for_end22:                                        ; preds = %vector.ph99, %for_begin20.preheader
  %420 = add nuw nsw i32 %87, 1
  %exitcond45 = icmp eq i32 %420, 58
  br i1 %exitcond45, label %for_end19, label %for_begin20.preheader, !prof !24

call_end32:                                       ; preds = %for_end16
  %421 = alloca %6, align 4
  %422 = getelementptr inbounds %6, %6* %421, i32 0, i32 0
  store i8* %7, i8** %422, align 4
  %423 = getelementptr inbounds %6, %6* %421, i32 0, i32 1
  store i8* %9, i8** %423, align 4
  %424 = getelementptr inbounds %6, %6* %421, i32 0, i32 2
  store i8* %2, i8** %424, align 4
  %425 = bitcast %6* %421 to i8*
  %426 = load i32 (i32 (i32, %0*, i8*)*, i8*, i32)*, i32 (i32 (i32, %0*, i8*)*, i8*, i32)** @__TVMBackendParallelLaunch, align 4, !tbaa !2
  %427 = call i32 %426(i32 (i32, %0*, i8*)* nonnull @__tvm_parallel_lambda.35, i8* nonnull %425, i32 0)
  %428 = icmp eq i32 %427, 0
  br i1 %428, label %call_end34, label %call_fail, !prof !1

call_end34:                                       ; preds = %call_end32
  %429 = load i32 (i32, i32, i8*)*, i32 (i32, i32, i8*)** @__TVMBackendFreeWorkspace, align 4, !tbaa !2
  %430 = call i32 %429(i32 1, i32 %3, i8* %9)
  %431 = load i32 (i32, i32, i8*)*, i32 (i32, i32, i8*)** @__TVMBackendFreeWorkspace, align 4, !tbaa !2
  %432 = call i32 %431(i32 1, i32 %3, i8* %7)
  %433 = load i32 (i32, i32, i8*)*, i32 (i32, i32, i8*)** @__TVMBackendFreeWorkspace, align 4, !tbaa !2
  %434 = call i32 %433(i32 1, i32 %3, i8* %5)
  br label %call_fail

for_begin11.preheader.1:                          ; preds = %for_end13.1, %for_end10
  %435 = phi i32 [ 0, %for_end10 ], [ %489, %for_end13.1 ]
  %436 = add nuw nsw i32 %435, %74
  %437 = shl i32 %436, 6
  %438 = shl nsw i32 %436, 9
  br label %vector.body58

vector.body58:                                    ; preds = %vector.body58, %for_begin11.preheader.1
  %index62 = phi i32 [ 0, %for_begin11.preheader.1 ], [ %index.next63, %vector.body58 ]
  %439 = add nuw nsw i32 %index62, %437
  %440 = add nuw nsw i32 %index62, %438
  %441 = add nuw nsw i32 %440, 448
  %442 = getelementptr inbounds i8, i8* %0, i32 %441
  %443 = bitcast i8* %442 to <4 x i8>*
  %wide.load69 = load <4 x i8>, <4 x i8>* %443, align 1, !tbaa !142
  %444 = and <4 x i8> %wide.load69, <i8 1, i8 1, i8 1, i8 1>
  %445 = add nuw nsw i32 %440, 384
  %446 = getelementptr inbounds i8, i8* %0, i32 %445
  %447 = bitcast i8* %446 to <4 x i8>*
  %wide.load70 = load <4 x i8>, <4 x i8>* %447, align 1, !tbaa !142
  %448 = and <4 x i8> %wide.load70, <i8 1, i8 1, i8 1, i8 1>
  %449 = add nuw nsw i32 %440, 320
  %450 = getelementptr inbounds i8, i8* %0, i32 %449
  %451 = bitcast i8* %450 to <4 x i8>*
  %wide.load71 = load <4 x i8>, <4 x i8>* %451, align 1, !tbaa !142
  %452 = and <4 x i8> %wide.load71, <i8 1, i8 1, i8 1, i8 1>
  %453 = add nuw nsw i32 %440, 256
  %454 = getelementptr inbounds i8, i8* %0, i32 %453
  %455 = bitcast i8* %454 to <4 x i8>*
  %wide.load72 = load <4 x i8>, <4 x i8>* %455, align 1, !tbaa !142
  %456 = and <4 x i8> %wide.load72, <i8 1, i8 1, i8 1, i8 1>
  %457 = add nuw nsw i32 %440, 192
  %458 = getelementptr inbounds i8, i8* %0, i32 %457
  %459 = bitcast i8* %458 to <4 x i8>*
  %wide.load73 = load <4 x i8>, <4 x i8>* %459, align 1, !tbaa !142
  %460 = and <4 x i8> %wide.load73, <i8 1, i8 1, i8 1, i8 1>
  %461 = add nuw nsw i32 %440, 128
  %462 = getelementptr inbounds i8, i8* %0, i32 %461
  %463 = bitcast i8* %462 to <4 x i8>*
  %wide.load74 = load <4 x i8>, <4 x i8>* %463, align 1, !tbaa !142
  %464 = and <4 x i8> %wide.load74, <i8 1, i8 1, i8 1, i8 1>
  %465 = add nuw nsw i32 %440, 64
  %466 = getelementptr inbounds i8, i8* %0, i32 %465
  %467 = bitcast i8* %466 to <4 x i8>*
  %wide.load75 = load <4 x i8>, <4 x i8>* %467, align 1, !tbaa !142
  %468 = and <4 x i8> %wide.load75, <i8 1, i8 1, i8 1, i8 1>
  %469 = getelementptr inbounds i8, i8* %0, i32 %440
  %470 = bitcast i8* %469 to <4 x i8>*
  %wide.load76 = load <4 x i8>, <4 x i8>* %470, align 1, !tbaa !142
  %471 = shl <4 x i8> %wide.load76, <i8 1, i8 1, i8 1, i8 1>
  %472 = and <4 x i8> %471, <i8 2, i8 2, i8 2, i8 2>
  %473 = or <4 x i8> %472, %468
  %474 = shl nuw nsw <4 x i8> %473, <i8 1, i8 1, i8 1, i8 1>
  %475 = or <4 x i8> %474, %464
  %476 = shl nuw nsw <4 x i8> %475, <i8 1, i8 1, i8 1, i8 1>
  %477 = or <4 x i8> %476, %460
  %478 = shl nuw nsw <4 x i8> %477, <i8 1, i8 1, i8 1, i8 1>
  %479 = or <4 x i8> %478, %456
  %480 = shl <4 x i8> %479, <i8 1, i8 1, i8 1, i8 1>
  %481 = or <4 x i8> %480, %452
  %482 = shl <4 x i8> %481, <i8 1, i8 1, i8 1, i8 1>
  %483 = or <4 x i8> %482, %448
  %484 = shl <4 x i8> %483, <i8 1, i8 1, i8 1, i8 1>
  %485 = or <4 x i8> %484, %444
  %486 = getelementptr inbounds i8, i8* %5, i32 %439
  %487 = bitcast i8* %486 to <4 x i8>*
  store <4 x i8> %485, <4 x i8>* %487, align 1, !tbaa !145
  %index.next63 = add i32 %index62, 4
  %488 = icmp eq i32 %index.next63, 64
  br i1 %488, label %for_end13.1, label %vector.body58, !llvm.loop !154

for_end13.1:                                      ; preds = %vector.body58
  %489 = add nuw nsw i32 %435, 1
  %exitcond48.1 = icmp eq i32 %489, 8
  br i1 %exitcond48.1, label %for_end10.1, label %for_begin11.preheader.1, !prof !24

for_end10.1:                                      ; preds = %for_end13.1
  %490 = mul i32 %10, 24
  %491 = add i32 %490, 16
  br label %for_begin11.preheader.2

for_begin11.preheader.2:                          ; preds = %for_end13.2, %for_end10.1
  %492 = phi i32 [ 0, %for_end10.1 ], [ %546, %for_end13.2 ]
  %493 = add nuw nsw i32 %492, %491
  %494 = shl i32 %493, 6
  %495 = shl nsw i32 %493, 9
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %for_begin11.preheader.2
  %index = phi i32 [ 0, %for_begin11.preheader.2 ], [ %index.next, %vector.body ]
  %496 = add nuw nsw i32 %index, %494
  %497 = add nuw nsw i32 %index, %495
  %498 = add nuw nsw i32 %497, 448
  %499 = getelementptr inbounds i8, i8* %0, i32 %498
  %500 = bitcast i8* %499 to <4 x i8>*
  %wide.load = load <4 x i8>, <4 x i8>* %500, align 1, !tbaa !142
  %501 = and <4 x i8> %wide.load, <i8 1, i8 1, i8 1, i8 1>
  %502 = add nuw nsw i32 %497, 384
  %503 = getelementptr inbounds i8, i8* %0, i32 %502
  %504 = bitcast i8* %503 to <4 x i8>*
  %wide.load51 = load <4 x i8>, <4 x i8>* %504, align 1, !tbaa !142
  %505 = and <4 x i8> %wide.load51, <i8 1, i8 1, i8 1, i8 1>
  %506 = add nuw nsw i32 %497, 320
  %507 = getelementptr inbounds i8, i8* %0, i32 %506
  %508 = bitcast i8* %507 to <4 x i8>*
  %wide.load52 = load <4 x i8>, <4 x i8>* %508, align 1, !tbaa !142
  %509 = and <4 x i8> %wide.load52, <i8 1, i8 1, i8 1, i8 1>
  %510 = add nuw nsw i32 %497, 256
  %511 = getelementptr inbounds i8, i8* %0, i32 %510
  %512 = bitcast i8* %511 to <4 x i8>*
  %wide.load53 = load <4 x i8>, <4 x i8>* %512, align 1, !tbaa !142
  %513 = and <4 x i8> %wide.load53, <i8 1, i8 1, i8 1, i8 1>
  %514 = add nuw nsw i32 %497, 192
  %515 = getelementptr inbounds i8, i8* %0, i32 %514
  %516 = bitcast i8* %515 to <4 x i8>*
  %wide.load54 = load <4 x i8>, <4 x i8>* %516, align 1, !tbaa !142
  %517 = and <4 x i8> %wide.load54, <i8 1, i8 1, i8 1, i8 1>
  %518 = add nuw nsw i32 %497, 128
  %519 = getelementptr inbounds i8, i8* %0, i32 %518
  %520 = bitcast i8* %519 to <4 x i8>*
  %wide.load55 = load <4 x i8>, <4 x i8>* %520, align 1, !tbaa !142
  %521 = and <4 x i8> %wide.load55, <i8 1, i8 1, i8 1, i8 1>
  %522 = add nuw nsw i32 %497, 64
  %523 = getelementptr inbounds i8, i8* %0, i32 %522
  %524 = bitcast i8* %523 to <4 x i8>*
  %wide.load56 = load <4 x i8>, <4 x i8>* %524, align 1, !tbaa !142
  %525 = and <4 x i8> %wide.load56, <i8 1, i8 1, i8 1, i8 1>
  %526 = getelementptr inbounds i8, i8* %0, i32 %497
  %527 = bitcast i8* %526 to <4 x i8>*
  %wide.load57 = load <4 x i8>, <4 x i8>* %527, align 1, !tbaa !142
  %528 = shl <4 x i8> %wide.load57, <i8 1, i8 1, i8 1, i8 1>
  %529 = and <4 x i8> %528, <i8 2, i8 2, i8 2, i8 2>
  %530 = or <4 x i8> %529, %525
  %531 = shl nuw nsw <4 x i8> %530, <i8 1, i8 1, i8 1, i8 1>
  %532 = or <4 x i8> %531, %521
  %533 = shl nuw nsw <4 x i8> %532, <i8 1, i8 1, i8 1, i8 1>
  %534 = or <4 x i8> %533, %517
  %535 = shl nuw nsw <4 x i8> %534, <i8 1, i8 1, i8 1, i8 1>
  %536 = or <4 x i8> %535, %513
  %537 = shl <4 x i8> %536, <i8 1, i8 1, i8 1, i8 1>
  %538 = or <4 x i8> %537, %509
  %539 = shl <4 x i8> %538, <i8 1, i8 1, i8 1, i8 1>
  %540 = or <4 x i8> %539, %505
  %541 = shl <4 x i8> %540, <i8 1, i8 1, i8 1, i8 1>
  %542 = or <4 x i8> %541, %501
  %543 = getelementptr inbounds i8, i8* %5, i32 %496
  %544 = bitcast i8* %543 to <4 x i8>*
  store <4 x i8> %542, <4 x i8>* %544, align 1, !tbaa !145
  %index.next = add i32 %index, 4
  %545 = icmp eq i32 %index.next, 64
  br i1 %545, label %for_end13.2, label %vector.body, !llvm.loop !155

for_end13.2:                                      ; preds = %vector.body
  %546 = add nuw nsw i32 %492, 1
  %exitcond48.2 = icmp eq i32 %546, 8
  br i1 %exitcond48.2, label %for_end10.2, label %for_begin11.preheader.2, !prof !24

for_end10.2:                                      ; preds = %for_end13.2
  %547 = add nuw nsw i32 %10, 1
  %exitcond50 = icmp eq i32 %547, 3
  br i1 %exitcond50, label %for_end, label %for_begin5.preheader, !prof !24
}

; Function Attrs: norecurse nounwind
define private i32 @__tvm_parallel_lambda(i32, %0* nocapture readonly, i8* nocapture readonly) #1 {
entry:
  %3 = bitcast i8* %2 to i8**
  %4 = load i8*, i8** %3, align 4
  %5 = getelementptr inbounds i8, i8* %2, i32 4
  %6 = bitcast i8* %5 to i8**
  %7 = load i8*, i8** %6, align 4
  %8 = getelementptr inbounds %0, %0* %1, i32 0, i32 1
  %9 = load i32, i32* %8, align 4
  %10 = add nsw i32 %9, 7
  %11 = sdiv i32 %10, %9
  %12 = add nsw i32 %0, 1
  %13 = mul nsw i32 %11, %12
  %14 = icmp slt i32 %13, 8
  %15 = select i1 %14, i32 %13, i32 8
  %16 = mul nsw i32 %11, %0
  %17 = icmp slt i32 %16, 8
  %18 = select i1 %17, i32 %16, i32 8
  %19 = icmp slt i32 %18, %15
  br i1 %19, label %for_begin1.preheader, label %for_end, !prof !1

for_begin1.preheader:                             ; preds = %entry, %for_end3
  %20 = phi i32 [ %27, %for_end3 ], [ %18, %entry ]
  %21 = mul nsw i32 %20, 3
  br label %for_begin4.preheader

for_end:                                          ; preds = %for_end3, %entry
  ret i32 0

for_begin4.preheader:                             ; preds = %for_end6, %for_begin1.preheader
  %22 = phi i32 [ 0, %for_begin1.preheader ], [ %361, %for_end6 ]
  %23 = add nsw i32 %22, %21
  %24 = mul nsw i32 %23, 3
  %25 = mul nuw nsw i32 %22, 192
  %26 = add nsw i32 %25, %20
  br label %for_begin7.preheader

for_end3:                                         ; preds = %for_end6
  %27 = add nsw i32 %20, 1
  %28 = icmp slt i32 %27, %15
  br i1 %28, label %for_begin1.preheader, label %for_end, !prof !1

for_begin7.preheader:                             ; preds = %for_begin7.preheader, %for_begin4.preheader
  %29 = phi i32 [ 0, %for_begin4.preheader ], [ %360, %for_begin7.preheader ]
  %30 = add nsw i32 %29, %24
  %31 = shl i32 %30, 6
  %32 = shl i32 %29, 6
  %33 = add nsw i32 %26, %32
  %34 = shl i32 %33, 3
  %35 = shl i32 %30, 6
  %36 = getelementptr inbounds i8, i8* %7, i32 %34
  %37 = load i8, i8* %36, align 1, !tbaa !145
  %38 = getelementptr inbounds i8, i8* %4, i32 %35
  store i8 %37, i8* %38, align 1, !tbaa !156
  %39 = or i32 %35, 1
  %40 = add nsw i32 %34, 64
  %41 = getelementptr inbounds i8, i8* %7, i32 %40
  %42 = load i8, i8* %41, align 1, !tbaa !145
  %43 = getelementptr inbounds i8, i8* %4, i32 %39
  store i8 %42, i8* %43, align 1, !tbaa !156
  %44 = or i32 %35, 2
  %45 = add nsw i32 %34, 128
  %46 = getelementptr inbounds i8, i8* %7, i32 %45
  %47 = load i8, i8* %46, align 1, !tbaa !145
  %48 = getelementptr inbounds i8, i8* %4, i32 %44
  store i8 %47, i8* %48, align 1, !tbaa !156
  %49 = or i32 %35, 3
  %50 = add nsw i32 %34, 192
  %51 = getelementptr inbounds i8, i8* %7, i32 %50
  %52 = load i8, i8* %51, align 1, !tbaa !145
  %53 = getelementptr inbounds i8, i8* %4, i32 %49
  store i8 %52, i8* %53, align 1, !tbaa !156
  %54 = or i32 %35, 4
  %55 = add nsw i32 %34, 256
  %56 = getelementptr inbounds i8, i8* %7, i32 %55
  %57 = load i8, i8* %56, align 1, !tbaa !145
  %58 = getelementptr inbounds i8, i8* %4, i32 %54
  store i8 %57, i8* %58, align 1, !tbaa !156
  %59 = or i32 %35, 5
  %60 = add nsw i32 %34, 320
  %61 = getelementptr inbounds i8, i8* %7, i32 %60
  %62 = load i8, i8* %61, align 1, !tbaa !145
  %63 = getelementptr inbounds i8, i8* %4, i32 %59
  store i8 %62, i8* %63, align 1, !tbaa !156
  %64 = or i32 %35, 6
  %65 = add nsw i32 %34, 384
  %66 = getelementptr inbounds i8, i8* %7, i32 %65
  %67 = load i8, i8* %66, align 1, !tbaa !145
  %68 = getelementptr inbounds i8, i8* %4, i32 %64
  store i8 %67, i8* %68, align 1, !tbaa !156
  %69 = or i32 %35, 7
  %70 = add nsw i32 %34, 448
  %71 = getelementptr inbounds i8, i8* %7, i32 %70
  %72 = load i8, i8* %71, align 1, !tbaa !145
  %73 = getelementptr inbounds i8, i8* %4, i32 %69
  store i8 %72, i8* %73, align 1, !tbaa !156
  %74 = shl i32 %30, 6
  %75 = or i32 %74, 8
  %76 = or i32 %34, 1
  %77 = getelementptr inbounds i8, i8* %7, i32 %76
  %78 = load i8, i8* %77, align 1, !tbaa !145
  %79 = getelementptr inbounds i8, i8* %4, i32 %75
  store i8 %78, i8* %79, align 1, !tbaa !156
  %80 = or i32 %74, 9
  %81 = add nsw i32 %76, 64
  %82 = getelementptr inbounds i8, i8* %7, i32 %81
  %83 = load i8, i8* %82, align 1, !tbaa !145
  %84 = getelementptr inbounds i8, i8* %4, i32 %80
  store i8 %83, i8* %84, align 1, !tbaa !156
  %85 = or i32 %74, 10
  %86 = add nsw i32 %76, 128
  %87 = getelementptr inbounds i8, i8* %7, i32 %86
  %88 = load i8, i8* %87, align 1, !tbaa !145
  %89 = getelementptr inbounds i8, i8* %4, i32 %85
  store i8 %88, i8* %89, align 1, !tbaa !156
  %90 = or i32 %74, 11
  %91 = add nsw i32 %76, 192
  %92 = getelementptr inbounds i8, i8* %7, i32 %91
  %93 = load i8, i8* %92, align 1, !tbaa !145
  %94 = getelementptr inbounds i8, i8* %4, i32 %90
  store i8 %93, i8* %94, align 1, !tbaa !156
  %95 = or i32 %74, 12
  %96 = add nsw i32 %76, 256
  %97 = getelementptr inbounds i8, i8* %7, i32 %96
  %98 = load i8, i8* %97, align 1, !tbaa !145
  %99 = getelementptr inbounds i8, i8* %4, i32 %95
  store i8 %98, i8* %99, align 1, !tbaa !156
  %100 = or i32 %74, 13
  %101 = add nsw i32 %76, 320
  %102 = getelementptr inbounds i8, i8* %7, i32 %101
  %103 = load i8, i8* %102, align 1, !tbaa !145
  %104 = getelementptr inbounds i8, i8* %4, i32 %100
  store i8 %103, i8* %104, align 1, !tbaa !156
  %105 = or i32 %74, 14
  %106 = add nsw i32 %76, 384
  %107 = getelementptr inbounds i8, i8* %7, i32 %106
  %108 = load i8, i8* %107, align 1, !tbaa !145
  %109 = getelementptr inbounds i8, i8* %4, i32 %105
  store i8 %108, i8* %109, align 1, !tbaa !156
  %110 = or i32 %74, 15
  %111 = add nsw i32 %76, 448
  %112 = getelementptr inbounds i8, i8* %7, i32 %111
  %113 = load i8, i8* %112, align 1, !tbaa !145
  %114 = getelementptr inbounds i8, i8* %4, i32 %110
  store i8 %113, i8* %114, align 1, !tbaa !156
  %115 = shl i32 %30, 6
  %116 = or i32 %115, 16
  %117 = or i32 %34, 2
  %118 = getelementptr inbounds i8, i8* %7, i32 %117
  %119 = load i8, i8* %118, align 1, !tbaa !145
  %120 = getelementptr inbounds i8, i8* %4, i32 %116
  store i8 %119, i8* %120, align 1, !tbaa !156
  %121 = or i32 %115, 17
  %122 = add nsw i32 %117, 64
  %123 = getelementptr inbounds i8, i8* %7, i32 %122
  %124 = load i8, i8* %123, align 1, !tbaa !145
  %125 = getelementptr inbounds i8, i8* %4, i32 %121
  store i8 %124, i8* %125, align 1, !tbaa !156
  %126 = or i32 %115, 18
  %127 = add nsw i32 %117, 128
  %128 = getelementptr inbounds i8, i8* %7, i32 %127
  %129 = load i8, i8* %128, align 1, !tbaa !145
  %130 = getelementptr inbounds i8, i8* %4, i32 %126
  store i8 %129, i8* %130, align 1, !tbaa !156
  %131 = or i32 %115, 19
  %132 = add nsw i32 %117, 192
  %133 = getelementptr inbounds i8, i8* %7, i32 %132
  %134 = load i8, i8* %133, align 1, !tbaa !145
  %135 = getelementptr inbounds i8, i8* %4, i32 %131
  store i8 %134, i8* %135, align 1, !tbaa !156
  %136 = or i32 %115, 20
  %137 = add nsw i32 %117, 256
  %138 = getelementptr inbounds i8, i8* %7, i32 %137
  %139 = load i8, i8* %138, align 1, !tbaa !145
  %140 = getelementptr inbounds i8, i8* %4, i32 %136
  store i8 %139, i8* %140, align 1, !tbaa !156
  %141 = or i32 %115, 21
  %142 = add nsw i32 %117, 320
  %143 = getelementptr inbounds i8, i8* %7, i32 %142
  %144 = load i8, i8* %143, align 1, !tbaa !145
  %145 = getelementptr inbounds i8, i8* %4, i32 %141
  store i8 %144, i8* %145, align 1, !tbaa !156
  %146 = or i32 %115, 22
  %147 = add nsw i32 %117, 384
  %148 = getelementptr inbounds i8, i8* %7, i32 %147
  %149 = load i8, i8* %148, align 1, !tbaa !145
  %150 = getelementptr inbounds i8, i8* %4, i32 %146
  store i8 %149, i8* %150, align 1, !tbaa !156
  %151 = or i32 %115, 23
  %152 = add nsw i32 %117, 448
  %153 = getelementptr inbounds i8, i8* %7, i32 %152
  %154 = load i8, i8* %153, align 1, !tbaa !145
  %155 = getelementptr inbounds i8, i8* %4, i32 %151
  store i8 %154, i8* %155, align 1, !tbaa !156
  %156 = shl i32 %30, 6
  %157 = or i32 %156, 24
  %158 = or i32 %34, 3
  %159 = getelementptr inbounds i8, i8* %7, i32 %158
  %160 = load i8, i8* %159, align 1, !tbaa !145
  %161 = getelementptr inbounds i8, i8* %4, i32 %157
  store i8 %160, i8* %161, align 1, !tbaa !156
  %162 = or i32 %156, 25
  %163 = add nsw i32 %158, 64
  %164 = getelementptr inbounds i8, i8* %7, i32 %163
  %165 = load i8, i8* %164, align 1, !tbaa !145
  %166 = getelementptr inbounds i8, i8* %4, i32 %162
  store i8 %165, i8* %166, align 1, !tbaa !156
  %167 = or i32 %156, 26
  %168 = add nsw i32 %158, 128
  %169 = getelementptr inbounds i8, i8* %7, i32 %168
  %170 = load i8, i8* %169, align 1, !tbaa !145
  %171 = getelementptr inbounds i8, i8* %4, i32 %167
  store i8 %170, i8* %171, align 1, !tbaa !156
  %172 = or i32 %156, 27
  %173 = add nsw i32 %158, 192
  %174 = getelementptr inbounds i8, i8* %7, i32 %173
  %175 = load i8, i8* %174, align 1, !tbaa !145
  %176 = getelementptr inbounds i8, i8* %4, i32 %172
  store i8 %175, i8* %176, align 1, !tbaa !156
  %177 = or i32 %156, 28
  %178 = add nsw i32 %158, 256
  %179 = getelementptr inbounds i8, i8* %7, i32 %178
  %180 = load i8, i8* %179, align 1, !tbaa !145
  %181 = getelementptr inbounds i8, i8* %4, i32 %177
  store i8 %180, i8* %181, align 1, !tbaa !156
  %182 = or i32 %156, 29
  %183 = add nsw i32 %158, 320
  %184 = getelementptr inbounds i8, i8* %7, i32 %183
  %185 = load i8, i8* %184, align 1, !tbaa !145
  %186 = getelementptr inbounds i8, i8* %4, i32 %182
  store i8 %185, i8* %186, align 1, !tbaa !156
  %187 = or i32 %156, 30
  %188 = add nsw i32 %158, 384
  %189 = getelementptr inbounds i8, i8* %7, i32 %188
  %190 = load i8, i8* %189, align 1, !tbaa !145
  %191 = getelementptr inbounds i8, i8* %4, i32 %187
  store i8 %190, i8* %191, align 1, !tbaa !156
  %192 = or i32 %156, 31
  %193 = add nsw i32 %158, 448
  %194 = getelementptr inbounds i8, i8* %7, i32 %193
  %195 = load i8, i8* %194, align 1, !tbaa !145
  %196 = getelementptr inbounds i8, i8* %4, i32 %192
  store i8 %195, i8* %196, align 1, !tbaa !156
  %197 = shl i32 %30, 6
  %198 = or i32 %197, 32
  %199 = or i32 %34, 4
  %200 = getelementptr inbounds i8, i8* %7, i32 %199
  %201 = load i8, i8* %200, align 1, !tbaa !145
  %202 = getelementptr inbounds i8, i8* %4, i32 %198
  store i8 %201, i8* %202, align 1, !tbaa !156
  %203 = or i32 %197, 33
  %204 = add nsw i32 %199, 64
  %205 = getelementptr inbounds i8, i8* %7, i32 %204
  %206 = load i8, i8* %205, align 1, !tbaa !145
  %207 = getelementptr inbounds i8, i8* %4, i32 %203
  store i8 %206, i8* %207, align 1, !tbaa !156
  %208 = or i32 %197, 34
  %209 = add nsw i32 %199, 128
  %210 = getelementptr inbounds i8, i8* %7, i32 %209
  %211 = load i8, i8* %210, align 1, !tbaa !145
  %212 = getelementptr inbounds i8, i8* %4, i32 %208
  store i8 %211, i8* %212, align 1, !tbaa !156
  %213 = or i32 %197, 35
  %214 = add nsw i32 %199, 192
  %215 = getelementptr inbounds i8, i8* %7, i32 %214
  %216 = load i8, i8* %215, align 1, !tbaa !145
  %217 = getelementptr inbounds i8, i8* %4, i32 %213
  store i8 %216, i8* %217, align 1, !tbaa !156
  %218 = or i32 %197, 36
  %219 = add nsw i32 %199, 256
  %220 = getelementptr inbounds i8, i8* %7, i32 %219
  %221 = load i8, i8* %220, align 1, !tbaa !145
  %222 = getelementptr inbounds i8, i8* %4, i32 %218
  store i8 %221, i8* %222, align 1, !tbaa !156
  %223 = or i32 %197, 37
  %224 = add nsw i32 %199, 320
  %225 = getelementptr inbounds i8, i8* %7, i32 %224
  %226 = load i8, i8* %225, align 1, !tbaa !145
  %227 = getelementptr inbounds i8, i8* %4, i32 %223
  store i8 %226, i8* %227, align 1, !tbaa !156
  %228 = or i32 %197, 38
  %229 = add nsw i32 %199, 384
  %230 = getelementptr inbounds i8, i8* %7, i32 %229
  %231 = load i8, i8* %230, align 1, !tbaa !145
  %232 = getelementptr inbounds i8, i8* %4, i32 %228
  store i8 %231, i8* %232, align 1, !tbaa !156
  %233 = or i32 %197, 39
  %234 = add nsw i32 %199, 448
  %235 = getelementptr inbounds i8, i8* %7, i32 %234
  %236 = load i8, i8* %235, align 1, !tbaa !145
  %237 = getelementptr inbounds i8, i8* %4, i32 %233
  store i8 %236, i8* %237, align 1, !tbaa !156
  %238 = shl i32 %30, 6
  %239 = or i32 %238, 40
  %240 = or i32 %34, 5
  %241 = getelementptr inbounds i8, i8* %7, i32 %240
  %242 = load i8, i8* %241, align 1, !tbaa !145
  %243 = getelementptr inbounds i8, i8* %4, i32 %239
  store i8 %242, i8* %243, align 1, !tbaa !156
  %244 = or i32 %238, 41
  %245 = add nsw i32 %240, 64
  %246 = getelementptr inbounds i8, i8* %7, i32 %245
  %247 = load i8, i8* %246, align 1, !tbaa !145
  %248 = getelementptr inbounds i8, i8* %4, i32 %244
  store i8 %247, i8* %248, align 1, !tbaa !156
  %249 = or i32 %238, 42
  %250 = add nsw i32 %240, 128
  %251 = getelementptr inbounds i8, i8* %7, i32 %250
  %252 = load i8, i8* %251, align 1, !tbaa !145
  %253 = getelementptr inbounds i8, i8* %4, i32 %249
  store i8 %252, i8* %253, align 1, !tbaa !156
  %254 = or i32 %238, 43
  %255 = add nsw i32 %240, 192
  %256 = getelementptr inbounds i8, i8* %7, i32 %255
  %257 = load i8, i8* %256, align 1, !tbaa !145
  %258 = getelementptr inbounds i8, i8* %4, i32 %254
  store i8 %257, i8* %258, align 1, !tbaa !156
  %259 = or i32 %238, 44
  %260 = add nsw i32 %240, 256
  %261 = getelementptr inbounds i8, i8* %7, i32 %260
  %262 = load i8, i8* %261, align 1, !tbaa !145
  %263 = getelementptr inbounds i8, i8* %4, i32 %259
  store i8 %262, i8* %263, align 1, !tbaa !156
  %264 = or i32 %238, 45
  %265 = add nsw i32 %240, 320
  %266 = getelementptr inbounds i8, i8* %7, i32 %265
  %267 = load i8, i8* %266, align 1, !tbaa !145
  %268 = getelementptr inbounds i8, i8* %4, i32 %264
  store i8 %267, i8* %268, align 1, !tbaa !156
  %269 = or i32 %238, 46
  %270 = add nsw i32 %240, 384
  %271 = getelementptr inbounds i8, i8* %7, i32 %270
  %272 = load i8, i8* %271, align 1, !tbaa !145
  %273 = getelementptr inbounds i8, i8* %4, i32 %269
  store i8 %272, i8* %273, align 1, !tbaa !156
  %274 = or i32 %238, 47
  %275 = add nsw i32 %240, 448
  %276 = getelementptr inbounds i8, i8* %7, i32 %275
  %277 = load i8, i8* %276, align 1, !tbaa !145
  %278 = getelementptr inbounds i8, i8* %4, i32 %274
  store i8 %277, i8* %278, align 1, !tbaa !156
  %279 = shl i32 %30, 6
  %280 = or i32 %279, 48
  %281 = or i32 %34, 6
  %282 = getelementptr inbounds i8, i8* %7, i32 %281
  %283 = load i8, i8* %282, align 1, !tbaa !145
  %284 = getelementptr inbounds i8, i8* %4, i32 %280
  store i8 %283, i8* %284, align 1, !tbaa !156
  %285 = or i32 %279, 49
  %286 = add nsw i32 %281, 64
  %287 = getelementptr inbounds i8, i8* %7, i32 %286
  %288 = load i8, i8* %287, align 1, !tbaa !145
  %289 = getelementptr inbounds i8, i8* %4, i32 %285
  store i8 %288, i8* %289, align 1, !tbaa !156
  %290 = or i32 %279, 50
  %291 = add nsw i32 %281, 128
  %292 = getelementptr inbounds i8, i8* %7, i32 %291
  %293 = load i8, i8* %292, align 1, !tbaa !145
  %294 = getelementptr inbounds i8, i8* %4, i32 %290
  store i8 %293, i8* %294, align 1, !tbaa !156
  %295 = or i32 %279, 51
  %296 = add nsw i32 %281, 192
  %297 = getelementptr inbounds i8, i8* %7, i32 %296
  %298 = load i8, i8* %297, align 1, !tbaa !145
  %299 = getelementptr inbounds i8, i8* %4, i32 %295
  store i8 %298, i8* %299, align 1, !tbaa !156
  %300 = or i32 %279, 52
  %301 = add nsw i32 %281, 256
  %302 = getelementptr inbounds i8, i8* %7, i32 %301
  %303 = load i8, i8* %302, align 1, !tbaa !145
  %304 = getelementptr inbounds i8, i8* %4, i32 %300
  store i8 %303, i8* %304, align 1, !tbaa !156
  %305 = or i32 %279, 53
  %306 = add nsw i32 %281, 320
  %307 = getelementptr inbounds i8, i8* %7, i32 %306
  %308 = load i8, i8* %307, align 1, !tbaa !145
  %309 = getelementptr inbounds i8, i8* %4, i32 %305
  store i8 %308, i8* %309, align 1, !tbaa !156
  %310 = or i32 %279, 54
  %311 = add nsw i32 %281, 384
  %312 = getelementptr inbounds i8, i8* %7, i32 %311
  %313 = load i8, i8* %312, align 1, !tbaa !145
  %314 = getelementptr inbounds i8, i8* %4, i32 %310
  store i8 %313, i8* %314, align 1, !tbaa !156
  %315 = or i32 %279, 55
  %316 = add nsw i32 %281, 448
  %317 = getelementptr inbounds i8, i8* %7, i32 %316
  %318 = load i8, i8* %317, align 1, !tbaa !145
  %319 = getelementptr inbounds i8, i8* %4, i32 %315
  store i8 %318, i8* %319, align 1, !tbaa !156
  %320 = or i32 %31, 56
  %321 = or i32 %34, 7
  %322 = getelementptr inbounds i8, i8* %7, i32 %321
  %323 = load i8, i8* %322, align 1, !tbaa !145
  %324 = getelementptr inbounds i8, i8* %4, i32 %320
  store i8 %323, i8* %324, align 1, !tbaa !156
  %325 = or i32 %31, 57
  %326 = add nsw i32 %321, 64
  %327 = getelementptr inbounds i8, i8* %7, i32 %326
  %328 = load i8, i8* %327, align 1, !tbaa !145
  %329 = getelementptr inbounds i8, i8* %4, i32 %325
  store i8 %328, i8* %329, align 1, !tbaa !156
  %330 = or i32 %31, 58
  %331 = add nsw i32 %321, 128
  %332 = getelementptr inbounds i8, i8* %7, i32 %331
  %333 = load i8, i8* %332, align 1, !tbaa !145
  %334 = getelementptr inbounds i8, i8* %4, i32 %330
  store i8 %333, i8* %334, align 1, !tbaa !156
  %335 = or i32 %31, 59
  %336 = add nsw i32 %321, 192
  %337 = getelementptr inbounds i8, i8* %7, i32 %336
  %338 = load i8, i8* %337, align 1, !tbaa !145
  %339 = getelementptr inbounds i8, i8* %4, i32 %335
  store i8 %338, i8* %339, align 1, !tbaa !156
  %340 = or i32 %31, 60
  %341 = add nsw i32 %321, 256
  %342 = getelementptr inbounds i8, i8* %7, i32 %341
  %343 = load i8, i8* %342, align 1, !tbaa !145
  %344 = getelementptr inbounds i8, i8* %4, i32 %340
  store i8 %343, i8* %344, align 1, !tbaa !156
  %345 = or i32 %31, 61
  %346 = add nsw i32 %321, 320
  %347 = getelementptr inbounds i8, i8* %7, i32 %346
  %348 = load i8, i8* %347, align 1, !tbaa !145
  %349 = getelementptr inbounds i8, i8* %4, i32 %345
  store i8 %348, i8* %349, align 1, !tbaa !156
  %350 = or i32 %31, 62
  %351 = add nsw i32 %321, 384
  %352 = getelementptr inbounds i8, i8* %7, i32 %351
  %353 = load i8, i8* %352, align 1, !tbaa !145
  %354 = getelementptr inbounds i8, i8* %4, i32 %350
  store i8 %353, i8* %354, align 1, !tbaa !156
  %355 = or i32 %31, 63
  %356 = add nsw i32 %321, 448
  %357 = getelementptr inbounds i8, i8* %7, i32 %356
  %358 = load i8, i8* %357, align 1, !tbaa !145
  %359 = getelementptr inbounds i8, i8* %4, i32 %355
  store i8 %358, i8* %359, align 1, !tbaa !156
  %360 = add nuw nsw i32 %29, 1
  %exitcond = icmp eq i32 %360, 3
  br i1 %exitcond, label %for_end6, label %for_begin7.preheader, !prof !24

for_end6:                                         ; preds = %for_begin7.preheader
  %361 = add nuw nsw i32 %22, 1
  %exitcond13 = icmp eq i32 %361, 3
  br i1 %exitcond13, label %for_end3, label %for_begin4.preheader, !prof !24
}

; Function Attrs: norecurse nounwind
define private i32 @__tvm_parallel_lambda.34(i32, %0* nocapture readonly, i8* nocapture readonly) #1 {
entry:
  %3 = bitcast i8* %2 to i8**
  %4 = load i8*, i8** %3, align 4
  %5 = getelementptr inbounds i8, i8* %2, i32 4
  %6 = bitcast i8* %5 to i8**
  %7 = load i8*, i8** %6, align 4
  %8 = getelementptr inbounds %0, %0* %1, i32 0, i32 1
  %9 = load i32, i32* %8, align 4
  %10 = add nsw i32 %9, 55
  %11 = sdiv i32 %10, %9
  %12 = add nsw i32 %0, 1
  %13 = mul nsw i32 %11, %12
  %14 = icmp slt i32 %13, 56
  %15 = select i1 %14, i32 %13, i32 56
  %16 = mul i32 %11, %0
  %17 = icmp slt i32 %16, 56
  %18 = select i1 %17, i32 %16, i32 56
  %19 = icmp slt i32 %18, %15
  br i1 %19, label %for_begin1.preheader.preheader, label %for_end, !prof !1

for_begin1.preheader.preheader:                   ; preds = %entry
  %20 = xor i32 %16, -1
  %21 = icmp sgt i32 %20, -57
  %smax = select i1 %21, i32 %20, i32 -57
  %22 = mul i32 %smax, -4032
  %23 = add i32 %22, -4032
  br label %for_begin1.preheader

for_begin1.preheader:                             ; preds = %for_begin1.preheader.preheader, %for_end3
  %indvar = phi i32 [ 0, %for_begin1.preheader.preheader ], [ %indvar.next, %for_end3 ]
  %24 = phi i32 [ %18, %for_begin1.preheader.preheader ], [ %35, %for_end3 ]
  %25 = mul i32 %indvar, 4032
  %26 = add i32 %23, %25
  %27 = mul nsw i32 %24, 56
  br label %for_begin4.preheader

for_end:                                          ; preds = %for_end3, %entry
  ret i32 0

for_begin4.preheader:                             ; preds = %for_end6, %for_begin1.preheader
  %28 = phi i32 [ 0, %for_begin1.preheader ], [ %88, %for_end6 ]
  %29 = mul nuw nsw i32 %28, 72
  %30 = add i32 %26, %29
  %31 = add nsw i32 %28, %27
  %32 = mul nsw i32 %31, 3
  %33 = icmp eq i32 %28, 0
  %34 = icmp ult i32 %28, 55
  br label %for_begin7.preheader

for_end3:                                         ; preds = %for_end6
  %35 = add nsw i32 %24, 1
  %36 = icmp slt i32 %35, %15
  %indvar.next = add nuw i32 %indvar, 1
  br i1 %36, label %for_begin1.preheader, label %for_end, !prof !1

for_begin7.preheader:                             ; preds = %for_end9, %for_begin4.preheader
  %37 = phi i32 [ 0, %for_begin4.preheader ], [ %89, %for_end9 ]
  %38 = mul nuw nsw i32 %37, 24
  %39 = add i32 %30, %38
  %scevgep20 = getelementptr i8, i8* %4, i32 %39
  %40 = add nsw i32 %37, %32
  %41 = sub nuw nsw i32 57, %37
  %42 = icmp slt i32 %24, %41
  %43 = sub nsw i32 1, %37
  %44 = icmp sle i32 %43, %24
  %45 = and i1 %44, %42
  %reass.add = add i32 %37, %24
  %reass.mul = mul i32 %reass.add, 58
  %46 = add i32 %reass.mul, %28
  br i1 %45, label %for_begin10.preheader.us.preheader, label %for_begin10.preheader.preheader

for_begin10.preheader.preheader:                  ; preds = %for_begin7.preheader
  call void @llvm.memset.p0i8.i32(i8* align 1 %scevgep20, i8 0, i32 24, i1 false)
  br label %for_end9

for_begin10.preheader.us.preheader:               ; preds = %for_begin7.preheader
  %47 = mul i32 %40, 24
  %48 = shl i32 %46, 3
  br i1 %33, label %for_body11.us13.preheader, label %for_body11.us.us.preheader

for_body11.us13.preheader:                        ; preds = %for_begin10.preheader.us.preheader
  %49 = bitcast i8* %scevgep20 to i64*
  store i64 0, i64* %49, align 1
  br label %for_end12.us.1

for_body11.us.us.preheader:                       ; preds = %for_begin10.preheader.us.preheader
  %50 = getelementptr inbounds i8, i8* %7, i32 %48
  %51 = load i8, i8* %50, align 1, !tbaa !145
  %52 = getelementptr inbounds i8, i8* %4, i32 %47
  store i8 %51, i8* %52, align 1, !tbaa !159
  %53 = or i32 %47, 1
  %54 = or i32 %48, 1
  %55 = getelementptr inbounds i8, i8* %7, i32 %54
  %56 = load i8, i8* %55, align 1, !tbaa !145
  %57 = getelementptr inbounds i8, i8* %4, i32 %53
  store i8 %56, i8* %57, align 1, !tbaa !159
  %58 = or i32 %47, 2
  %59 = or i32 %48, 2
  %60 = getelementptr inbounds i8, i8* %7, i32 %59
  %61 = load i8, i8* %60, align 1, !tbaa !145
  %62 = getelementptr inbounds i8, i8* %4, i32 %58
  store i8 %61, i8* %62, align 1, !tbaa !159
  %63 = or i32 %47, 3
  %64 = or i32 %48, 3
  %65 = getelementptr inbounds i8, i8* %7, i32 %64
  %66 = load i8, i8* %65, align 1, !tbaa !145
  %67 = getelementptr inbounds i8, i8* %4, i32 %63
  store i8 %66, i8* %67, align 1, !tbaa !159
  %68 = or i32 %47, 4
  %69 = or i32 %48, 4
  %70 = getelementptr inbounds i8, i8* %7, i32 %69
  %71 = load i8, i8* %70, align 1, !tbaa !145
  %72 = getelementptr inbounds i8, i8* %4, i32 %68
  store i8 %71, i8* %72, align 1, !tbaa !159
  %73 = or i32 %47, 5
  %74 = or i32 %48, 5
  %75 = getelementptr inbounds i8, i8* %7, i32 %74
  %76 = load i8, i8* %75, align 1, !tbaa !145
  %77 = getelementptr inbounds i8, i8* %4, i32 %73
  store i8 %76, i8* %77, align 1, !tbaa !159
  %78 = or i32 %47, 6
  %79 = or i32 %48, 6
  %80 = getelementptr inbounds i8, i8* %7, i32 %79
  %81 = load i8, i8* %80, align 1, !tbaa !145
  %82 = getelementptr inbounds i8, i8* %4, i32 %78
  store i8 %81, i8* %82, align 1, !tbaa !159
  %83 = or i32 %47, 7
  %84 = or i32 %48, 7
  %85 = getelementptr inbounds i8, i8* %7, i32 %84
  %86 = load i8, i8* %85, align 1, !tbaa !145
  %87 = getelementptr inbounds i8, i8* %4, i32 %83
  store i8 %86, i8* %87, align 1, !tbaa !159
  br label %for_end12.us.1

for_end6:                                         ; preds = %for_end9
  %88 = add nuw nsw i32 %28, 1
  %exitcond22 = icmp eq i32 %88, 56
  br i1 %exitcond22, label %for_end3, label %for_begin4.preheader, !prof !24

for_end9:                                         ; preds = %for_body11.us13.preheader.2, %for_body11.us.us.preheader.2, %for_begin10.preheader.preheader
  %89 = add nuw nsw i32 %37, 1
  %exitcond = icmp eq i32 %89, 3
  br i1 %exitcond, label %for_end6, label %for_begin7.preheader, !prof !24

for_end12.us.1:                                   ; preds = %for_body11.us13.preheader, %for_body11.us.us.preheader
  %90 = mul i32 %40, 24
  %91 = shl i32 %46, 3
  %92 = add i32 %91, 8
  %93 = add i32 %90, 8
  %94 = getelementptr inbounds i8, i8* %7, i32 %92
  %95 = load i8, i8* %94, align 1, !tbaa !145
  %96 = getelementptr inbounds i8, i8* %4, i32 %93
  store i8 %95, i8* %96, align 1, !tbaa !159
  %97 = add i32 %90, 9
  %98 = add i32 %91, 9
  %99 = getelementptr inbounds i8, i8* %7, i32 %98
  %100 = load i8, i8* %99, align 1, !tbaa !145
  %101 = getelementptr inbounds i8, i8* %4, i32 %97
  store i8 %100, i8* %101, align 1, !tbaa !159
  %102 = add i32 %90, 10
  %103 = add i32 %91, 10
  %104 = getelementptr inbounds i8, i8* %7, i32 %103
  %105 = load i8, i8* %104, align 1, !tbaa !145
  %106 = getelementptr inbounds i8, i8* %4, i32 %102
  store i8 %105, i8* %106, align 1, !tbaa !159
  %107 = add i32 %90, 11
  %108 = add i32 %91, 11
  %109 = getelementptr inbounds i8, i8* %7, i32 %108
  %110 = load i8, i8* %109, align 1, !tbaa !145
  %111 = getelementptr inbounds i8, i8* %4, i32 %107
  store i8 %110, i8* %111, align 1, !tbaa !159
  %112 = add i32 %90, 12
  %113 = add i32 %91, 12
  %114 = getelementptr inbounds i8, i8* %7, i32 %113
  %115 = load i8, i8* %114, align 1, !tbaa !145
  %116 = getelementptr inbounds i8, i8* %4, i32 %112
  store i8 %115, i8* %116, align 1, !tbaa !159
  %117 = add i32 %90, 13
  %118 = add i32 %91, 13
  %119 = getelementptr inbounds i8, i8* %7, i32 %118
  %120 = load i8, i8* %119, align 1, !tbaa !145
  %121 = getelementptr inbounds i8, i8* %4, i32 %117
  store i8 %120, i8* %121, align 1, !tbaa !159
  %122 = add i32 %90, 14
  %123 = add i32 %91, 14
  %124 = getelementptr inbounds i8, i8* %7, i32 %123
  %125 = load i8, i8* %124, align 1, !tbaa !145
  %126 = getelementptr inbounds i8, i8* %4, i32 %122
  store i8 %125, i8* %126, align 1, !tbaa !159
  %127 = add i32 %90, 15
  %128 = add i32 %91, 15
  %129 = getelementptr inbounds i8, i8* %7, i32 %128
  %130 = load i8, i8* %129, align 1, !tbaa !145
  %131 = getelementptr inbounds i8, i8* %4, i32 %127
  store i8 %130, i8* %131, align 1, !tbaa !159
  %132 = mul i32 %40, 24
  %133 = shl i32 %46, 3
  br i1 %34, label %for_body11.us.us.preheader.2, label %for_body11.us13.preheader.2

for_body11.us13.preheader.2:                      ; preds = %for_end12.us.1
  %134 = add i32 %39, 16
  %scevgep.2 = getelementptr i8, i8* %4, i32 %134
  %135 = bitcast i8* %scevgep.2 to i64*
  store i64 0, i64* %135, align 1
  br label %for_end9

for_body11.us.us.preheader.2:                     ; preds = %for_end12.us.1
  %136 = add i32 %133, 16
  %137 = add i32 %132, 16
  %138 = getelementptr inbounds i8, i8* %7, i32 %136
  %139 = load i8, i8* %138, align 1, !tbaa !145
  %140 = getelementptr inbounds i8, i8* %4, i32 %137
  store i8 %139, i8* %140, align 1, !tbaa !159
  %141 = add i32 %132, 17
  %142 = add i32 %133, 17
  %143 = getelementptr inbounds i8, i8* %7, i32 %142
  %144 = load i8, i8* %143, align 1, !tbaa !145
  %145 = getelementptr inbounds i8, i8* %4, i32 %141
  store i8 %144, i8* %145, align 1, !tbaa !159
  %146 = add i32 %132, 18
  %147 = add i32 %133, 18
  %148 = getelementptr inbounds i8, i8* %7, i32 %147
  %149 = load i8, i8* %148, align 1, !tbaa !145
  %150 = getelementptr inbounds i8, i8* %4, i32 %146
  store i8 %149, i8* %150, align 1, !tbaa !159
  %151 = add i32 %132, 19
  %152 = add i32 %133, 19
  %153 = getelementptr inbounds i8, i8* %7, i32 %152
  %154 = load i8, i8* %153, align 1, !tbaa !145
  %155 = getelementptr inbounds i8, i8* %4, i32 %151
  store i8 %154, i8* %155, align 1, !tbaa !159
  %156 = add i32 %132, 20
  %157 = add i32 %133, 20
  %158 = getelementptr inbounds i8, i8* %7, i32 %157
  %159 = load i8, i8* %158, align 1, !tbaa !145
  %160 = getelementptr inbounds i8, i8* %4, i32 %156
  store i8 %159, i8* %160, align 1, !tbaa !159
  %161 = add i32 %132, 21
  %162 = add i32 %133, 21
  %163 = getelementptr inbounds i8, i8* %7, i32 %162
  %164 = load i8, i8* %163, align 1, !tbaa !145
  %165 = getelementptr inbounds i8, i8* %4, i32 %161
  store i8 %164, i8* %165, align 1, !tbaa !159
  %166 = add i32 %132, 22
  %167 = add i32 %133, 22
  %168 = getelementptr inbounds i8, i8* %7, i32 %167
  %169 = load i8, i8* %168, align 1, !tbaa !145
  %170 = getelementptr inbounds i8, i8* %4, i32 %166
  store i8 %169, i8* %170, align 1, !tbaa !159
  %171 = add i32 %132, 23
  %172 = add i32 %133, 23
  %173 = getelementptr inbounds i8, i8* %7, i32 %172
  %174 = load i8, i8* %173, align 1, !tbaa !145
  %175 = getelementptr inbounds i8, i8* %4, i32 %171
  store i8 %174, i8* %175, align 1, !tbaa !159
  br label %for_end9
}

; Function Attrs: nounwind
define private i32 @__tvm_parallel_lambda.35(i32, %0* nocapture readonly, i8* nocapture readonly) #2 {
entry:
  %3 = bitcast i8* %2 to i8**
  %4 = load i8*, i8** %3, align 4
  %5 = getelementptr inbounds i8, i8* %2, i32 4
  %6 = bitcast i8* %5 to i8**
  %7 = load i8*, i8** %6, align 4
  %8 = getelementptr inbounds i8, i8* %2, i32 8
  %9 = bitcast i8* %8 to i16**
  %10 = load i16*, i16** %9, align 4
  %11 = getelementptr inbounds %0, %0* %1, i32 0, i32 1
  %12 = load i32, i32* %11, align 4
  %13 = add nsw i32 %12, 55
  %14 = sdiv i32 %13, %12
  %15 = add nsw i32 %0, 1
  %16 = mul nsw i32 %14, %15
  %17 = icmp slt i32 %16, 56
  %18 = select i1 %17, i32 %16, i32 56
  %19 = mul i32 %14, %0
  %20 = icmp slt i32 %19, 56
  %21 = select i1 %20, i32 %19, i32 56
  %22 = icmp slt i32 %21, %18
  br i1 %22, label %for_body.preheader, label %for_end, !prof !1

for_body.preheader:                               ; preds = %entry
  %23 = xor i32 %19, -1
  %24 = icmp sgt i32 %23, -57
  %smax = select i1 %24, i32 %23, i32 -57
  %25 = mul i32 %smax, -3584
  %26 = add i32 %25, -3584
  br label %for_body

for_body:                                         ; preds = %for_body.preheader, %for_end3
  %indvar = phi i32 [ 0, %for_body.preheader ], [ %indvar.next, %for_end3 ]
  %27 = phi i32 [ %21, %for_body.preheader ], [ %37, %for_end3 ]
  %28 = mul i32 %indvar, 3584
  %29 = add i32 %26, %28
  %30 = alloca [64 x i16], align 16
  %31 = mul nsw i32 %27, 56
  %scevgep2425 = bitcast [64 x i16]* %30 to i8*
  %scevgep24.1 = getelementptr inbounds [64 x i16], [64 x i16]* %30, i32 0, i32 8
  %scevgep2425.1 = bitcast i16* %scevgep24.1 to i8*
  %scevgep24.2 = getelementptr inbounds [64 x i16], [64 x i16]* %30, i32 0, i32 16
  %scevgep2425.2 = bitcast i16* %scevgep24.2 to i8*
  %scevgep24.3 = getelementptr inbounds [64 x i16], [64 x i16]* %30, i32 0, i32 24
  %scevgep2425.3 = bitcast i16* %scevgep24.3 to i8*
  %scevgep24.4 = getelementptr inbounds [64 x i16], [64 x i16]* %30, i32 0, i32 32
  %scevgep2425.4 = bitcast i16* %scevgep24.4 to i8*
  %scevgep24.5 = getelementptr inbounds [64 x i16], [64 x i16]* %30, i32 0, i32 40
  %scevgep2425.5 = bitcast i16* %scevgep24.5 to i8*
  %scevgep24.6 = getelementptr inbounds [64 x i16], [64 x i16]* %30, i32 0, i32 48
  %scevgep2425.6 = bitcast i16* %scevgep24.6 to i8*
  %scevgep24.7 = getelementptr inbounds [64 x i16], [64 x i16]* %30, i32 0, i32 56
  %scevgep2425.7 = bitcast i16* %scevgep24.7 to i8*
  br label %for_begin4.preheader

for_end:                                          ; preds = %for_end3, %entry
  ret i32 0

for_begin4.preheader:                             ; preds = %for_begin13.preheader, %for_body
  %32 = phi i32 [ 0, %for_body ], [ %46, %for_begin13.preheader ]
  %33 = shl i32 %32, 6
  %34 = add i32 %29, %33
  %35 = add nsw i32 %32, %31
  %36 = mul nsw i32 %35, 3
  br label %for_body5

for_end3:                                         ; preds = %for_begin13.preheader
  %37 = add nsw i32 %27, 1
  %38 = icmp slt i32 %37, %18
  %indvar.next = add nuw i32 %indvar, 1
  br i1 %38, label %for_body, label %for_end, !prof !1

for_begin13.preheader:                            ; preds = %for_end9
  %scevgep = getelementptr i16, i16* %10, i32 %34
  %scevgep23 = bitcast i16* %scevgep to i8*
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 2 %scevgep23, i8* nonnull align 16 %scevgep2425, i32 16, i1 false)
  %39 = or i32 %34, 8
  %scevgep.1 = getelementptr i16, i16* %10, i32 %39
  %scevgep23.1 = bitcast i16* %scevgep.1 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 2 %scevgep23.1, i8* nonnull align 16 %scevgep2425.1, i32 16, i1 false)
  %40 = or i32 %34, 16
  %scevgep.2 = getelementptr i16, i16* %10, i32 %40
  %scevgep23.2 = bitcast i16* %scevgep.2 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 2 %scevgep23.2, i8* nonnull align 16 %scevgep2425.2, i32 16, i1 false)
  %41 = or i32 %34, 24
  %scevgep.3 = getelementptr i16, i16* %10, i32 %41
  %scevgep23.3 = bitcast i16* %scevgep.3 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 2 %scevgep23.3, i8* nonnull align 16 %scevgep2425.3, i32 16, i1 false)
  %42 = or i32 %34, 32
  %scevgep.4 = getelementptr i16, i16* %10, i32 %42
  %scevgep23.4 = bitcast i16* %scevgep.4 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 2 %scevgep23.4, i8* nonnull align 16 %scevgep2425.4, i32 16, i1 false)
  %43 = or i32 %34, 40
  %scevgep.5 = getelementptr i16, i16* %10, i32 %43
  %scevgep23.5 = bitcast i16* %scevgep.5 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 2 %scevgep23.5, i8* nonnull align 16 %scevgep2425.5, i32 16, i1 false)
  %44 = or i32 %34, 48
  %scevgep.6 = getelementptr i16, i16* %10, i32 %44
  %scevgep23.6 = bitcast i16* %scevgep.6 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 2 %scevgep23.6, i8* nonnull align 16 %scevgep2425.6, i32 16, i1 false)
  %45 = or i32 %34, 56
  %scevgep.7 = getelementptr i16, i16* %10, i32 %45
  %scevgep23.7 = bitcast i16* %scevgep.7 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* align 2 %scevgep23.7, i8* nonnull align 16 %scevgep2425.7, i32 16, i1 false)
  %46 = add nuw nsw i32 %32, 1
  %exitcond27 = icmp eq i32 %46, 56
  br i1 %exitcond27, label %for_end3, label %for_begin4.preheader, !prof !24

for_body5:                                        ; preds = %for_end9, %for_begin4.preheader
  %47 = phi i32 [ 0, %for_begin4.preheader ], [ %240, %for_end9 ]
  %48 = shl nsw i32 %47, 3
  %49 = getelementptr inbounds [64 x i16], [64 x i16]* %30, i32 0, i32 %48
  %50 = bitcast i16* %49 to <8 x i16>*
  store <8 x i16> zeroinitializer, <8 x i16>* %50, align 16, !tbaa !162
  %51 = mul nuw nsw i32 %47, 3
  br label %for_begin10.preheader

for_begin10.preheader:                            ; preds = %for_begin10.preheader, %for_body5
  %.lcssa20 = phi <8 x i16> [ zeroinitializer, %for_body5 ], [ %238, %for_begin10.preheader ]
  %52 = phi i32 [ 0, %for_body5 ], [ %239, %for_begin10.preheader ]
  %53 = add nsw i32 %52, %36
  %54 = add nuw nsw i32 %52, %51
  %55 = mul i32 %53, 24
  %56 = getelementptr inbounds i8, i8* %7, i32 %55
  %57 = bitcast i8* %56 to <8 x i8>*
  %58 = load <8 x i8>, <8 x i8>* %57, align 8, !tbaa !159
  %59 = mul i32 %54, 192
  %60 = getelementptr inbounds i8, i8* %4, i32 %59
  %61 = bitcast i8* %60 to <8 x i8>*
  %62 = load <8 x i8>, <8 x i8>* %61, align 64, !tbaa !156
  %63 = and <8 x i8> %62, %58
  %64 = tail call <8 x i8> @llvm.ctpop.v8i8(<8 x i8> %63)
  %65 = or i32 %59, 8
  %66 = getelementptr inbounds i8, i8* %4, i32 %65
  %67 = bitcast i8* %66 to <8 x i8>*
  %68 = load <8 x i8>, <8 x i8>* %67, align 8, !tbaa !156
  %69 = and <8 x i8> %68, %58
  %70 = tail call <8 x i8> @llvm.ctpop.v8i8(<8 x i8> %69)
  %71 = tail call <8 x i8> @llvm.arm.neon.vpadd.v8i8(<8 x i8> %64, <8 x i8> %70)
  %72 = or i32 %59, 16
  %73 = getelementptr inbounds i8, i8* %4, i32 %72
  %74 = bitcast i8* %73 to <8 x i8>*
  %75 = load <8 x i8>, <8 x i8>* %74, align 16, !tbaa !156
  %76 = and <8 x i8> %75, %58
  %77 = tail call <8 x i8> @llvm.ctpop.v8i8(<8 x i8> %76)
  %78 = or i32 %59, 24
  %79 = getelementptr inbounds i8, i8* %4, i32 %78
  %80 = bitcast i8* %79 to <8 x i8>*
  %81 = load <8 x i8>, <8 x i8>* %80, align 8, !tbaa !156
  %82 = and <8 x i8> %81, %58
  %83 = tail call <8 x i8> @llvm.ctpop.v8i8(<8 x i8> %82)
  %84 = tail call <8 x i8> @llvm.arm.neon.vpadd.v8i8(<8 x i8> %77, <8 x i8> %83)
  %85 = tail call <8 x i8> @llvm.arm.neon.vpadd.v8i8(<8 x i8> %71, <8 x i8> %84)
  %86 = or i32 %59, 32
  %87 = getelementptr inbounds i8, i8* %4, i32 %86
  %88 = bitcast i8* %87 to <8 x i8>*
  %89 = load <8 x i8>, <8 x i8>* %88, align 32, !tbaa !156
  %90 = and <8 x i8> %89, %58
  %91 = tail call <8 x i8> @llvm.ctpop.v8i8(<8 x i8> %90)
  %92 = or i32 %59, 40
  %93 = getelementptr inbounds i8, i8* %4, i32 %92
  %94 = bitcast i8* %93 to <8 x i8>*
  %95 = load <8 x i8>, <8 x i8>* %94, align 8, !tbaa !156
  %96 = and <8 x i8> %95, %58
  %97 = tail call <8 x i8> @llvm.ctpop.v8i8(<8 x i8> %96)
  %98 = tail call <8 x i8> @llvm.arm.neon.vpadd.v8i8(<8 x i8> %91, <8 x i8> %97)
  %99 = or i32 %59, 48
  %100 = getelementptr inbounds i8, i8* %4, i32 %99
  %101 = bitcast i8* %100 to <8 x i8>*
  %102 = load <8 x i8>, <8 x i8>* %101, align 16, !tbaa !156
  %103 = and <8 x i8> %102, %58
  %104 = tail call <8 x i8> @llvm.ctpop.v8i8(<8 x i8> %103)
  %105 = or i32 %59, 56
  %106 = getelementptr inbounds i8, i8* %4, i32 %105
  %107 = bitcast i8* %106 to <8 x i8>*
  %108 = load <8 x i8>, <8 x i8>* %107, align 8, !tbaa !156
  %109 = and <8 x i8> %108, %58
  %110 = tail call <8 x i8> @llvm.ctpop.v8i8(<8 x i8> %109)
  %111 = tail call <8 x i8> @llvm.arm.neon.vpadd.v8i8(<8 x i8> %104, <8 x i8> %110)
  %112 = tail call <8 x i8> @llvm.arm.neon.vpadd.v8i8(<8 x i8> %98, <8 x i8> %111)
  %113 = shufflevector <8 x i8> %85, <8 x i8> %112, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %114 = tail call <8 x i16> @llvm.arm.neon.vpadalu.v8i16.v16i8(<8 x i16> %.lcssa20, <16 x i8> %113)
  %115 = mul i32 %53, 24
  %116 = add i32 %115, 8
  %117 = getelementptr inbounds i8, i8* %7, i32 %116
  %118 = bitcast i8* %117 to <8 x i8>*
  %119 = load <8 x i8>, <8 x i8>* %118, align 8, !tbaa !159
  %120 = mul i32 %54, 192
  %121 = add i32 %120, 64
  %122 = getelementptr inbounds i8, i8* %4, i32 %121
  %123 = bitcast i8* %122 to <8 x i8>*
  %124 = load <8 x i8>, <8 x i8>* %123, align 64, !tbaa !156
  %125 = and <8 x i8> %124, %119
  %126 = tail call <8 x i8> @llvm.ctpop.v8i8(<8 x i8> %125)
  %127 = or i32 %121, 8
  %128 = getelementptr inbounds i8, i8* %4, i32 %127
  %129 = bitcast i8* %128 to <8 x i8>*
  %130 = load <8 x i8>, <8 x i8>* %129, align 8, !tbaa !156
  %131 = and <8 x i8> %130, %119
  %132 = tail call <8 x i8> @llvm.ctpop.v8i8(<8 x i8> %131)
  %133 = tail call <8 x i8> @llvm.arm.neon.vpadd.v8i8(<8 x i8> %126, <8 x i8> %132)
  %134 = or i32 %121, 16
  %135 = getelementptr inbounds i8, i8* %4, i32 %134
  %136 = bitcast i8* %135 to <8 x i8>*
  %137 = load <8 x i8>, <8 x i8>* %136, align 16, !tbaa !156
  %138 = and <8 x i8> %137, %119
  %139 = tail call <8 x i8> @llvm.ctpop.v8i8(<8 x i8> %138)
  %140 = or i32 %121, 24
  %141 = getelementptr inbounds i8, i8* %4, i32 %140
  %142 = bitcast i8* %141 to <8 x i8>*
  %143 = load <8 x i8>, <8 x i8>* %142, align 8, !tbaa !156
  %144 = and <8 x i8> %143, %119
  %145 = tail call <8 x i8> @llvm.ctpop.v8i8(<8 x i8> %144)
  %146 = tail call <8 x i8> @llvm.arm.neon.vpadd.v8i8(<8 x i8> %139, <8 x i8> %145)
  %147 = tail call <8 x i8> @llvm.arm.neon.vpadd.v8i8(<8 x i8> %133, <8 x i8> %146)
  %148 = or i32 %121, 32
  %149 = getelementptr inbounds i8, i8* %4, i32 %148
  %150 = bitcast i8* %149 to <8 x i8>*
  %151 = load <8 x i8>, <8 x i8>* %150, align 32, !tbaa !156
  %152 = and <8 x i8> %151, %119
  %153 = tail call <8 x i8> @llvm.ctpop.v8i8(<8 x i8> %152)
  %154 = or i32 %121, 40
  %155 = getelementptr inbounds i8, i8* %4, i32 %154
  %156 = bitcast i8* %155 to <8 x i8>*
  %157 = load <8 x i8>, <8 x i8>* %156, align 8, !tbaa !156
  %158 = and <8 x i8> %157, %119
  %159 = tail call <8 x i8> @llvm.ctpop.v8i8(<8 x i8> %158)
  %160 = tail call <8 x i8> @llvm.arm.neon.vpadd.v8i8(<8 x i8> %153, <8 x i8> %159)
  %161 = or i32 %121, 48
  %162 = getelementptr inbounds i8, i8* %4, i32 %161
  %163 = bitcast i8* %162 to <8 x i8>*
  %164 = load <8 x i8>, <8 x i8>* %163, align 16, !tbaa !156
  %165 = and <8 x i8> %164, %119
  %166 = tail call <8 x i8> @llvm.ctpop.v8i8(<8 x i8> %165)
  %167 = or i32 %121, 56
  %168 = getelementptr inbounds i8, i8* %4, i32 %167
  %169 = bitcast i8* %168 to <8 x i8>*
  %170 = load <8 x i8>, <8 x i8>* %169, align 8, !tbaa !156
  %171 = and <8 x i8> %170, %119
  %172 = tail call <8 x i8> @llvm.ctpop.v8i8(<8 x i8> %171)
  %173 = tail call <8 x i8> @llvm.arm.neon.vpadd.v8i8(<8 x i8> %166, <8 x i8> %172)
  %174 = tail call <8 x i8> @llvm.arm.neon.vpadd.v8i8(<8 x i8> %160, <8 x i8> %173)
  %175 = shufflevector <8 x i8> %147, <8 x i8> %174, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %176 = tail call <8 x i16> @llvm.arm.neon.vpadalu.v8i16.v16i8(<8 x i16> %114, <16 x i8> %175)
  %177 = mul i32 %53, 24
  %178 = add i32 %177, 16
  %179 = getelementptr inbounds i8, i8* %7, i32 %178
  %180 = bitcast i8* %179 to <8 x i8>*
  %181 = load <8 x i8>, <8 x i8>* %180, align 8, !tbaa !159
  %182 = mul i32 %54, 192
  %183 = add i32 %182, 128
  %184 = getelementptr inbounds i8, i8* %4, i32 %183
  %185 = bitcast i8* %184 to <8 x i8>*
  %186 = load <8 x i8>, <8 x i8>* %185, align 64, !tbaa !156
  %187 = and <8 x i8> %186, %181
  %188 = tail call <8 x i8> @llvm.ctpop.v8i8(<8 x i8> %187)
  %189 = or i32 %183, 8
  %190 = getelementptr inbounds i8, i8* %4, i32 %189
  %191 = bitcast i8* %190 to <8 x i8>*
  %192 = load <8 x i8>, <8 x i8>* %191, align 8, !tbaa !156
  %193 = and <8 x i8> %192, %181
  %194 = tail call <8 x i8> @llvm.ctpop.v8i8(<8 x i8> %193)
  %195 = tail call <8 x i8> @llvm.arm.neon.vpadd.v8i8(<8 x i8> %188, <8 x i8> %194)
  %196 = or i32 %183, 16
  %197 = getelementptr inbounds i8, i8* %4, i32 %196
  %198 = bitcast i8* %197 to <8 x i8>*
  %199 = load <8 x i8>, <8 x i8>* %198, align 16, !tbaa !156
  %200 = and <8 x i8> %199, %181
  %201 = tail call <8 x i8> @llvm.ctpop.v8i8(<8 x i8> %200)
  %202 = or i32 %183, 24
  %203 = getelementptr inbounds i8, i8* %4, i32 %202
  %204 = bitcast i8* %203 to <8 x i8>*
  %205 = load <8 x i8>, <8 x i8>* %204, align 8, !tbaa !156
  %206 = and <8 x i8> %205, %181
  %207 = tail call <8 x i8> @llvm.ctpop.v8i8(<8 x i8> %206)
  %208 = tail call <8 x i8> @llvm.arm.neon.vpadd.v8i8(<8 x i8> %201, <8 x i8> %207)
  %209 = tail call <8 x i8> @llvm.arm.neon.vpadd.v8i8(<8 x i8> %195, <8 x i8> %208)
  %210 = or i32 %183, 32
  %211 = getelementptr inbounds i8, i8* %4, i32 %210
  %212 = bitcast i8* %211 to <8 x i8>*
  %213 = load <8 x i8>, <8 x i8>* %212, align 32, !tbaa !156
  %214 = and <8 x i8> %213, %181
  %215 = tail call <8 x i8> @llvm.ctpop.v8i8(<8 x i8> %214)
  %216 = or i32 %183, 40
  %217 = getelementptr inbounds i8, i8* %4, i32 %216
  %218 = bitcast i8* %217 to <8 x i8>*
  %219 = load <8 x i8>, <8 x i8>* %218, align 8, !tbaa !156
  %220 = and <8 x i8> %219, %181
  %221 = tail call <8 x i8> @llvm.ctpop.v8i8(<8 x i8> %220)
  %222 = tail call <8 x i8> @llvm.arm.neon.vpadd.v8i8(<8 x i8> %215, <8 x i8> %221)
  %223 = or i32 %183, 48
  %224 = getelementptr inbounds i8, i8* %4, i32 %223
  %225 = bitcast i8* %224 to <8 x i8>*
  %226 = load <8 x i8>, <8 x i8>* %225, align 16, !tbaa !156
  %227 = and <8 x i8> %226, %181
  %228 = tail call <8 x i8> @llvm.ctpop.v8i8(<8 x i8> %227)
  %229 = or i32 %183, 56
  %230 = getelementptr inbounds i8, i8* %4, i32 %229
  %231 = bitcast i8* %230 to <8 x i8>*
  %232 = load <8 x i8>, <8 x i8>* %231, align 8, !tbaa !156
  %233 = and <8 x i8> %232, %181
  %234 = tail call <8 x i8> @llvm.ctpop.v8i8(<8 x i8> %233)
  %235 = tail call <8 x i8> @llvm.arm.neon.vpadd.v8i8(<8 x i8> %228, <8 x i8> %234)
  %236 = tail call <8 x i8> @llvm.arm.neon.vpadd.v8i8(<8 x i8> %222, <8 x i8> %235)
  %237 = shufflevector <8 x i8> %209, <8 x i8> %236, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  %238 = tail call <8 x i16> @llvm.arm.neon.vpadalu.v8i16.v16i8(<8 x i16> %176, <16 x i8> %237)
  %239 = add nuw nsw i32 %52, 1
  %exitcond = icmp eq i32 %239, 3
  br i1 %exitcond, label %for_end9, label %for_begin10.preheader, !prof !24

for_end9:                                         ; preds = %for_begin10.preheader
  store <8 x i16> %238, <8 x i16>* %50, align 16, !tbaa !162
  %240 = add nuw nsw i32 %47, 1
  %exitcond21 = icmp eq i32 %240, 8
  br i1 %exitcond21, label %for_begin13.preheader, label %for_body5, !prof !24
}

; Function Attrs: nounwind readnone speculatable
declare <8 x i8> @llvm.ctpop.v8i8(<8 x i8>) #3

; Function Attrs: nounwind readnone
declare <8 x i8> @llvm.arm.neon.vpadd.v8i8(<8 x i8>, <8 x i8>) #4

; Function Attrs: nounwind readnone
declare <8 x i16> @llvm.arm.neon.vpadalu.v8i16.v16i8(<8 x i16>, <16 x i8>) #4

; Function Attrs: argmemonly nounwind
declare void @llvm.memset.p0i8.i32(i8* nocapture writeonly, i8, i32, i1) #5

; Function Attrs: argmemonly nounwind
declare void @llvm.memcpy.p0i8.p0i8.i32(i8* nocapture writeonly, i8* nocapture readonly, i32, i1) #5

attributes #0 = { noinline }
attributes #1 = { norecurse nounwind }
attributes #2 = { nounwind }
attributes #3 = { nounwind readnone speculatable }
attributes #4 = { nounwind readnone }
attributes #5 = { argmemonly nounwind }

!llvm.module.flags = !{!0}

!0 = !{i32 2, !"tvm_target", !"llvm -device=arm_cpu -target=arm-linux-gnueabihf -mattr=+neon"}
!1 = !{!"branch_weights", i32 1048576, i32 1}
!2 = !{!3, !3, i64 0}
!3 = !{!"ctx_ptr", !4, i64 0}
!4 = !{!"tvm-tbaa"}
!5 = !{!6, !6, i64 0}
!6 = !{!"0x217a4c0.w1.b0", !7, i64 0}
!7 = !{!"0x217a4c0.w2.b0", !8, i64 0}
!8 = !{!"0x217a4c0.w4.b0", !9, i64 0}
!9 = !{!"0x217a4c0.w8.b0", !10, i64 0}
!10 = !{!"0x217a4c0.w16.b0", !11, i64 0}
!11 = !{!"0x217a4c0.w32.b0", !12, i64 0}
!12 = !{!"0x217a4c0.w64.b0", !13, i64 0}
!13 = !{!"0x217a4c0.w128.b0", !14, i64 0}
!14 = !{!"0x217a4c0.w256.b0", !15, i64 0}
!15 = !{!"0x217a4c0.w512.b0", !16, i64 0}
!16 = !{!"0x217a4c0.w1024.b0", !17, i64 0}
!17 = !{!"int32", !18, i64 0}
!18 = !{!"0x217a4c0", !4, i64 0}
!19 = !{!20, !20, i64 0}
!20 = !{!"0x217a4c0.w1.b1", !7, i64 0}
!21 = !{!22, !22, i64 0}
!22 = !{!"0x217a4c0.w1.b2", !23, i64 0}
!23 = !{!"0x217a4c0.w2.b2", !8, i64 0}
!24 = !{!"branch_weights", i32 1, i32 1048576}
!25 = !{!26, !26, i64 0}
!26 = !{!"0x2267300.w1.b0", !27, i64 0}
!27 = !{!"0x2267300.w2.b0", !28, i64 0}
!28 = !{!"0x2267300.w4.b0", !29, i64 0}
!29 = !{!"0x2267300.w8.b0", !30, i64 0}
!30 = !{!"0x2267300.w16.b0", !31, i64 0}
!31 = !{!"0x2267300.w32.b0", !32, i64 0}
!32 = !{!"0x2267300.w64.b0", !33, i64 0}
!33 = !{!"0x2267300.w128.b0", !34, i64 0}
!34 = !{!"0x2267300.w256.b0", !35, i64 0}
!35 = !{!"0x2267300.w512.b0", !36, i64 0}
!36 = !{!"0x2267300.w1024.b0", !37, i64 0}
!37 = !{!"int64", !38, i64 0}
!38 = !{!"0x2267300", !4, i64 0}
!39 = !{!40, !40, i64 0}
!40 = !{!"0x2267300.w1.b1", !27, i64 0}
!41 = !{!42, !42, i64 0}
!42 = !{!"0x2267300.w2.b2", !28, i64 0}
!43 = !{!44, !44, i64 0}
!44 = !{!"0x20f23b0.w1.b0", !45, i64 0}
!45 = !{!"0x20f23b0.w2.b0", !46, i64 0}
!46 = !{!"0x20f23b0.w4.b0", !47, i64 0}
!47 = !{!"0x20f23b0.w8.b0", !48, i64 0}
!48 = !{!"0x20f23b0.w16.b0", !49, i64 0}
!49 = !{!"0x20f23b0.w32.b0", !50, i64 0}
!50 = !{!"0x20f23b0.w64.b0", !51, i64 0}
!51 = !{!"0x20f23b0.w128.b0", !52, i64 0}
!52 = !{!"0x20f23b0.w256.b0", !53, i64 0}
!53 = !{!"0x20f23b0.w512.b0", !54, i64 0}
!54 = !{!"0x20f23b0.w1024.b0", !55, i64 0}
!55 = !{!"int64", !56, i64 0}
!56 = !{!"0x20f23b0", !4, i64 0}
!57 = !{!58, !58, i64 0}
!58 = !{!"0x20f23b0.w1.b1", !45, i64 0}
!59 = !{!60, !60, i64 0}
!60 = !{!"0x20f23b0.w2.b2", !46, i64 0}
!61 = !{!62, !62, i64 0}
!62 = !{!"0x212ce30.w1.b0", !63, i64 0}
!63 = !{!"0x212ce30.w2.b0", !64, i64 0}
!64 = !{!"0x212ce30.w4.b0", !65, i64 0}
!65 = !{!"0x212ce30.w8.b0", !66, i64 0}
!66 = !{!"0x212ce30.w16.b0", !67, i64 0}
!67 = !{!"0x212ce30.w32.b0", !68, i64 0}
!68 = !{!"0x212ce30.w64.b0", !69, i64 0}
!69 = !{!"0x212ce30.w128.b0", !70, i64 0}
!70 = !{!"0x212ce30.w256.b0", !71, i64 0}
!71 = !{!"0x212ce30.w512.b0", !72, i64 0}
!72 = !{!"0x212ce30.w1024.b0", !73, i64 0}
!73 = !{!"int64", !74, i64 0}
!74 = !{!"0x212ce30", !4, i64 0}
!75 = !{!76, !76, i64 0}
!76 = !{!"0x212ce30.w1.b1", !63, i64 0}
!77 = !{!78, !78, i64 0}
!78 = !{!"0x212ce30.w2.b2", !64, i64 0}
!79 = !{!80, !80, i64 0}
!80 = !{!"0x2240e90.w1.b0", !81, i64 0}
!81 = !{!"0x2240e90.w2.b0", !82, i64 0}
!82 = !{!"0x2240e90.w4.b0", !83, i64 0}
!83 = !{!"0x2240e90.w8.b0", !84, i64 0}
!84 = !{!"0x2240e90.w16.b0", !85, i64 0}
!85 = !{!"0x2240e90.w32.b0", !86, i64 0}
!86 = !{!"0x2240e90.w64.b0", !87, i64 0}
!87 = !{!"0x2240e90.w128.b0", !88, i64 0}
!88 = !{!"0x2240e90.w256.b0", !89, i64 0}
!89 = !{!"0x2240e90.w512.b0", !90, i64 0}
!90 = !{!"0x2240e90.w1024.b0", !91, i64 0}
!91 = !{!"int64", !92, i64 0}
!92 = !{!"0x2240e90", !4, i64 0}
!93 = !{!94, !94, i64 0}
!94 = !{!"0x2240e90.w1.b1", !81, i64 0}
!95 = !{!96, !96, i64 0}
!96 = !{!"0x2240e90.w1.b2", !97, i64 0}
!97 = !{!"0x2240e90.w2.b2", !82, i64 0}
!98 = !{!99, !99, i64 0}
!99 = !{!"0x2240e90.w1.b3", !97, i64 0}
!100 = !{!101, !101, i64 0}
!101 = !{!"0x227fe60.w1.b0", !102, i64 0}
!102 = !{!"0x227fe60.w2.b0", !103, i64 0}
!103 = !{!"0x227fe60.w4.b0", !104, i64 0}
!104 = !{!"0x227fe60.w8.b0", !105, i64 0}
!105 = !{!"0x227fe60.w16.b0", !106, i64 0}
!106 = !{!"0x227fe60.w32.b0", !107, i64 0}
!107 = !{!"0x227fe60.w64.b0", !108, i64 0}
!108 = !{!"0x227fe60.w128.b0", !109, i64 0}
!109 = !{!"0x227fe60.w256.b0", !110, i64 0}
!110 = !{!"0x227fe60.w512.b0", !111, i64 0}
!111 = !{!"0x227fe60.w1024.b0", !112, i64 0}
!112 = !{!"int64", !113, i64 0}
!113 = !{!"0x227fe60", !4, i64 0}
!114 = !{!115, !115, i64 0}
!115 = !{!"0x227fe60.w1.b1", !102, i64 0}
!116 = !{!117, !117, i64 0}
!117 = !{!"0x227fe60.w1.b2", !118, i64 0}
!118 = !{!"0x227fe60.w2.b2", !103, i64 0}
!119 = !{!120, !120, i64 0}
!120 = !{!"0x227fe60.w1.b3", !118, i64 0}
!121 = !{!122, !122, i64 0}
!122 = !{!"0x2284940.w1.b0", !123, i64 0}
!123 = !{!"0x2284940.w2.b0", !124, i64 0}
!124 = !{!"0x2284940.w4.b0", !125, i64 0}
!125 = !{!"0x2284940.w8.b0", !126, i64 0}
!126 = !{!"0x2284940.w16.b0", !127, i64 0}
!127 = !{!"0x2284940.w32.b0", !128, i64 0}
!128 = !{!"0x2284940.w64.b0", !129, i64 0}
!129 = !{!"0x2284940.w128.b0", !130, i64 0}
!130 = !{!"0x2284940.w256.b0", !131, i64 0}
!131 = !{!"0x2284940.w512.b0", !132, i64 0}
!132 = !{!"0x2284940.w1024.b0", !133, i64 0}
!133 = !{!"int64", !134, i64 0}
!134 = !{!"0x2284940", !4, i64 0}
!135 = !{!136, !136, i64 0}
!136 = !{!"0x2284940.w1.b1", !123, i64 0}
!137 = !{!138, !138, i64 0}
!138 = !{!"0x2284940.w1.b2", !139, i64 0}
!139 = !{!"0x2284940.w2.b2", !124, i64 0}
!140 = !{!141, !141, i64 0}
!141 = !{!"0x2284940.w1.b3", !139, i64 0}
!142 = !{!143, !143, i64 0}
!143 = !{!"uint8", !144, i64 0}
!144 = !{!"0x1f79080", !4, i64 0}
!145 = !{!146, !146, i64 0}
!146 = !{!"uint8", !147, i64 0}
!147 = !{!"0x2257430", !4, i64 0}
!148 = distinct !{!148, !149}
!149 = !{!"llvm.loop.isvectorized", i32 1}
!150 = !{!"branch_weights", i32 -2147483648, i32 8192}
!151 = !{!152, !152, i64 0}
!152 = !{!"uint8", !153, i64 0}
!153 = !{!"0x21319d0", !4, i64 0}
!154 = distinct !{!154, !149}
!155 = distinct !{!155, !149}
!156 = !{!157, !157, i64 0}
!157 = !{!"uint8", !158, i64 0}
!158 = !{!"0x2194050", !4, i64 0}
!159 = !{!160, !160, i64 0}
!160 = !{!"uint8", !161, i64 0}
!161 = !{!"0x21ea480", !4, i64 0}
!162 = !{!163, !163, i64 0}
!163 = !{!"uint16", !164, i64 0}
!164 = !{!"0x2179840", !4, i64 0}
