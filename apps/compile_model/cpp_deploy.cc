/*!
 *  Copyright (c) 2017 by Contributors
 * \brief Example code on load and run TVM module.s
 * \file cpp_deploy.cc
 */
#include <dlpack/dlpack.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/packed_func.h>

#include <fstream>
#include <iterator>
#include <algorithm>

void RunModel(tvm::runtime::Module mod_syslib, std::string fname) {
  // Load json graph.
  std::ifstream json_in(fname + ".json", std::ios::in);
  std::string json_data((std::istreambuf_iterator<char>(json_in)), std::istreambuf_iterator<char>());
  json_in.close();

  // Load parameter binary
  std::ifstream params_in(fname + ".params", std::ios::binary);
  std::string params_data((std::istreambuf_iterator<char>(params_in)), std::istreambuf_iterator<char>());
  params_in.close();

  // Convert parameters to TVMByteArray
  TVMByteArray params_arr;
  params_arr.data = params_data.c_str();
  params_arr.size = params_data.length();

  // Allocate the DLPack data structures.
  //
  // Note that we use TVM runtime API to allocate the DLTensor in this example.
  // TVM accept DLPack compatible DLTensors, so function can be invoked
  // as long as we pass correct pointer to DLTensor array.
  //
  // For more information please refer to dlpack.
  // One thing to notice is that DLPack contains alignment requirement for
  // the data pointer and TVM takes advantage of that.
  // If you plan to use your customized data container, please
  // make sure the DLTensor you pass in meet the alignment requirement.
  //
  
  // Define characteristics of input and output tensors.
  int dtype_code = kDLFloat;
  int dtype_bits = 32;
  int dtype_lanes = 1;
  int device_type = kDLCPU;
  int device_id = 0;

  // get global function for graph runtime
  tvm::runtime::Module mod = (*tvm::runtime::Registry::Get("tvm.graph_runtime.create"))(json_data, mod_syslib, device_type, device_id);

  // Set up input tensor.
  DLTensor* x;
  int in_ndim = 3;
  int64_t in_shape[in_ndim] = {224, 224, 3};
  // Load image data saved in binary.
  std::ifstream data_fin("cat.bin", std::ios::binary);
  // number of bytes in image is pixels times 4 bytes per float value.
  data_fin.read(static_cast<char*>(x->data), 224*224*3*4);

  // get the set input function of the module.
  tvm::runtime::PackedFunc set_input = mod.GetFunction("set_input");
  set_input("data", x);

  // Get the load parameter function from the module.
  tvm::runtime::PackedFunc load_params = mod.GetFunction("load_params");
  load_params(params_arr);

  // get the function from the module(run it)
  tvm::runtime::PackedFunc run = mod.GetFunction("run");
  run();

  // Define the output tensor.
  DLTensor* y;
  int out_ndim = 1;
  int64_t out_shape[1] = {1000};
  TVMArrayAlloc(out_shape, out_ndim, dtype_code, dtype_bits, dtype_lanes, device_type, device_id, &y);

  // Get the run function and run our system.
  tvm::runtime::PackedFunc get_output = mod.GetFunction("get_output");
  get_output(0, y);

  // Free the input and output tensors.
  TVMArrayFree(x);
  TVMArrayFree(y);
}

int main(void) {
  LOG(INFO) << "Run system lib graph function on binary image.";
  tvm::runtime::Module mod_syslib = (*tvm::runtime::Registry::Get("module._GetSystemLib"))();
  RunModel(mod_syslib, "models/vggnet");
  return 0;
}
