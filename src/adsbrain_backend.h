// Copyright 2021-2022, MICROSOFT CORPORATION & AFFILIATES. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of MICROSOFT CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include <string>
#include <unordered_map>

// To use the adsbrain backend, you need to 1) derive AdsbrainInferenceModel to
// implement the model-specific logic; 2) implement the C API
// `CreateInferenceModel(...)` to create the model instance. 3) compile the C++
// model inference code into a shared library and put it and all the dependent
// shared libraies to the model serving directory; 4) update config.pbtxt to use
// the adsbrain backend and specify the shared library name and required
// parameters.

namespace triton { namespace backend { namespace adsbrain {

// This class is the base class for the implementation of customized inference
// model using adsbrain backend. The derived class should implement the
// following functions:
// - Initialize: initialize the model instance with the given model config.
// - RunInference: run the inference with the given input data. Currently it
// only supports one input and one output with the string data type. This
// function had better be thread-safe.
// - Destrunctor: destroy the model instance and release the resources.
class AdsbrainInferenceModel {
 public:
  AdsbrainInferenceModel() {}
  virtual ~AdsbrainInferenceModel(){};

  // Initialize the model. All the parameters in config.pbtxt will be passed
  // into this function.
  virtual void Initialize(
      const std::unordered_map<std::string, std::string>& configs) = 0;

  // Run inference on the model using the provided input data and return the
  // output as a string. This function fully controls the output content
  // format/shema. This function had better be thread-safe.
  virtual std::string RunInference(const std::string& input_data) = 0;
};

}}}  // namespace triton::backend::adsbrain

#ifdef __cplusplus
extern "C" {
#endif

// Create a new inference model instance. The returned object is owned by the
// adsbrain backend but the model code need to release the allocated resource in
// the destructor.
std::unique_ptr<triton::backend::adsbrain::AdsbrainInferenceModel>
CreateInferenceModel();

#ifdef __cplusplus
}
#endif
