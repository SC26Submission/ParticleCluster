# Install script for directory: /global/homes/c/crren/draco

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "0")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/attributes" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/attributes/attribute_octahedron_transform.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/attributes" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/attributes/attribute_quantization_transform.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/attributes" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/attributes/attribute_transform.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/attributes" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/attributes/attribute_transform_data.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/attributes" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/attributes/attribute_transform_type.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/attributes" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/attributes/geometry_attribute.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/attributes" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/attributes/geometry_indices.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/attributes" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/attributes/point_attribute.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/compression/attributes" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/compression/attributes/attributes_decoder.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/compression/attributes" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/compression/attributes/attributes_decoder_interface.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/compression/attributes" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/compression/attributes/kd_tree_attributes_decoder.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/compression/attributes" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/compression/attributes/kd_tree_attributes_shared.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/compression/attributes" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/compression/attributes/mesh_attribute_indices_encoding_data.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/compression/attributes" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/compression/attributes/normal_compression_utils.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/compression/attributes" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/compression/attributes/point_d_vector.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/compression/attributes" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/compression/attributes/sequential_attribute_decoder.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/compression/attributes" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/compression/attributes/sequential_attribute_decoders_controller.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/compression/attributes" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/compression/attributes/sequential_integer_attribute_decoder.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/compression/attributes" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/compression/attributes/sequential_normal_attribute_decoder.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/compression/attributes" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/compression/attributes/sequential_quantization_attribute_decoder.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/compression/attributes" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/compression/attributes/attributes_encoder.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/compression/attributes" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/compression/attributes/kd_tree_attributes_encoder.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/compression/attributes" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/compression/attributes/linear_sequencer.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/compression/attributes" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/compression/attributes/points_sequencer.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/compression/attributes" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/compression/attributes/sequential_attribute_encoder.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/compression/attributes" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/compression/attributes/sequential_attribute_encoders_controller.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/compression/attributes" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/compression/attributes/sequential_integer_attribute_encoder.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/compression/attributes" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/compression/attributes/sequential_normal_attribute_encoder.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/compression/attributes" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/compression/attributes/sequential_quantization_attribute_encoder.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/compression/attributes/prediction_schemes" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/compression/attributes/prediction_schemes/mesh_prediction_scheme_constrained_multi_parallelogram_decoder.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/compression/attributes/prediction_schemes" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/compression/attributes/prediction_schemes/mesh_prediction_scheme_constrained_multi_parallelogram_shared.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/compression/attributes/prediction_schemes" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/compression/attributes/prediction_schemes/mesh_prediction_scheme_data.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/compression/attributes/prediction_schemes" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/compression/attributes/prediction_schemes/mesh_prediction_scheme_decoder.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/compression/attributes/prediction_schemes" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/compression/attributes/prediction_schemes/mesh_prediction_scheme_geometric_normal_decoder.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/compression/attributes/prediction_schemes" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/compression/attributes/prediction_schemes/mesh_prediction_scheme_geometric_normal_predictor_area.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/compression/attributes/prediction_schemes" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/compression/attributes/prediction_schemes/mesh_prediction_scheme_geometric_normal_predictor_base.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/compression/attributes/prediction_schemes" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/compression/attributes/prediction_schemes/mesh_prediction_scheme_multi_parallelogram_decoder.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/compression/attributes/prediction_schemes" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/compression/attributes/prediction_schemes/mesh_prediction_scheme_parallelogram_encoder.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/compression/attributes/prediction_schemes" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/compression/attributes/prediction_schemes/mesh_prediction_scheme_parallelogram_shared.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/compression/attributes/prediction_schemes" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/compression/attributes/prediction_schemes/mesh_prediction_scheme_tex_coords_decoder.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/compression/attributes/prediction_schemes" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/compression/attributes/prediction_schemes/mesh_prediction_scheme_tex_coords_portable_decoder.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/compression/attributes/prediction_schemes" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/compression/attributes/prediction_schemes/mesh_prediction_scheme_tex_coords_portable_predictor.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/compression/attributes/prediction_schemes" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/compression/attributes/prediction_schemes/prediction_scheme_decoder.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/compression/attributes/prediction_schemes" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/compression/attributes/prediction_schemes/prediction_scheme_decoder_factory.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/compression/attributes/prediction_schemes" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/compression/attributes/prediction_schemes/prediction_scheme_decoder_interface.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/compression/attributes/prediction_schemes" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/compression/attributes/prediction_schemes/prediction_scheme_decoding_transform.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/compression/attributes/prediction_schemes" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/compression/attributes/prediction_schemes/prediction_scheme_delta_decoder.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/compression/attributes/prediction_schemes" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/compression/attributes/prediction_schemes/prediction_scheme_factory.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/compression/attributes/prediction_schemes" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/compression/attributes/prediction_schemes/prediction_scheme_interface.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/compression/attributes/prediction_schemes" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/compression/attributes/prediction_schemes/prediction_scheme_normal_octahedron_canonicalized_decoding_transform.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/compression/attributes/prediction_schemes" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/compression/attributes/prediction_schemes/prediction_scheme_normal_octahedron_canonicalized_transform_base.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/compression/attributes/prediction_schemes" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/compression/attributes/prediction_schemes/prediction_scheme_normal_octahedron_decoding_transform.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/compression/attributes/prediction_schemes" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/compression/attributes/prediction_schemes/prediction_scheme_normal_octahedron_transform_base.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/compression/attributes/prediction_schemes" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/compression/attributes/prediction_schemes/prediction_scheme_wrap_decoding_transform.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/compression/attributes/prediction_schemes" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/compression/attributes/prediction_schemes/prediction_scheme_wrap_transform_base.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/compression/attributes/prediction_schemes" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/compression/attributes/prediction_schemes/mesh_prediction_scheme_constrained_multi_parallelogram_encoder.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/compression/attributes/prediction_schemes" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/compression/attributes/prediction_schemes/mesh_prediction_scheme_encoder.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/compression/attributes/prediction_schemes" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/compression/attributes/prediction_schemes/mesh_prediction_scheme_geometric_normal_encoder.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/compression/attributes/prediction_schemes" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/compression/attributes/prediction_schemes/mesh_prediction_scheme_multi_parallelogram_encoder.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/compression/attributes/prediction_schemes" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/compression/attributes/prediction_schemes/mesh_prediction_scheme_tex_coords_encoder.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/compression/attributes/prediction_schemes" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/compression/attributes/prediction_schemes/mesh_prediction_scheme_tex_coords_portable_encoder.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/compression/attributes/prediction_schemes" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/compression/attributes/prediction_schemes/prediction_scheme_delta_encoder.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/compression/attributes/prediction_schemes" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/compression/attributes/prediction_schemes/prediction_scheme_encoder.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/compression/attributes/prediction_schemes" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/compression/attributes/prediction_schemes/prediction_scheme_encoder_factory.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/compression/attributes/prediction_schemes" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/compression/attributes/prediction_schemes/prediction_scheme_encoder_interface.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/compression/attributes/prediction_schemes" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/compression/attributes/prediction_schemes/prediction_scheme_encoding_transform.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/compression/attributes/prediction_schemes" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/compression/attributes/prediction_schemes/prediction_scheme_normal_octahedron_canonicalized_encoding_transform.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/compression/attributes/prediction_schemes" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/compression/attributes/prediction_schemes/prediction_scheme_normal_octahedron_encoding_transform.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/compression/attributes/prediction_schemes" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/compression/attributes/prediction_schemes/prediction_scheme_wrap_encoding_transform.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/compression/bit_coders" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/compression/bit_coders/adaptive_rans_bit_coding_shared.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/compression/bit_coders" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/compression/bit_coders/adaptive_rans_bit_decoder.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/compression/bit_coders" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/compression/bit_coders/adaptive_rans_bit_encoder.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/compression/bit_coders" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/compression/bit_coders/direct_bit_decoder.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/compression/bit_coders" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/compression/bit_coders/direct_bit_encoder.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/compression/bit_coders" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/compression/bit_coders/folded_integer_bit_decoder.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/compression/bit_coders" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/compression/bit_coders/folded_integer_bit_encoder.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/compression/bit_coders" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/compression/bit_coders/rans_bit_decoder.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/compression/bit_coders" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/compression/bit_coders/rans_bit_encoder.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/compression/bit_coders" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/compression/bit_coders/symbol_bit_decoder.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/compression/bit_coders" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/compression/bit_coders/symbol_bit_encoder.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/compression/config" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/compression/config/compression_shared.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/compression/config" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/compression/config/draco_options.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/compression/config" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/compression/config/encoder_options.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/compression/config" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/compression/config/encoding_features.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/compression/config" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/compression/config/decoder_options.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/compression" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/compression/decode.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/compression" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/compression/encode.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/compression" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/compression/encode_base.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/compression" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/compression/expert_encode.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/compression/entropy" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/compression/entropy/ans.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/compression/entropy" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/compression/entropy/rans_symbol_coding.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/compression/entropy" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/compression/entropy/rans_symbol_decoder.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/compression/entropy" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/compression/entropy/rans_symbol_encoder.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/compression/entropy" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/compression/entropy/shannon_entropy.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/compression/entropy" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/compression/entropy/symbol_decoding.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/compression/entropy" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/compression/entropy/symbol_encoding.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/compression/mesh/traverser" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/compression/mesh/traverser/depth_first_traverser.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/compression/mesh/traverser" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/compression/mesh/traverser/max_prediction_degree_traverser.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/compression/mesh/traverser" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/compression/mesh/traverser/mesh_attribute_indices_encoding_observer.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/compression/mesh/traverser" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/compression/mesh/traverser/mesh_traversal_sequencer.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/compression/mesh/traverser" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/compression/mesh/traverser/traverser_base.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/compression/mesh" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/compression/mesh/mesh_decoder.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/compression/mesh" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/compression/mesh/mesh_edgebreaker_decoder.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/compression/mesh" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/compression/mesh/mesh_edgebreaker_decoder_impl.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/compression/mesh" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/compression/mesh/mesh_edgebreaker_decoder_impl_interface.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/compression/mesh" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/compression/mesh/mesh_edgebreaker_shared.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/compression/mesh" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/compression/mesh/mesh_edgebreaker_traversal_decoder.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/compression/mesh" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/compression/mesh/mesh_edgebreaker_traversal_predictive_decoder.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/compression/mesh" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/compression/mesh/mesh_edgebreaker_traversal_valence_decoder.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/compression/mesh" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/compression/mesh/mesh_sequential_decoder.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/compression/mesh" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/compression/mesh/mesh_edgebreaker_encoder.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/compression/mesh" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/compression/mesh/mesh_edgebreaker_encoder_impl.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/compression/mesh" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/compression/mesh/mesh_edgebreaker_encoder_impl_interface.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/compression/mesh" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/compression/mesh/mesh_edgebreaker_traversal_encoder.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/compression/mesh" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/compression/mesh/mesh_edgebreaker_traversal_predictive_encoder.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/compression/mesh" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/compression/mesh/mesh_edgebreaker_traversal_valence_encoder.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/compression/mesh" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/compression/mesh/mesh_encoder.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/compression/mesh" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/compression/mesh/mesh_sequential_encoder.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/compression" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/compression/draco_compression_options.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/compression/point_cloud" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/compression/point_cloud/point_cloud_decoder.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/compression/point_cloud" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/compression/point_cloud/point_cloud_kd_tree_decoder.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/compression/point_cloud" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/compression/point_cloud/point_cloud_sequential_decoder.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/compression/point_cloud" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/compression/point_cloud/point_cloud_encoder.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/compression/point_cloud" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/compression/point_cloud/point_cloud_kd_tree_encoder.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/compression/point_cloud" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/compression/point_cloud/point_cloud_sequential_encoder.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/core" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/core/bit_utils.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/core" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/core/bounding_box.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/core" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/core/constants.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/core" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/core/cycle_timer.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/core" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/core/data_buffer.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/core" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/core/decoder_buffer.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/core" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/core/divide.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/core" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/core/draco_index_type.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/core" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/core/draco_index_type_vector.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/core" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/core/draco_types.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/core" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/core/draco_version.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/core" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/core/encoder_buffer.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/core" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/core/hash_utils.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/core" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/core/macros.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/core" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/core/math_utils.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/core" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/core/options.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/core" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/core/quantization_utils.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/core" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/core/status.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/core" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/core/status_or.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/core" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/core/varint_decoding.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/core" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/core/varint_encoding.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/core" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/core/vector_d.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/io" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/io/file_reader_factory.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/io" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/io/file_reader_interface.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/io" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/io/file_utils.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/io" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/io/file_writer_factory.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/io" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/io/file_writer_interface.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/io" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/io/file_writer_utils.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/io" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/io/mesh_io.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/io" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/io/obj_decoder.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/io" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/io/obj_encoder.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/io" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/io/parser_utils.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/io" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/io/ply_decoder.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/io" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/io/ply_encoder.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/io" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/io/ply_property_reader.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/io" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/io/ply_property_writer.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/io" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/io/ply_reader.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/io" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/io/stl_decoder.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/io" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/io/stl_encoder.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/io" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/io/point_cloud_io.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/io" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/io/stdio_file_reader.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/io" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/io/stdio_file_writer.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/mesh" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/mesh/corner_table.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/mesh" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/mesh/corner_table_iterators.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/mesh" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/mesh/mesh.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/mesh" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/mesh/mesh_are_equivalent.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/mesh" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/mesh/mesh_attribute_corner_table.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/mesh" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/mesh/mesh_cleanup.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/mesh" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/mesh/mesh_features.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/mesh" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/mesh/mesh_indices.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/mesh" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/mesh/mesh_misc_functions.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/mesh" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/mesh/mesh_stripifier.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/mesh" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/mesh/triangle_soup_mesh_builder.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/mesh" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/mesh/valence_cache.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/metadata" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/metadata/metadata_decoder.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/metadata" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/metadata/metadata_encoder.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/metadata" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/metadata/geometry_metadata.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/metadata" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/metadata/metadata.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/metadata" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/metadata/property_attribute.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/metadata" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/metadata/property_table.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/metadata" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/metadata/structural_metadata.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/metadata" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/metadata/structural_metadata_schema.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/animation" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/animation/keyframe_animation_decoder.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/animation" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/animation/keyframe_animation_encoder.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/animation" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/animation/keyframe_animation.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/point_cloud" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/point_cloud/point_cloud.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/point_cloud" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/point_cloud/point_cloud_builder.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/compression/point_cloud/algorithms" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/compression/point_cloud/algorithms/point_cloud_compression_method.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/compression/point_cloud/algorithms" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/compression/point_cloud/algorithms/point_cloud_types.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/compression/point_cloud/algorithms" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/compression/point_cloud/algorithms/quantize_points_3.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/compression/point_cloud/algorithms" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/compression/point_cloud/algorithms/queuing_policy.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/compression/point_cloud/algorithms" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/compression/point_cloud/algorithms/dynamic_integer_points_kd_tree_decoder.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/compression/point_cloud/algorithms" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/compression/point_cloud/algorithms/float_points_tree_decoder.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/compression/point_cloud/algorithms" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/compression/point_cloud/algorithms/dynamic_integer_points_kd_tree_encoder.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco/compression/point_cloud/algorithms" TYPE FILE FILES "/global/homes/c/crren/draco/src/draco/compression/point_cloud/algorithms/float_points_tree_encoder.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/draco" TYPE FILE FILES "/global/homes/c/crren/draco/build/draco/draco_features.h")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  foreach(file
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/draco_decoder-1.5.7"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/draco_decoder"
      )
    if(EXISTS "${file}" AND
       NOT IS_SYMLINK "${file}")
      file(RPATH_CHECK
           FILE "${file}"
           RPATH "")
    endif()
  endforeach()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES
    "/global/homes/c/crren/draco/build/draco_decoder-1.5.7"
    "/global/homes/c/crren/draco/build/draco_decoder"
    )
  foreach(file
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/draco_decoder-1.5.7"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/draco_decoder"
      )
    if(EXISTS "${file}" AND
       NOT IS_SYMLINK "${file}")
      if(CMAKE_INSTALL_DO_STRIP)
        execute_process(COMMAND "/usr/bin/strip" "${file}")
      endif()
    endif()
  endforeach()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  include("/global/homes/c/crren/draco/build/CMakeFiles/draco_decoder.dir/install-cxx-module-bmi-Release.cmake" OPTIONAL)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  foreach(file
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/draco_encoder-1.5.7"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/draco_encoder"
      )
    if(EXISTS "${file}" AND
       NOT IS_SYMLINK "${file}")
      file(RPATH_CHECK
           FILE "${file}"
           RPATH "")
    endif()
  endforeach()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES
    "/global/homes/c/crren/draco/build/draco_encoder-1.5.7"
    "/global/homes/c/crren/draco/build/draco_encoder"
    )
  foreach(file
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/draco_encoder-1.5.7"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/draco_encoder"
      )
    if(EXISTS "${file}" AND
       NOT IS_SYMLINK "${file}")
      if(CMAKE_INSTALL_DO_STRIP)
        execute_process(COMMAND "/usr/bin/strip" "${file}")
      endif()
    endif()
  endforeach()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  include("/global/homes/c/crren/draco/build/CMakeFiles/draco_encoder.dir/install-cxx-module-bmi-Release.cmake" OPTIONAL)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib64" TYPE STATIC_LIBRARY FILES "/global/homes/c/crren/draco/build/libdraco.a")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  include("/global/homes/c/crren/draco/build/CMakeFiles/draco_static.dir/install-cxx-module-bmi-Release.cmake" OPTIONAL)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib64/pkgconfig" TYPE FILE FILES "/global/homes/c/crren/draco/build/draco.pc")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/cmake/draco/draco-targets.cmake")
    file(DIFFERENT _cmake_export_file_changed FILES
         "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/cmake/draco/draco-targets.cmake"
         "/global/homes/c/crren/draco/build/CMakeFiles/Export/c3fcbaaf5f3ede1976ea6ab78921f9c3/draco-targets.cmake")
    if(_cmake_export_file_changed)
      file(GLOB _cmake_old_config_files "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/cmake/draco/draco-targets-*.cmake")
      if(_cmake_old_config_files)
        string(REPLACE ";" ", " _cmake_old_config_files_text "${_cmake_old_config_files}")
        message(STATUS "Old export file \"$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/share/cmake/draco/draco-targets.cmake\" will be replaced.  Removing files [${_cmake_old_config_files_text}].")
        unset(_cmake_old_config_files_text)
        file(REMOVE ${_cmake_old_config_files})
      endif()
      unset(_cmake_old_config_files)
    endif()
    unset(_cmake_export_file_changed)
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/cmake/draco" TYPE FILE FILES "/global/homes/c/crren/draco/build/CMakeFiles/Export/c3fcbaaf5f3ede1976ea6ab78921f9c3/draco-targets.cmake")
  if(CMAKE_INSTALL_CONFIG_NAME MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/cmake/draco" TYPE FILE FILES "/global/homes/c/crren/draco/build/CMakeFiles/Export/c3fcbaaf5f3ede1976ea6ab78921f9c3/draco-targets-release.cmake")
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/cmake/draco" TYPE FILE FILES
    "/global/homes/c/crren/draco/build/draco-config.cmake"
    "/global/homes/c/crren/draco/build/draco-config-version.cmake"
    )
endif()

if(CMAKE_INSTALL_COMPONENT)
  set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
file(WRITE "/global/homes/c/crren/draco/build/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
