# generated from ament/cmake/core/templates/nameConfig.cmake.in

# prevent multiple inclusion
if(_snn_aloha_CONFIG_INCLUDED)
  # ensure to keep the found flag the same
  if(NOT DEFINED snn_aloha_FOUND)
    # explicitly set it to FALSE, otherwise CMake will set it to TRUE
    set(snn_aloha_FOUND FALSE)
  elseif(NOT snn_aloha_FOUND)
    # use separate condition to avoid uninitialized variable warning
    set(snn_aloha_FOUND FALSE)
  endif()
  return()
endif()
set(_snn_aloha_CONFIG_INCLUDED TRUE)

# output package information
if(NOT snn_aloha_FIND_QUIETLY)
  message(STATUS "Found snn_aloha: 0.0.0 (${snn_aloha_DIR})")
endif()

# warn when using a deprecated package
if(NOT "" STREQUAL "")
  set(_msg "Package 'snn_aloha' is deprecated")
  # append custom deprecation text if available
  if(NOT "" STREQUAL "TRUE")
    set(_msg "${_msg} ()")
  endif()
  # optionally quiet the deprecation message
  if(NOT ${snn_aloha_DEPRECATED_QUIET})
    message(DEPRECATION "${_msg}")
  endif()
endif()

# flag package as ament-based to distinguish it after being find_package()-ed
set(snn_aloha_FOUND_AMENT_PACKAGE TRUE)

# include all config extra files
set(_extras "")
foreach(_extra ${_extras})
  include("${snn_aloha_DIR}/${_extra}")
endforeach()
