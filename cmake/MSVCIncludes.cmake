if(MSVC AND CMAKE_GENERATOR MATCHES "Ninja")
    set(_probe "${CMAKE_BINARY_DIR}/CMakeFiles/llmserve-include-probe.cpp")
    file(WRITE "${_probe}" "#include <stddef.h>\n")
    execute_process(
        COMMAND "${CMAKE_CXX_COMPILER}" /nologo /EP /showIncludes "${_probe}"
        OUTPUT_VARIABLE _include_output ERROR_VARIABLE _include_error
        ENCODING ANSI
    )
    string(REGEX MATCH "[^\r\n]*[A-Za-z]:[/\\\\][^\r\n]*stddef\\.h" _include_line
        "${_include_output}\n${_include_error}")
    if(_include_line)
        string(REGEX REPLACE "[A-Za-z]:[/\\\\].*$" "" CMAKE_CL_SHOWINCLUDES_PREFIX "${_include_line}")
    endif()
endif()
