%module mithral

// include files that SWIG will need to see
%{
    #define SWIG_FILE_WITH_INIT
    #include "src/include/mithral.hpp"
%}

// include other SWIG interface files
%include <config.i>

// tell SWIG to wrap the relevant files
%include "src/include/mithral.hpp"
