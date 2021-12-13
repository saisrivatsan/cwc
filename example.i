%module example

// include files that SWIG will need to see
%{
    #define SWIG_FILE_WITH_INIT
    #include <vector>
    #include "src/include/public_interface.hpp"
    #include "src/include/public_interface_eigen.hpp"
    #include "src/include/mithral.hpp"
%}

// include other SWIG interface files
%include <config.i>

// tell SWIG to wrap the relevant files
%include "src/include/public_interface.hpp"
%include "src/include/public_interface_eigen.hpp"
%include "src/include/mithral.hpp"
