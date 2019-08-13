% compile on mac OS:

1. install XCode
2. install eigen: brew install eigen (important: at the end of the installation keep a note of where it installed it)
3. run on matlab:
mex -I"/path/to/eigen/header-files/folder" /path/to/cpp/files/computeBlaBla.cpp

E.g.:
mex -I"/usr/local/Cellar/eigen/3.3.5/include/eigen3" /Users/MyUser/surface_networks_covers/matlab/flattening/mex/computeMeshTranformationCoeffsMex.cpp