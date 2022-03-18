#ifndef CUDA_STUFF
#define CUDA_STUFF

// System
#include <iostream>

// Custom
#include "kernel_stuff.cuh"

///
/// \brief     HECK_CUDA
/// \details   Checks a Cuda Function for successful completion
/// \param err CUDA Error type, checking for success is handled internally
///
void CHECK_CUDA(cudaError_t err) {
    if(err != cudaSuccess) {
        printf("%s\n",cudaGetErrorString(err) );
        exit(-1);
    }
}

///
/// \brief     CHECK_KERNEL
/// \details   Checks for CUDA errors without taking a CUDA function as an argument
///            Called before and after the launch of a kernel function
/// \note      Check with cudaGetLastError() before kernel launch for pre-launch errors
///            Check with cudaGetLastError() after kernel launch since kernel launches do not return errors
///            source: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#error-checking
void CHECK_KERNEL(){
    cudaError_t err;
       err = cudaGetLastError();
    if(err != cudaSuccess) {
        std::cerr << "ERROR: " << cudaGetErrorString(err) << std::endl;
        exit(-1);
    }
}

///
/// \brief     initCUDA
/// \details   Make Sure to set right Device for CUDA
///
void initCUDA()
{
    int deviceCount;
    CHECK_CUDA(cudaGetDeviceCount(&deviceCount));

    int device = 0; // manually
    std::cerr << "Selecting GPU: " << device << std::endl;
    CHECK_CUDA(cudaSetDevice(device));
    std::cerr << "CUDA SUCCESSFULLY INITIALIZED" << std::endl;
}


///////////////////////////////////////////////////////////////////////////////
//! LAUNCHER
///////////////////////////////////////////////////////////////////////////////

///
/// \brief     launch_firstGen
/// \details   launches kernel for Computation of the first Generation of Cells
/// \param outputSurfObj    Surface Object to write to
/// \param width
/// \param height
/// \param extra            if extra rules apply (colored cells)
///
__host__ void launch_firstGen(cudaSurfaceObject_t outputSurfObj, int width, int height, bool extra)
{
    CHECK_KERNEL();

    dim3 threadsperBlock(16, 16);
    dim3 numBlocks((width + threadsperBlock.x - 1) / threadsperBlock.x,
                  (height + threadsperBlock.y - 1) / threadsperBlock.y);

    firstGen<<<threadsperBlock, numBlocks>>>(outputSurfObj, width, height, extra);

    CHECK_KERNEL();
}

///
/// \brief   launch_nextGen
/// \details  launches kernel for Computation of the next Generation of Cells
/// \param inputSurfObj
/// \param outputSurfObj
/// \param width
/// \param height
/// \param extra        if extra rules apply (colored cells)
///
__host__ void launch_nextGen(cudaSurfaceObject_t inputSurfObj, cudaSurfaceObject_t outputSurfObj, int width, int height, bool extra)
{
    CHECK_KERNEL();

    dim3 threadsperBlock(16, 16);
    dim3 numBlocks((width + threadsperBlock.x - 1) / threadsperBlock.x,
                  (height + threadsperBlock.y - 1) / threadsperBlock.y);

    nextGen<<<threadsperBlock, numBlocks>>>(inputSurfObj, outputSurfObj, width, height , extra);

    CHECK_KERNEL();
}

///
/// \brief launch_printGen
/// \details launches kernel that prints the cells of a given generation (current layer)
///          used for Debug-Purposes
/// \param outputSurfObj
/// \param width
/// \param height
/// \param layer
///
__host__ void launch_printGen(cudaSurfaceObject_t outputSurfObj, int width, int height, int layer)
{

    CHECK_KERNEL();

    printf("\n------layer: %d----------------------------------------------------------------------\n", layer);

    dim3 threadsperBlock(16, 16);
    dim3 numBlocks((width + threadsperBlock.x - 1) / threadsperBlock.x,
                  (height + threadsperBlock.y - 1) / threadsperBlock.y);

    printGen<<<threadsperBlock, numBlocks>>>(outputSurfObj, width, height);

    CHECK_KERNEL();
}




#endif // KERNEL_CUH
