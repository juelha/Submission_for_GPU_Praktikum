#ifndef CUDA_STUFF
#define CUDA_STUFF


#include <iostream>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#include <stdio.h>
#include <stdlib.h>

#include "curand_kernel.h"
#include "device_launch_parameters.h"


// debug stuff /////////////////////////////////////////////////////////////

// how to debug:
// compute-sanitizer nvcc main.cu
// compute-sanitizer  --check-api-memory-access 1 nvcc main.cu



// functions
void CHECK_CUDA(cudaError_t err);
void initCUDA();
void runCudaTest();

// kernel stuff /////////
__global__ void firstGen(cudaSurfaceObject_t inputSurfObj,
                         int width, int height);


__device__ bool inBounds(int x, int y);
__device__ int countNeighbors(
        const char* pos,
        cudaSurfaceObject_t inputSurfObj,
        int x, int y,
        int width, int height,
        int layer_in);

__global__ void nextGen(
                        cudaSurfaceObject_t inputSurfObj,
                        cudaSurfaceObject_t outputSurfObj,
                        int width, int height);
__global__ void printGen(
                        cudaSurfaceObject_t inputSurfObj,
                        int width, int height);

// launcher ///
__host__ void launch_firstGen(cudaSurfaceObject_t  outputSurfObj,int width, int height);

__host__ void launch_nextGen(
                cudaSurfaceObject_t inputSurfObj,
              //  cudaSurfaceObject_t outputSurfObj,
              int width, int height,
                int layer_in,
                int layer_out);

__host__ void launch_printGen(cudaSurfaceObject_t  outputSurfObj, int width, int height,int layer);

////////////////////////////////////////////////////////////////////////////////
void CHECK_CUDA(cudaError_t err) {
    if(err != cudaSuccess) {
        printf("%s\n",cudaGetErrorString(err) );
        exit(-1);
    }
}

void initCUDA()
{
    int deviceCount;
    CHECK_CUDA(cudaGetDeviceCount(&deviceCount));

   // cerr << "CUDA device count: " << deviceCount << endl;
    int device = 0; //SELECT GPU HERE
    std::cerr << "Selecting GPU: " << device << std::endl;
    CHECK_CUDA(cudaSetDevice(device));
   // CHECK_CUDA(cudaGLSetGLDevice( device ));
    std::cerr << "CUDA SUCCESSFULLY INITIALIZED" << std::endl;
}



///////////////////////////////////////////////////////////////////////////////
//! KERNEL
///////////////////////////////////////////////////////////////////////////////
__global__ void firstGen(
                       cudaSurfaceObject_t inputSurfObj,
                       int width, int height)
{
    // Calculate surface coordinates
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    //Don't do any computation if this thread is outside of the surface bounds.
    if(x >= width)
        return;
    if(y >= height)
        return;

    bool online;
    // random
    curandState custate;
    curand_init((unsigned long long)clock() + x+y, 0, 0, &custate);
    online = (curand(&custate) % 2);
    float rand1 = curand_uniform(&custate);
    float rand2 = curand_uniform(&custate);
    float rand3 = curand_uniform(&custate);
    // block in middle
//    if(x >= width/2 -10 && x <= width/2 +10 && y >= height/2 -10 && y <= height/2 +10)
//        online = 1;
//    else
//        online = 0;

   // Write to output surface
   float4 element;
   if(online){
        element = make_float4(rand1, rand2, rand3, 1.0f);
        //element = make_float4(1.0f, 1.0f, 1.0f, 1.0f);
   }
   else{
       element = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
   }

   // surf2DLayeredwrite(element,inputSurfObj, x*sizeof(float4),y,0);
    surf2Dwrite(element, inputSurfObj, x*sizeof(float4), y);
}


__global__ void nextGen(
                        cudaSurfaceObject_t inputSurfObj,
                        cudaSurfaceObject_t outputSurfObj,
                        int width, int height)
{

    // Calculate surface coordinates
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    //Don't do any computation if this thread is outside of the surface bounds.
    if(x >= width)
        return;
    if(y >= height)
        return;


    // Read from input surface
    float4 cell;
    //surf2DLayeredread(&cell,inputSurfObj,x*sizeof(float4),y,layer_in);
    surf2Dread(&cell,  inputSurfObj, x*sizeof(float4), y);
    bool state = cell.w==1.0f ? 1 : 0;

    // find species
    // todo make better
    const char* species;
    // red
    if(cell.x >= cell.y && cell.x >= cell.z)
        species = "x";
    // green
    else if(cell.y > cell.x && cell.y > cell.z)
        species = "y";
    // blue
    else
        species = "z";

    //int n_neighbors = 0;
printf("---%s",species);
    int n_neighbors = countNeighbors(species,inputSurfObj,x,y, width,height,0);

    //int n_neighbors = countNeighbors(inputSurfObj,x,y, width,height,0);
    bool new_state =
       n_neighbors == 3 || (n_neighbors == 2 && state) ? 1 : 0;





    // Write to output surface
    float4 element;
    if(new_state){
        // todo make better
        if(species=="x")
            element = make_float4(1.0f, 0.0f, 0.0f, 1.0f);
        else if(species=="y")
            element = make_float4(0.0f, 1.0f, 0.0f, 1.0f);
        else
            element = make_float4(0.0f, 0.0f, 1.0f, 1.0f);

    }
    else{
        element = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }

    surf2Dwrite(element, outputSurfObj,x*sizeof(float4), y);


}



// 8 neighbors are counted
__device__ int countNeighbors(
        const char* species,
        cudaSurfaceObject_t inputSurfObj,
        int x, int y,
        int width, int height,
        int layer_in)
{
    int sum = 0;

    for (int k = -1; k < 2; k++)
    {
        if(x+k >= width || x+k < 0 ) // check border and skip self todo make better
         //  return 0;
           continue;

        for (int l = -1; l < 2; l++)
        {
            if(y+l >= height || y+l < 0) // check border and skip self todo make better
                continue;

            // Read from input surface
            //printf("---%s",pos);
            float4 cell;
            surf2Dread(&cell,  inputSurfObj, (x+k)*sizeof(float4), y+l);

            int neighbor;
            // todo make better
            if(species=="x")
                 neighbor = cell.x>=.33f ? 1 : 0;

            else if(species=="y")
                 neighbor = cell.y>=.33f ? 1 : 0;

            else
                 neighbor = cell.z>=.33f ? 1 : 0;



            sum += neighbor;
        }
    }
    // Read from input surface

    sum -= 1; // substarc self
    return sum;
}




__global__ void printGen(
                        cudaSurfaceObject_t inputSurfObj,
                        int width, int height)
{
    // Calculate surface coordinates
     int x = blockIdx.x * blockDim.x + threadIdx.x;
     int y = blockIdx.y * blockDim.y + threadIdx.y;
    if(x >= width)
        return;
    if(y >= height)
        return;

    // read from surface and write to console
    float4 data;
    surf2Dread(&data,  inputSurfObj, x * sizeof(float4), y);
    printf("-%d", (int)data.w);

}


///////////////////////////////////////////////////////////////////////////////
//! LAUNCHER
///////////////////////////////////////////////////////////////////////////////
__host__ void launch_firstGen(cudaSurfaceObject_t  outputSurfObj, int width, int height)//cudaArray *cuda_image_array )
{


    // to ensure that any error returned by cudaPeekAtLastError() or cudaGetLastError()
    // does not originate from calls prior to the kernel launch, one has to make sure
    // that the runtime error variable is set to cudaSuccess just before the kernel launch,
    // for example, by calling cudaGetLastError() just before the kernel launch
    // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#error-checking

    cudaError_t err;
       err = cudaGetLastError();
    if(err != cudaSuccess) {
        std::cerr << "ERROR: " << cudaGetErrorString(err) << std::endl;
        exit(-1);
    }



    dim3 threadsperBlock(16, 16);
   // dim3 numBlocks((unsigned int)ceil((float)IMAGE_COLS / threadsperBlock.x), IMAGE_ROWS);

   dim3 numBlocks((width + threadsperBlock.x - 1) / threadsperBlock.x,
                  (height + threadsperBlock.y - 1) / threadsperBlock.y);

    firstGen<<<threadsperBlock, numBlocks>>>(outputSurfObj, width, height);

    // Kernel launches do not return any error code,
    // so cudaPeekAtLastError() or cudaGetLastError() must be called
    // just after the kernel launch to retrieve any pre-launch errors.
  //  cudaError_t err;
    err = cudaGetLastError();
    if(err != cudaSuccess) {
        std::cerr << "ERROR: " << cudaGetErrorString(err) << std::endl;
        exit(-1);
    }
}

//extern "C"

__host__ void launch_nextGen(
                            cudaSurfaceObject_t inputSurfObj,
                            cudaSurfaceObject_t outputSurfObj,
                            int width, int height)
{


    // to ensure that any error returned by cudaPeekAtLastError() or cudaGetLastError()
    // does not originate from calls prior to the kernel launch, one has to make sure
    // that the runtime error variable is set to cudaSuccess just before the kernel launch,
    // for example, by calling cudaGetLastError() just before the kernel launch
    // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#error-checking

    cudaError_t err;
       err = cudaGetLastError();
    if(err != cudaSuccess) {
        std::cerr << "ERROR: " << cudaGetErrorString(err) << std::endl;
        exit(-1);
    }


    dim3 threadsperBlock(16, 16);
   // dim3 numBlocks((unsigned int)ceil((float)IMAGE_COLS / threadsperBlock.x), IMAGE_ROWS);

   dim3 numBlocks((width + threadsperBlock.x - 1) / threadsperBlock.x,
                  (height + threadsperBlock.y - 1) / threadsperBlock.y);


    nextGen<<<threadsperBlock, numBlocks>>>(inputSurfObj, outputSurfObj, width, height );

    // Kernel launches do not return any error code,
    // so cudaPeekAtLastError() or cudaGetLastError() must be called
    // just after the kernel launch to retrieve any pre-launch errors.
  //  cudaError_t err;
    err = cudaGetLastError();
    if(err != cudaSuccess) {
        std::cerr << "ERROR: " << cudaGetErrorString(err) << std::endl;
        exit(-1);
    }
}


__host__ void launch_printGen(cudaSurfaceObject_t  outputSurfObj, int width, int height,int layer)//cudaArray *cuda_image_array )
{


    // to ensure that any error returned by cudaPeekAtLastError() or cudaGetLastError()
    // does not originate from calls prior to the kernel launch, one has to make sure
    // that the runtime error variable is set to cudaSuccess just before the kernel launch,
    // for example, by calling cudaGetLastError() just before the kernel launch
    // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#error-checking

    cudaError_t err;
       err = cudaGetLastError();
    if(err != cudaSuccess) {
        std::cerr << "ERROR: " << cudaGetErrorString(err) << std::endl;
        exit(-1);
    }

 printf("\n------layer: %d----------------------------------------------------------------------\n", layer); // 95???



    dim3 threadsperBlock(16, 16);
   // dim3 numBlocks((unsigned int)ceil((float)IMAGE_COLS / threadsperBlock.x), IMAGE_ROWS);

   dim3 numBlocks((width + threadsperBlock.x - 1) / threadsperBlock.x,
                  (height + threadsperBlock.y - 1) / threadsperBlock.y);

    printGen<<<threadsperBlock, numBlocks>>>(outputSurfObj, width, height);

    // Kernel launches do not return any error code,
    // so cudaPeekAtLastError() or cudaGetLastError() must be called
    // just after the kernel launch to retrieve any pre-launch errors.
  //  cudaError_t err;
    err = cudaGetLastError();
    if(err != cudaSuccess) {
        std::cerr << "ERROR: " << cudaGetErrorString(err) << std::endl;
        exit(-1);
    }
}




#endif // KERNEL_CUH
