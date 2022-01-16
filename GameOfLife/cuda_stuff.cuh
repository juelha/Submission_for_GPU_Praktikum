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

void CHECK_CUDA(cudaError_t err) {
    if(err != cudaSuccess) {
        printf("%s",cudaGetErrorString(err) );

       // std::cerr << "Error: " << cudaGetErrorString(err) << std::endl;
       // fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n",
        //     cudaGetErrorString(err),
          //  __FILE__, __LINE__);
       //   fprint(stderr,"h",  __FILE__, __LINE__);

      //  fprintf(stderr, "CUDA CHECK FAILED "  "%s:%d Returned:%d\n", __FILE__, __LINE__, err);

        printf("%s","\n");
        exit(-1);
    }
}



//Global scope surface to bind to
void initCUDA();

// how to debug:
// compute-sanitizer nvcc main.cu
// compute-sanitizer  --check-api-memory-access 1 nvcc main.cu

void runCudaTest();


void CHECK_CUDA(cudaError_t err);

// kernel stuff /////////
__global__ void firstGen(
                       cudaSurfaceObject_t inputSurfObj,
                       int width, int height);

__device__ bool inBounds(int x, int y);
__device__ int countNeighbors(
        cudaSurfaceObject_t inputSurfObj,
        int x, int y,
        int width, int height,
        int layer_in);

__global__ void nextGen(
                        cudaSurfaceObject_t inputSurfObj,
                        cudaSurfaceObject_t outputSurfObj,
                        int width, int height
/*                        int layer_in,
                        int layer_out*/);

// launcher ///
__host__ void launch_firstGen(cudaSurfaceObject_t  outputSurfObj,int width, int height);

__host__ void launch_nextGen(
                cudaSurfaceObject_t inputSurfObj,
              //  cudaSurfaceObject_t outputSurfObj,
              int width, int height,
                int layer_in,
                int layer_out);

////////////////////////////////////////////////////////////////////////////////
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

// launcher ////////////////////////////////////////////////////////////////



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

             //   int layer_in,
              //  int layer_out)
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


// kernel //////////////////////////////////////////////////////////////////

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
    // block in middle
    if(x >= width/2 -10 && x <= width/2 +10 && y >= height/2 -10 && y <= height/2 +10)
        online = 1;
    else
        online = 0;

   online = (curand(&custate) % 2);

    float4 element;
    if(online){
         element = make_float4(255.0f, 255.0f, 255.0f, 255.0f);
    }
    else{
         element = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
   // surf2DLayeredwrite(element,inputSurfObj, x*sizeof(float4),y,0);
    surf2Dwrite(element, inputSurfObj, x*sizeof(float4), y);


  //unsigned int element = (curand(&state) % 2)*255;
   //


  //  surf2Dwrite(element, inputSurfObj, x*sizeof(float4), y);
    // Write to output surface
  //  surf2DLayeredwrite(element,inputSurfObj, x*sizeof(unsigned int),y,0);

 //  surf2Dwrite(element,inputSurfObj, x*sizeof(int),y);

}

///////////////////////////////////////////////////////////////////////////////
//! Simple kernel to modify vertex positions in sine wave pattern
//! @param data  data in global memory
///////////////////////////////////////////////////////////////////////////////
// instead of using internal variables to figure out location in array we use pitch
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
    bool state = cell.x==255.0f ? 1 : 0;
    int n_neighbors = countNeighbors( inputSurfObj,x,y, width,height,0);
    bool new_state =
       n_neighbors == 3 || (n_neighbors == 2 && state) ? 1 : 0;

    curandState custate;
    curand_init((unsigned long long)clock() + x, 0, 0, &custate);
    bool online = (curand(&custate) % 2);

    // Write to output surface
    float4 element;
    if(new_state){
         element = make_float4(255.0f, 255.0f, 255.0f, 255.0f);
    }
    else{
        element = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
 //   surf2DLayeredwrite(element,inputSurfObj, x*sizeof(float4),y,layer_out);
   printf("____%d____",new_state);

    surf2Dwrite(element, outputSurfObj,x*sizeof(float4), y);


}



// 8 neighbors are counted
__device__ int countNeighbors(
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
            float4 cell;
            surf2Dread(&cell,  inputSurfObj, (x+k)*sizeof(float4), y+l);
            int neighbor = cell.x==255.0f ? 1 : 0;
            sum += neighbor;
        }
    }
    // Read from input surface
    float4 cell;
    surf2Dread(&cell,  inputSurfObj, x*sizeof(float4), y);
    int neighbor = cell.x==255.0f ? 1 : 0;
    sum -= neighbor;
    return sum;
}









__global__ void check_kernel(
                        cudaSurfaceObject_t inputSurfObj,
                        int width, int height,
        int layer)
{
    // Calculate surface coordinates
     int x = blockIdx.x * blockDim.x + threadIdx.x;
     int y = blockIdx.y * blockDim.y + threadIdx.y;
    if(x >= width)
        return;
    if(y >= height)
        return;


        //uchar4 data;-
        float4 data;

    //    surf2DLayeredread(&data,inputSurfObj,(x)*sizeof(int),y,layer);
          surf2Dread(&data,  inputSurfObj, x * sizeof(float4), y);



    printf("-%d", (int)data.x);



}


__host__ void launch_checkKernel(cudaSurfaceObject_t  outputSurfObj, int width, int height,int layer)//cudaArray *cuda_image_array )
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

    check_kernel<<<threadsperBlock, numBlocks>>>(outputSurfObj, width, height,layer);

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


void launch_init(){

}





#endif // KERNEL_CUH
