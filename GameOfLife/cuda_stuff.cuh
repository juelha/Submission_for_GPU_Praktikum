#ifndef CUDA_STUFF
#define CUDA_STUFF


#include <iostream>

#include <stdio.h>
#include <stdlib.h>

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "curand_kernel.h"
#include "device_launch_parameters.h"


// debug stuff /////////////////////////////////////////////////////////////

// how to debug:
// compute-sanitizer nvcc main.cu
// compute-sanitizer  --check-api-memory-access 1 nvcc main.cu




////////////////////////////////////////////////////////////////////////////////
void CHECK_CUDA(cudaError_t err) {
    if(err != cudaSuccess) {
        printf("%s\n",cudaGetErrorString(err) );
        exit(-1);
    }
}

int CHECK_KERNEL(){
    // to ensure that any error returned by cudaPeekAtLastError() or cudaGetLastError()
    // does not originate from calls prior to the kernel launch, one has to make sure
    // that the runtime error variable is set to cudaSuccess just before the kernel launch,
    // for example, by calling cudaGetLastError() just before the kernel launch
    // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#error-checking

    cudaError_t err;
       err = cudaGetLastError();
    if(err != cudaSuccess) {
        std::cerr << "ERROR: " << cudaGetErrorString(err) << std::endl;
       // exit(-1);

        return -1;
    }
    // Kernel launches do not return any error code,
    // so cudaPeekAtLastError() or cudaGetLastError() must be called
    // just after the kernel launch to retrieve any pre-launch errors.
    return 0;
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
///

__global__ void firstGen(cudaSurfaceObject_t inputSurfObj, int width, int height)
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

///
/// \brief   nextGen
/// \details
/// \param   inputSurfObj
/// \param   outputSurfObj
/// \param   width
/// \param   height
///
__global__ void nextGen(cudaSurfaceObject_t inputSurfObj, cudaSurfaceObject_t outputSurfObj, int width, int height)
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



    float n_neighbors = 0;
    float n_neighbors_green = 0;

    float n_neighbors_blue = 0;

    bool kill = 0;

    for (int k = -1; k < 2; k++)
    {
        if(x+k >= width || x+k < 0 )
           continue;
        for (int l = -1; l < 2; l++)
        {
            if(y+l >= height || y+l < 0)
                continue;
            if(k==0 && l==0) // skip self todo make better
                continue;

            // Read from input surface
            float4 neighbor;
            surf2Dread(&neighbor, inputSurfObj, (x+k)*sizeof(float4), y+l);

            // todo make better


              if(cell.x > cell.y && cell.x > cell.z)
          //   if(cell.x > .50f)

             {
                  //n_neighbors += neighbor.x  >  .33f  ? .33f : 0;
                 n_neighbors += neighbor.x;//  >  .50f  ? .33f : 0;

                 // printf("---%f",neighbor.y);


             }
             else if(cell.y > cell.x && cell.y > cell.z)
         //  else if(cell.y > .50f)

           {
                n_neighbors += neighbor.y;// > .50f ? .33f : 0;
                kill = (neighbor.z >= neighbor.x && neighbor.z >= neighbor.y)  ? 1 : 0; // red kills green
                if(kill){
                    float4 rip = make_float4(0.0f, 1.0f, 0.0f, 1.0f);
                    surf2Dwrite(rip, inputSurfObj, (x+k)*sizeof(float4), y+l);
                    surf2Dwrite(rip, outputSurfObj, (x+k)*sizeof(float4), y+l);
                }

           }

            else if(cell.z > cell.x && cell.z > cell.y)
           //else if(cell.z > .50f)

            {
                 n_neighbors += neighbor.z;// > .50f ? .33f : 0;
                 kill = (neighbor.x >= neighbor.y && neighbor.x >= neighbor.z) ? 1 : 0; // red kills green
                 if(kill){
                     float4 rip = make_float4(0.0f, 0.0f, 1.0f, 1.0f);
                     surf2Dwrite(rip, inputSurfObj, (x+k)*sizeof(float4), y+l);
                     surf2Dwrite(rip, outputSurfObj, (x+k)*sizeof(float4), y+l);
                 }
            }



            // not friendly
            if(kill){
                // KILL
              //  printf("%s","tru");

                continue;
            }
        }
    }



    ///////////////////////////////////////////////////////////////////////////
    //int n_neighbors = countNeighbors(inputSurfObj,x,y, width,height,0);
    bool new_state =
     //  n_neighbors == 3 || (n_neighbors == 2 && state) || kill ? 1 : 0;

      (   n_neighbors <= 3.4f && n_neighbors >= 2.5f) || (   n_neighbors <= 2.4f && n_neighbors >= 1.5f && state)  ? 1 : 0;
    //printf("%f\n",n_neighbors);


    // Write to output surface
    float4 element;
    // random
    bool mutate;
    curandState custate;
    curand_init((unsigned long long)clock() + x+y, 0, 0, &custate);
    mutate = (curand(&custate) % 2);

    float rand1 = curand_uniform(&custate);
    float rand2 = curand_uniform(&custate);

    if(new_state){
        // todo make better
        if(cell.x > cell.y && cell.x > cell.z)
            element = make_float4(1.0f, 0, 0, 1.0f);

        else if(cell.y > cell.x && cell.y > cell.z)
            element = make_float4(0, 1.0f, 0, 1.0f);

        else if(cell.z > cell.x && cell.z > cell.y)
            element = make_float4(0, 0, 1.0f, 1.0f);
//        if(cell.x > .50f)
//            element = make_float4(1.0f, rand1*mutate, rand2*mutate, 1.0f);
//        else if(cell.y >.50f)
//            element = make_float4(rand1*mutate, 1.0f, rand2*mutate, 1.0f);
//        else if(cell.z >.50f)
//            element = make_float4(rand1*mutate, rand2*mutate, 1.0f, 1.0f);


    }
    else{
        element = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }

    surf2Dwrite(element, outputSurfObj,x*sizeof(float4), y);


}

__device__ void eat(
        cudaSurfaceObject_t &inputSurfObj,
        cudaSurfaceObject_t &outputSurfObj,
        int x, int y)
{
    float4 rip = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    surf2Dwrite(rip, inputSurfObj,x*sizeof(float4), y);

}




__global__ void printGen(cudaSurfaceObject_t inputSurfObj, int width, int height)
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
__host__ void launch_firstGen(cudaSurfaceObject_t outputSurfObj, int width, int height)//cudaArray *cuda_image_array )
{
    CHECK_KERNEL();

    dim3 threadsperBlock(16, 16);
    // dim3 numBlocks((unsigned int)ceil((float)IMAGE_COLS / threadsperBlock.x), IMAGE_ROWS);
    dim3 numBlocks((width + threadsperBlock.x - 1) / threadsperBlock.x,
                  (height + threadsperBlock.y - 1) / threadsperBlock.y);

    firstGen<<<threadsperBlock, numBlocks>>>(outputSurfObj, width, height);

    CHECK_KERNEL();
}


__host__ void launch_nextGen(cudaSurfaceObject_t inputSurfObj, cudaSurfaceObject_t outputSurfObj, int width, int height)
{
    CHECK_KERNEL();

    dim3 threadsperBlock(16, 16);
    // dim3 numBlocks((unsigned int)ceil((float)IMAGE_COLS / threadsperBlock.x), IMAGE_ROWS);
    dim3 numBlocks((width + threadsperBlock.x - 1) / threadsperBlock.x,
                  (height + threadsperBlock.y - 1) / threadsperBlock.y);

    nextGen<<<threadsperBlock, numBlocks>>>(inputSurfObj, outputSurfObj, width, height );

    CHECK_KERNEL();
}


__host__ void launch_printGen(cudaSurfaceObject_t  outputSurfObj, int width, int height,int layer)//cudaArray *cuda_image_array )
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
