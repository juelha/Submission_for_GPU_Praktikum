#ifndef INTEROP_STUFF_CUH
#define INTEROP_STUFF_CUH


// CUDA and interop
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "curand_kernel.h"
#include "device_launch_parameters.h"
#include <cudaGL.h>

#include "cuda_stuff.cuh"


cudaGraphicsResource *cuda_graphics_resource[2];
cudaArray            *cuda_array[2];


/// \brief setUpInterop
/// \param surf
/// \param texID
/// \param width
/// \param height
/// \param layer
/// \return
///
cudaSurfaceObject_t setUpInterop(cudaSurfaceObject_t surf,GLuint texID,int width,int height,int layer)
{


    // register Image (texture) to CUDA Resource
    // register gl image (texture) to CUDA Resource (cudaGraphicsResource)
    // A new cudaGraphicsResource is created, and its address is placed in cudaTextureID.
    cudaGraphicsGLRegisterImage( &cuda_graphics_resource[layer],
                                 texID,
                                 GL_TEXTURE_2D,
                                 CU_GRAPHICS_REGISTER_FLAGS_SURFACE_LDST);
    // map CUDA resource
    cudaGraphicsMapResources(1, &cuda_graphics_resource[layer], 0 );

    // Bind the textures to their respective cuda arrays
    cudaGraphicsSubResourceGetMappedArray( &cuda_array[layer]
                                     , cuda_graphics_resource[layer]
                                     , 0, 0 );

    //  cudaGraphicsUnmapResources(1, &m_cuda_graphicsResource[0], 0 );


    // surface ////////////////////////////////////////////////////////////////////
    // cudaBindSurfaceToArray is deprecitaed -> manually

    // Specify surface
    //Create a CUDA resource descriptor. This is used to get and set attributes of CUDA resources.
    struct cudaResourceDesc resDesc;
    //Clear it with 0s so that some flags aren't arbitrarily left at 1s
    memset(&resDesc, 0, sizeof(resDesc));
    //Set the resource type to be an array for convenient processing in the CUDA kernel.
    resDesc.resType = cudaResourceTypeArray;

    // Create the surface objects
    //Bind the new descriptor with the cuda array created earlier
    resDesc.res.array.array = cuda_array[layer];
    //Create a new CUDA surface ID reference.
    //This is really just an unsigned long long.
    surf = 0;
    //Create the surface with the given description.
    CHECK_CUDA(cudaCreateSurfaceObject(&surf, &resDesc));



//        // DO NOT FUCKING TOUCH BEGIN ////////////////////////////////////////////
//        // Allocate CUDA arrays in device memory
//        cudaChannelFormatDesc channelDesc =
//           cudaCreateChannelDesc(height, width, 0, 0, cudaChannelFormatKindUnsigned);
//        // extent of the2X2X2 float cube
//        //      Width in elements when referring to array memory, in bytes when referring to linear memory
//      //  cudaExtent extent = make_cudaExtent(width*4, height, 2);
//        cudaExtent extent = make_cudaExtent(width* sizeof(float4), height, 2);

//        //allocate memory on the layered array.
//        CHECK_CUDA(cudaMalloc3DArray(&cuda_array,
//                                       &channelDesc,
//                                       extent,
//                                       //cudaArrayLayered
//                                     cudaArraySurfaceLoadStore
//                                     ));
//        // DO NOT FUCKING TOUCH END ////////////////////////////////////////////


       return surf;

}





void end_interop(){

    // unmap
    CHECK_CUDA(cudaGraphicsUnmapResources(1, &cuda_graphics_resource[0], 0));
    CHECK_CUDA(cudaGraphicsUnmapResources(1, &cuda_graphics_resource[1], 0));

    // unregister
    CHECK_CUDA(cudaGraphicsUnregisterResource(cuda_graphics_resource[0]));
    CHECK_CUDA(cudaGraphicsUnregisterResource(cuda_graphics_resource[1]));

    // Free device memory
    cudaFreeArray(cuda_array[0]);
    cudaFreeArray(cuda_array[1]);

}





#endif // INTEROP_STUFF_CUH
