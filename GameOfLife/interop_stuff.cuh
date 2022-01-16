#ifndef INTEROP_STUFF_CUH
#define INTEROP_STUFF_CUH


// CUDA and interop
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "curand_kernel.h"
#include "device_launch_parameters.h"
#include <cudaGL.h>

#include "cuda_stuff.cuh"

/*

    1. cuda array

    2. link with graphics resource
16
    3. surface

*/


cudaGraphicsResource *cuda_graphics_resource[2];
cudaArray            *cuda_array[2];

//cudaGraphicsResource_t cuda_graphics_resource;
////cudaArray            *cuda_array;
//cudaArray_t cuda_array;
////CUarray m_cudaArray[2];
////CUgraphicsResource m_cuda_graphicsResource[2];

//cudaArray *m_cudaArray[2];
//cudaArray *m_cudaArray2;

//cudaGraphicsResource_t m_cuda_graphicsResource[2];

///////////////
/// \brief setUpInterop
/// \param texID
/// \param width
/// \param height
/// \return
///
cudaSurfaceObject_t setUpInterop(cudaSurfaceObject_t surf,GLuint texID,int width,int height,int layer){


       // Get cudaArray pointer from the resource
       // Register both volume textures (pingponged) in CUDA
            cudaGraphicsGLRegisterImage( &cuda_graphics_resource[layer]
                                     , texID
                                     , GL_TEXTURE_2D
                                     , CU_GRAPHICS_REGISTER_FLAGS_SURFACE_LDST );

            cudaGraphicsMapResources(1, &cuda_graphics_resource[layer], 0 );

              // Bind the volume textures to their respective cuda arrays.
              cudaGraphicsSubResourceGetMappedArray( &cuda_array[layer]
                                                 , cuda_graphics_resource[layer]
                                                 , 0, 0 );

//            cudaGraphicsUnmapResources(1, &m_cuda_graphicsResource[0], 0 );


       // surface ////////////////////////////////////////////////////////////////////
       // cudaBindSurfaceToArray is deprecitaed -> manually
       //Documentation for the struct: https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaResourceDesc.html#structcudaResourceDesc

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




//    // register Image (texture) to CUDA Resource
//    CHECK_CUDA(cudaGraphicsGLRegisterImage(&cuda_graphics_resource,
//                                           texID, GL_TEXTURE_2D,
//                                           cudaGraphicsRegisterFlagsSurfaceLoadStore));

//    // map CUDA resource
//    CHECK_CUDA(cudaGraphicsMapResources(1, &cuda_graphics_resource, 0));

//    //Get mapped array
//        CHECK_CUDA(cudaGraphicsSubResourceGetMappedArray(&cuda_array, cuda_graphics_resource, 0, 0));


////        // DO NOT FUCKING TOUCH BEGIN ////////////////////////////////////////////
////        // Allocate CUDA arrays in device memory
////        cudaChannelFormatDesc channelDesc =
////           cudaCreateChannelDesc(height, width, 0, 0, cudaChannelFormatKindUnsigned);
////        // extent of the2X2X2 float cube
////        //      Width in elements when referring to array memory, in bytes when referring to linear memory
////      //  cudaExtent extent = make_cudaExtent(width*4, height, 2);
////        cudaExtent extent = make_cudaExtent(width* sizeof(float4), height, 2);

////        //allocate memory on the layered array.
////        CHECK_CUDA(cudaMalloc3DArray(&cuda_array,
////                                       &channelDesc,
////                                       extent,
////                                       //cudaArrayLayered
////                                     cudaArraySurfaceLoadStore
////                                     ));
////        // DO NOT FUCKING TOUCH END ////////////////////////////////////////////





//        // Specify surface
//        struct cudaResourceDesc resDesc;
//        memset(&resDesc, 0, sizeof(resDesc));
//        resDesc.resType = cudaResourceTypeArray;

//        // Create the surface objects
//        // out -> cuda_image_array aka the one registered with gl
//        resDesc.res.array.array = cuda_array;
//        cudaSurfaceObject_t surf =0;
//        cudaCreateSurfaceObject(&surf, &resDesc);




//    // register Image (texture) to CUDA Resource
//    CHECK_CUDA(cudaGraphicsGLRegisterImage(&cuda_graphics_resource,
//                                           texID, GL_TEXTURE_2D,
//                                           cudaGraphicsRegisterFlagsSurfaceLoadStore));
//    // map CUDA resource
//    CHECK_CUDA(cudaGraphicsMapResources(1, &cuda_graphics_resource, 0));

//    //Get mapped array
//    CHECK_CUDA(cudaGraphicsSubResourceGetMappedArray(&cuda_array, cuda_graphics_resource, 0, 0));


//    // DO NOT FUCKING TOUCH BEGIN ////////////////////////////////////////////
//    // Allocate CUDA arrays in device memory
//    cudaChannelFormatDesc channelDesc =
//       cudaCreateChannelDesc(height, width, 0, 0, cudaChannelFormatKindUnsigned);
//    // extent of the2X2X2 float cube
//    //      Width in elements when referring to array memory, in bytes when referring to linear memory
//    cudaExtent extent = make_cudaExtent(width*4, height, 2);
//    //cudaExtent extent = make_cudaExtent(width * sizeof(int), height, 2);

//    //allocate memory on the layered array.
//    CHECK_CUDA(cudaMalloc3DArray(&cuda_array,
//                                   &channelDesc,
//                                   extent,
//                                   cudaArrayLayered
//                               //  cudaArraySurfaceLoadStore
//                                 ));
//    // DO NOT FUCKING TOUCH END ////////////////////////////////////////////

//    // Specify surface
//    struct cudaResourceDesc resDesc;
//    memset(&resDesc, 0, sizeof(resDesc));
//    resDesc.resType = cudaResourceTypeArray;

//    // Create the surface objects
//    // out -> cuda_image_array aka the one registered with gl
//    resDesc.res.array.array = cuda_array;
//    cudaSurfaceObject_t surfObj =0;
//    cudaCreateSurfaceObject(&surfObj, &resDesc);





//    // interop ////////////////////////////////////////////////////7
//    // allocate mem for input arr and link with graphics resource




//    // link with graphics resource ///////////////////////////////////////

//    // register gl image (texture) to CUDA Resource (cudaGraphicsResource)
//    //A new cudaGraphicsResource is created, and its address is placed in cudaTextureID.
//    CHECK_CUDA(cudaGraphicsGLRegisterImage(&cuda_graphics_resource,
//                                           texID[0],
//                                           GL_TEXTURE_2D,
//                                           cudaGraphicsRegisterFlagsSurfaceLoadStore)); // register w this flag to able able to write to it

//    /*
//    CHECK_CUDA(cudaGraphicsGLRegisterImage(&cuda_graphics_resource,
//                                           texID[1],
//                                           GL_TEXTURE_2D,
//                                           cudaGraphicsRegisterFlagsSurfaceLoadStore)); // register w this flag to able able to write to i
//*/

//    // map CUDA
//    CHECK_CUDA(cudaGraphicsMapResources(1, &cuda_graphics_resource, 0));

//    CHECK_CUDA(cudaGraphicsSubResourceGetMappedArray(&cuda_array, cuda_graphics_resource, 0, 0));

//    /*

//    */

//    // array /////////////////////////

//    // DO NOT FUCKING TOUCH BEGIN ////////////////////////////////////////////
//    // Allocate CUDA arrays in device memory
//    cudaChannelFormatDesc channelDesc =
//       cudaCreateChannelDesc(height, width, 0, 0, cudaChannelFormatKindUnsigned);
//    // extent of the2X2X2 float cube
//    cudaExtent extent = make_cudaExtent(width * sizeof(int), height, 2);
//    //allocate memory on the layered array.
//    CHECK_CUDA(cudaMalloc3DArray(&cuda_array,
//                                   &channelDesc,
//                                   extent,
//                                   cudaArrayLayered));
//    // DO NOT FUCKING TOUCH END ////////////////////////////////////////////

//

//    CHECK_CUDA(cudaDeviceSynchronize());
       return surf;

}










void end_interop(){




    CHECK_CUDA(cudaGraphicsUnmapResources(1, &cuda_graphics_resource[0], 0));
    CHECK_CUDA(cudaGraphicsUnmapResources(1, &cuda_graphics_resource[1], 0));


/*

    // unmap
    CHECK_CUDA(cudaGraphicsUnmapResources(2, cuda_graphics_resource, 0));

    // unregister
    CHECK_CUDA(cudaGraphicsUnregisterResource(cuda_graphics_resource[0]));
    CHECK_CUDA(cudaGraphicsUnregisterResource(cuda_graphics_resource[1]));
    glDeleteTextures(1, &texID);

    // Free device memory
    cudaFreeArray(cuda_array);


*/

}














#endif // INTEROP_STUFF_CUH
