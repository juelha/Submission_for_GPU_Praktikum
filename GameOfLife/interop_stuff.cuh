#ifndef INTEROP_STUFF_CUH
#define INTEROP_STUFF_CUH

// CUDA and Interop
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h>
#include <cudaGL.h>


// Vars needed during MainLoop
cudaGraphicsResource *cuda_graphics_resource[2];
cudaArray            *cuda_array[2];

///
/// \brief   setUpInterop
/// \details Make use of the interopabilty of CUDA and OpenGL and bind a
///          OpenGL texture to a CUDA Surface Object
/// \param surf
/// \param texID
/// \param width
/// \param height
/// \param layer
/// \return Cuda Surface Object which is mapped to the OpenGL texture
///
cudaSurfaceObject_t setUpInterop(cudaSurfaceObject_t surf, GLuint texID, int width, int height, int layer)
{
    // CUDA Graphics Resource ////////////////////////////////////////////////////////////////////
    // Register gl image (texture) to CUDA resource (cudaGraphicsResource)
    cudaGraphicsGLRegisterImage(&cuda_graphics_resource[layer],
                                 texID,
                                 GL_TEXTURE_2D,
                                 CU_GRAPHICS_REGISTER_FLAGS_SURFACE_LDST);
    // Map CUDA resource
    cudaGraphicsMapResources(1, &cuda_graphics_resource[layer], 0 );

    // Bind the texture through the graphics resource with its respective cuda array
    cudaGraphicsSubResourceGetMappedArray(&cuda_array[layer],
                                          cuda_graphics_resource[layer],
                                          0, 0 );

    // CUDA Surface Object ////////////////////////////////////////////////////////////////////
    // cudaBindSurfaceToArray is deprecitaed -> do it manually

    // Specify surface
    // Need CUDA resource descriptor to create surface object with "cudaCreateSurfaceObject()"
    struct cudaResourceDesc resDesc;
    // Init with Zeros
    memset(&resDesc, 0, sizeof(resDesc));
    // Set Type to Array
    resDesc.resType = cudaResourceTypeArray;

    // Create the surface object
    // Bind descriptor with the cuda array that is the subresource of the graphics resource
    resDesc.res.array.array = cuda_array[layer];
    // Init Surface ID
    surf = 0;
    // Create the surface with descriptor
    CHECK_CUDA(cudaCreateSurfaceObject(&surf, &resDesc));

    return surf;
}

///
/// \brief end_interop
/// \details cleans up vars
void end_interop()
{
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
