// headers
#include "opengl_stuff.h"
#include "cuda_stuff.cuh"
#include "interop_stuff.cuh"
#include "kernel_stuff.cuh"

// resolution
const int height =128;
const int width = 128;


int main()
{
    // Set up OpenGL ////////////////////////////////////////
    setUp_rendering();
    GLuint texID[2];
    texID[0] = config_tex(texID[0],width,height);
    texID[1] = config_tex(texID[1],width,height);

    // Set up Cuda ///////////////////////////////////////////
    initCUDA();
    cudaSurfaceObject_t surfObj[2];
    surfObj[0] = setUpInterop(surfObj[0],texID[0],width,height,0);
    surfObj[1] = setUpInterop(surfObj[1],texID[1],width,height,1);

    // Set up first Generation /////////////////////////////////////////
    CHECK_CUDA(cudaDeviceSynchronize());
    bool extra = 1;
    launch_firstGen(surfObj[0],width,height,extra);

    // RENDERING LOOP ////////////////////////////////////////////////
    int i = 0;
    while(!glfwWindowShouldClose(getWindow()))
    {
        //Synchronize the CUDA surface with the OpenGL texture,
        //so the texture has the same data as the newly written surface.
        // https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g10e20b05a95f638a4071a655503df25d
        CHECK_CUDA(cudaDeviceSynchronize());
        launch_printGen(surfObj[i%2],width,height, i%2);

        rendering(texID[i%2]);

        launch_nextGen(surfObj[i%2], surfObj[(i+1)%2], width, height, extra);
        i++;
    }

     //END ///////////////////////////////////////////////////////////////////////
    end_interop();
    // Destroy surface objects
    cudaDestroySurfaceObject(surfObj[0]);
    cudaDestroySurfaceObject(surfObj[1]);
    glDeleteTextures(2, texID);
    end_rendering();
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaDeviceReset());

    return 0;
}






