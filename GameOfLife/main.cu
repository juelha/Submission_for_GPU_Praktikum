// OpenGL Graphics includes
//#include <GL/glew.h>
//#include <GL/glut.h>       // Also included gl.h and glu.h
#include <glad/glad.h>
#include <GLFW/glfw3.h>

// CUDA and interop
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "curand_kernel.h"
#include "device_launch_parameters.h"

// syste
#include <stdio.h>
#include <stdlib.h>

// headers
#include "opengl_stuff.h"
#include "cuda_stuff.cuh"
#include "interop_stuff.cuh"

// resolution
const int height =128;
const int width = 128;


int main(int argc, char** argv)
{
    //////////////////////////////////////////////////////////////////////////////


    initCUDA();

    // rendering ////////////////////////////////////////////777
    begin_rendering();

    GLuint texID[2];
    texID[0] = set_tex_paras(texID[0],width,height);
    texID[1] = set_tex_paras(texID[1],width,height);

    // interop ///////////////////////////////////////////////////////
    cudaSurfaceObject_t surfObj[2];
    surfObj[0] = setUpInterop(surfObj[0],texID[0],width,height,0);
    surfObj[1] = setUpInterop(surfObj[1],texID[1],width,height,1);


    // cuda ////////////////////////////////////////////////////////////////////
    CHECK_CUDA(cudaDeviceSynchronize());
    launch_firstGen(surfObj[0],width,height);

    // render loop //////////////////////////////////////////
    int i = 0;
    while(!glfwWindowShouldClose(getWindow()))
    {
        //Synchronize the CUDA surface with the OpenGL texture,
        //so the texture has the same data as the newly written surface.
        // https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g10e20b05a95f638a4071a655503df25d
        CHECK_CUDA(cudaDeviceSynchronize());
      //  launch_printGen(surfObj[i%2],width,height, i%2);

        rendering_loop(texID[i%2]);

        launch_nextGen(surfObj[i%2],surfObj[(i+1)%2],width,height);
        i++;

   }

     //END ///////////////////////////////////////////////////////////////////////
    end_interop();
    // Destroy surface objects
    cudaDestroySurfaceObject(surfObj[0]);
    cudaDestroySurfaceObject(surfObj[1]);

    //Disable the use of 2D textures.
  //  glDisable(GL_TEXTURE_2D);
    glDeleteTextures(2, texID);
    end_rendering();
   CHECK_CUDA(cudaDeviceSynchronize());
   CHECK_CUDA(cudaDeviceReset());

    return 0;
}






