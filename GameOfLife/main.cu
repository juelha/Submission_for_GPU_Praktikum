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
// constants
//const unsigned int window_width  = 512;
//const unsigned int window_height = 512;


int main(int argc, char** argv)
{
    //////////////////////////////////////////////////////////////////////////////


    initCUDA();

    // rendering ////////////////////////////////////////////777
    begin_rendering();


    GLuint texID[2];
    glGenTextures(1, &texID[0]);
    glGenTextures(1, &texID[1]);
    texID[0] = set_tex_paras(texID[0],width,height);
    texID[1] = set_tex_paras(texID[1],width,height);
    // check for errors
    CHECK_ERROR_GL();

    // interop ///////////////////////////////////////////////////////
    /// \brief surfObj
    cudaSurfaceObject_t surfObj[2];
     surfObj[0] = setUpInterop(surfObj[0],texID[0],width,height,0);
     surfObj[1] = setUpInterop(surfObj[1],texID[1],width,height,1);


    // cuda ////////////////////////////////////////////////////////////////////

    CHECK_CUDA(cudaDeviceSynchronize());
    launch_firstGen(surfObj[0],width,height);
  //  launch_checkKernel(surfObj[0],width,height, 0);

   CHECK_CUDA(cudaDeviceSynchronize());

//   for(int i=0;i<5;i++){
//       launch_checkKernel(surfObj[i%2],width,height, 0);

//       launch_nextGen(surfObj[i%2],surfObj[(i)%2],width,height);
//       CHECK_CUDA(cudaDeviceSynchronize());

//   }


//   end_interop();
//   checkTex(&texID[0],width,height);

    // render loop //////////////////////////////////////////
    int i = 0;
    while(!glfwWindowShouldClose(getWindow()))
    {
        //Synchronize the CUDA surface with the OpenGL texture, so the texture has the same data as the newly written surface.
        // https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g10e20b05a95f638a4071a655503df25d
        CHECK_CUDA(cudaDeviceSynchronize());
        launch_checkKernel(surfObj[i%2],width,height, i%2);

        rendering_loop(texID[i%2]); // oi sth is fuc ked here
      //  rendering_loop(texID[1]); // oi sth is fuc ked here

       launch_nextGen(surfObj[i%2],surfObj[(i+1)%2],width,height);
        i++;

   }


     //END ///////////////////////////////////////////////////////////////////////

//     Destroy surface objects
//    cudaDestroySurfaceObject(surfObj);
//    end_interop();






/*
 *
 *
 *



    // cuda ////////////////////////////////////////////////////////////////////
    initCUDA();
    cudaSurfaceObject_t surfObj = setUpInterop(texID,width,height);

    CHECK_CUDA(cudaDeviceSynchronize());

    launch_firstGen(surfObj,width,height);


    launch_checkKernel(surfObj,width,height,0);

    CHECK_CUDA(cudaDeviceSynchronize());
    rendering_loop(texID[0]);





    // END ///////////////////////////////////////////////////////////////////////

    CHECK_CUDA(cudaDeviceSynchronize());





    end_interop();


    // unbind tex

    // Destroy surface objects
    cudaDestroySurfaceObject(surfObj);
    //Disable the use of 2D textures.
    glDisable(GL_TEXTURE_2D);

    glDeleteTextures(2, texID);


    CHECK_CUDA(cudaDeviceReset());

    //rendering_loop();
    end_rendering();




        // render loop //////////////////////////////////////////
for(int i=0;i<1;i++)
        {

            //Synchronize the CUDA surface with the OpenGL texture, so the texture has the same data as the newly written surface.
            // https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g10e20b05a95f638a4071a655503df25d
            //cudaDeviceSynchronize();


            cudaDeviceSynchronize();
             rendering_loop(texID[i%2]);

           // rendering_loop(texID[i%2]);
           launch_nextGen(surfObj,width,height, i%2,(i+1)%2);
            launch_checkKernel(surfObj,width,height,(i+1)%2);
            cudaDeviceSynchronize();

            // rendering_loop(texID[i%2]);
          //  launch_nextGen(surfObj,width,height, (i+1)%2,i%2);
          //   launch_checkKernel(surfObj,width,height,i%2);



        }





        // render loop //////////////////////////////////////////
        int i = 0;
        while(!glfwWindowShouldClose(getWindow()))
        {
            launch_checkKernel(surfObj,width,height,(i+1)%2);

            //Synchronize the CUDA surface with the OpenGL texture, so the texture has the same data as the newly written surface.
            // https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html#group__CUDART__DEVICE_1g10e20b05a95f638a4071a655503df25d
            CHECK_CUDA(cudaDeviceSynchronize());
            //cudaDeviceSynchronize();


             rendering_loop(texID[i%2]);

           // rendering_loop(texID[i%2]);
           launch_nextGen(surfObj,width,height, i%2,(i+1)%2);
            launch_checkKernel(surfObj,width,height,(i+1)%2);

            i++;


        }

    // END ///////////////////////////////////////////////////////////////////////
    // unbind tex
    glBindTexture(GL_TEXTURE_2D, 0);

    //Disable the use of 2D textures.
    glDisable(GL_TEXTURE_2D);

    glDeleteTextures(2, texID);

    end_interop();
    // Destroy surface objects
    cudaDestroySurfaceObject(surf);

  //  CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaDeviceReset());

    //rendering_loop();
    end_rendering();








*/


   CHECK_CUDA(cudaDeviceSynchronize());
   CHECK_CUDA(cudaDeviceReset());

    return 0;
}






