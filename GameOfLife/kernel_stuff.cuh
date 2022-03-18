#ifndef KERNEL_STUFF_CUH
#define KERNEL_STUFF_CUH

// System
#include <iostream>

// CUDA
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h>


///
/// \brief firstGen
/// \details Computes the first Generation of Cells randomly with curandState
/// \param outputSurfObj
/// \param width
/// \param height
/// \param extra
///
__global__ void firstGen(cudaSurfaceObject_t outputSurfObj, int width, int height, bool extra)
{
    // Calculate surface coordinates
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    // Don't do any computation if this thread is outside of the surface bounds.
    if(x >= width)
        return;
    if(y >= height)
        return;

    // Set up curandState for random numbers
    curandState custate;
    curand_init((unsigned long long)clock() + x+y, 0, 0, &custate);
    bool alive = (curand(&custate) % 2);

    // What to write to surface
    float4 element; // float4 for rgba values
    if (alive)
    {
       if (extra)
       {
           float rand1 = curand_uniform(&custate);
           float rand2 = curand_uniform(&custate);
           float rand3 = curand_uniform(&custate);
           element = make_float4(rand1, rand2, rand3, 1.0f);
       }
       else
           element = make_float4(1.0f, 1.0f, 1.0f, 1.0f);
    }
    else
    {
       element = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }

    // Write to output surface
    surf2Dwrite(element, outputSurfObj, x*sizeof(float4), y);
}


///
/// \brief   nextGen
/// \details Computes the next Generation of Cells according to the Game Of Life Rules
/// \param   inputSurfObj
/// \param   outputSurfObj
/// \param   width
/// \param   height
/// \param   extra
///
__global__ void nextGen(cudaSurfaceObject_t inputSurfObj, cudaSurfaceObject_t outputSurfObj, int width, int height, bool extra)
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
    surf2Dread(&cell,  inputSurfObj, x*sizeof(float4), y);
    bool state = cell.w == 1.0f ? 1 : 0;
    // Init vars
    bool kill = 0;
    int n_neighbors = 0;

    // Search for neighbors and adding to n_neighbors
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

            // extra rules
            if(extra)
            {
                // if a color component of the cell is greater than the other components,
                // the cell is defined as "being" that color
                if(cell.x > cell.y && cell.x > cell.z)
                {
                    n_neighbors += neighbor.x;
                }
                else if(cell.y > cell.x && cell.y > cell.z)
                {
                    n_neighbors += neighbor.y;
                    kill = (neighbor.z >= neighbor.x && neighbor.z >= neighbor.y)  ? 1 : 0; // green kills blue
                    kill = (neighbor.x >= neighbor.y && neighbor.x >= neighbor.z) ? 1 : 0; // green kills red
                    // if a cells kills another cell, a new cell is born in the color of the killer cell
                    if(kill){
                        float4 rip = make_float4(0.0f, 1.0f, 0.0f, 1.0f);
                        surf2Dwrite(rip, inputSurfObj, (x+k)*sizeof(float4), y+l); // needs to be regarded in the inputSurfaces too
                                                                                   // bc a cell cant be eaten twice
                        surf2Dwrite(rip, outputSurfObj, (x+k)*sizeof(float4), y+l);
                    }
                }
                else if(cell.z > cell.x && cell.z > cell.y)
                {
                    n_neighbors += neighbor.z;
                    kill = (neighbor.x >= neighbor.y && neighbor.x >= neighbor.z) ? 1 : 0; // red kills green
                    if(kill){
                        // if a cells kills another cell, a new cell is born in the color of the killer cell
                        float4 rip = make_float4(0.0f, 0.0f, 1.0f, 1.0f);
                        surf2Dwrite(rip, inputSurfObj, (x+k)*sizeof(float4), y+l); // needs to be regarded in the inputSurfaces too
                                                                                   // bc a cell cant be eaten twice
                        surf2Dwrite(rip, outputSurfObj, (x+k)*sizeof(float4), y+l);
                    }
                }
            }
            else
                n_neighbors += neighbor.w;
        }
    }

    // Apply rules from Game Of Life
    bool new_state =
       // changed to consider ranges, since we do not check the neighbors if they are a certain color
       // but simply add the neighbor's component of the cell's colour
      (n_neighbors <= 3.4f && n_neighbors >= 2.5f) ||
      (n_neighbors <= 2.4f && n_neighbors >= 1.5f && state)
       ? 1 : 0;

    // Write to output surface
    float4 element;
    // random
    bool mutate;
    curandState custate;
    curand_init((unsigned long long)clock() + x+y, 0, 0, &custate);
    mutate = (curand(&custate) % 2);

    float rand1 = curand_uniform(&custate);
    float rand2 = curand_uniform(&custate);

    if(new_state)
    {
        // if a cell survives, they become their dominant color -> value to 1.0f
        if(extra)
        {
            if(cell.x > .50f)
                element = make_float4(1.0f, rand1*mutate, rand2*mutate, 1.0f);
            else if(cell.y >.50f)
                element = make_float4(rand1*mutate, 1.0f, rand2*mutate, 1.0f);
            else if(cell.z >.50f)
                element = make_float4(rand1*mutate, rand2*mutate, 1.0f, 1.0f);
        }
        else
        {
            element = make_float4(1.0f, 1.0f, 1.0f, 1.0f);
        }
    }
    else
    {
        element = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
    surf2Dwrite(element, outputSurfObj,x*sizeof(float4), y);
}

///
/// \brief printGen
/// \details prints state of Cells to Console for Debugging Purposes
/// \param inputSurfObj
/// \param width
/// \param height
///
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




#endif // KERNEL_STUFF_CUH
