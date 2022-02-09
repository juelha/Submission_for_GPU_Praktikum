#ifndef SHADER_STUFF_H
#define SHADER_STUFF_H

// OpenGL Graphics includes
#include <glad/glad.h>
#include <GLFW/glfw3.h>


// includes, system
#include <iostream>


unsigned int vertexShaderID;
unsigned int fragmentShaderID;
// todo: make prtty scroll down https://learnopengl.com/Getting-started/Shaders

/*
    * handling all of the GLSL Shader stuff
    * shaders are stored in C-chars for convenience
    * source: https://learnopengl.com/Getting-started/Shaders
    * shader skeleton
    *   version declaration
    *   list of in- and output variables
    *       (note: in and out are GLSL keywords for input and outputs)
    *       type & name of vars that are passed along needs to be the same! -> wherever an output variable matches with an input variable of the next shader stage they're passed along (this is done when linking a program object)
    *   uniforms
    *   main func
    * vertex_shader -> fragment_shader
*/

///
/// \brief vertex_shader_src
/// each input variable is also known as a vertex attribute
const char* vertex_shader_src =
    "#version 460 core\n"
    "layout (location = 0) in vec3 aPos;\n"
    "layout (location = 1) in vec4 aColor;\n" // vec4 ???
    "layout (location = 2) in vec2 aTexCoord;\n"
    "\n"
    //"out vec4 ourColor;\n"
    "out vec2 TexCoord;\n"
    "\n"
    "void main()\n"
    "{\n"
  //  "   gl_Position = vec4(aPos.x, aPos.y, aPos.z, 1.0);\n"
    "    gl_Position = vec4(aPos, 1.0);\n"
    //"    ourColor = aColor;\n"
    //"    ourColor = vec4(1.0f, 0.5f, 0.2f, 1.0f); \n"
    "    TexCoord = aTexCoord;\n"
    "}";


const char* fragment_shader_src =
    "#version 460 core\n"
    "out vec4 FragColor;\n"
    "\n"
   // "in vec4 ourColor;\n"
    "in vec2 TexCoord;\n"
    "\n"
    "uniform sampler2D ourTexture;\n"
    "\n"
    "void main()\n"
    "{\n"
    //"    FragColor = vec4(vec3(texture(ourTexture, TexCoord).r), 1.);\n"
        "    FragColor = texture(ourTexture, TexCoord);\n" // works

    // "    FragColor = vec4(texture(ourTexture, TexCoord).r);\n"  // works
    // "FragColor = ourColor; \n"
    "}";



int check_shader(unsigned int shaderID){
    int  success;
    char infoLog[512];
    glGetShaderiv(shaderID, GL_COMPILE_STATUS, &success);
    if(!success)
    {
        glGetShaderInfoLog(shaderID, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" << infoLog << std::endl;
        return -1;
    }
    return 0;
}

int check_shaderProgram(unsigned int shaderProgram){
    // check for error
    int  success;
    char infoLog[512];
    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
    if(!success) {
        glGetProgramInfoLog(shaderProgram, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" << infoLog << std::endl;
     //printf("%s\n", infoLog1);
        return -1;
    }

    return 0;
}

unsigned int shader_init()
{
    unsigned int shaderProgram; //  final linked version of multiple shaders combined.

    // shaders ///////////////////////////////////////7

    // vertex shader
    vertexShaderID = glCreateShader(GL_VERTEX_SHADER);
    // attach the shader source code to the shader object and compile the shader
    glShaderSource(vertexShaderID, 1, &vertex_shader_src, NULL);
    glCompileShader(vertexShaderID);
    // check for error
    check_shader(vertexShaderID);

    // fragment shader
    fragmentShaderID = glCreateShader(GL_FRAGMENT_SHADER);
    // attach the shader source code to the shader object and compile the shader
    glShaderSource(fragmentShaderID, 1, &fragment_shader_src, NULL);
    glCompileShader(fragmentShaderID);
    // check for error
    check_shader(fragmentShaderID);

    // shader program ////////////////////////////////////////
    shaderProgram = glCreateProgram();
    // attach and link
    glAttachShader(shaderProgram, vertexShaderID);
    glAttachShader(shaderProgram, fragmentShaderID);
    glLinkProgram(shaderProgram);
    check_shaderProgram(shaderProgram);


    glDeleteShader(vertexShaderID);
    glDeleteShader(fragmentShaderID);



    return shaderProgram;

}


#endif // SHADER_STUFF_H
