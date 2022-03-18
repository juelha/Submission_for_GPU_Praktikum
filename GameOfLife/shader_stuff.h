#ifndef SHADER_STUFF_H
#define SHADER_STUFF_H

// OpenGL Graphics includes
#include <glad/glad.h>
#include <GLFW/glfw3.h>

// includes, system
#include <iostream>

// vars
unsigned int vertexShaderID;
unsigned int fragmentShaderID;



///
/// \brief vertex_shader_src
/// \details shaders are stored in C-chars for convenience
///          the original source was changed to match the float4 values
///          source: https://learnopengl.com/Getting-started/Shaders
///
const char* vertex_shader_src =
    "#version 460 core\n"
    "layout (location = 0) in vec3 aPos;\n"
    "layout (location = 1) in vec4 aColor;\n" // vec4 -> rgba
    "layout (location = 2) in vec2 aTexCoord;\n"
    "\n"
    "out vec2 TexCoord;\n"
    "\n"
    "void main()\n"
    "{\n"
    "    gl_Position = vec4(aPos, 1.0);\n"
    "    TexCoord = aTexCoord;\n"
    "}";

///
/// \brief fragment_shader_src
/// \details shaders are stored in C-chars for convenience
///          the original source was changed to match the float4 values
///          source: https://learnopengl.com/Getting-started/Shaders
///
const char* fragment_shader_src =
    "#version 460 core\n"
    "out vec4 FragColor;\n"
    "\n"
    "in vec2 TexCoord;\n"
    "\n"
    "uniform sampler2D ourTexture;\n"
    "\n"
    "void main()\n"
    "{\n"
        "    FragColor = texture(ourTexture, TexCoord);\n" // works
    "}";


///
/// \brief check_shader
/// \details Checks if (here: Vertex- and Fragment-) Shader can be compiled
///          source: https://learnopengl.com/Getting-started/Shaders
/// \param shaderID
///
void check_shader(unsigned int shaderID)
{
    int  success;
    char infoLog[512];
    glGetShaderiv(shaderID, GL_COMPILE_STATUS, &success);
    if(!success)
    {
        glGetShaderInfoLog(shaderID, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" << infoLog << std::endl;
        exit(-1);
    }
}

///
/// \brief check_shaderProgram
/// \details Checks if shaderProgram can be compiled
///           source: https://learnopengl.com/Getting-started/Shaders
/// \param shaderProgram
///
void check_shaderProgram(unsigned int shaderProgram)
{
    // check for error
    int  success;
    char infoLog[512];
    // HERE GL_LINK_STATUS
    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
    if(!success) {
        glGetProgramInfoLog(shaderProgram, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::LINK_FAILED\n" << infoLog << std::endl;
        exit(-1);
    }
}

///
/// \brief shader_init
/// \details sets up Vertex-, Fragment-Shader and inits Shader Program
/// \return Shader Program that is attached with Vertex and Fragment Shader
///
unsigned int shader_init()
{
    unsigned int shaderProgram; //  final linked version of multiple shaders combined.

    // shaders ///////////////////////////////////////

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
