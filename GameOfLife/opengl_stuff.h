#ifndef OPENGL_STUFF_H
#define OPENGL_STUFF_H

// custom written for readability
#include "shader_stuff.h"

// variables
GLFWwindow* window;
GLFWwindow* getWindow(){ return window;}

unsigned int vboID;
unsigned int vaoID;
unsigned int eboID; // used for indexed drawing
unsigned int shaderProgram;
unsigned uniform_sampler_ourTexture;

float vertices[] =
{
    // positions          // colors           // texture coords
     1.0f,  1.0f, 0.0f,   0.0f, 0.0f, 0.0f,   1.0f, 1.0f,   // top right
     1.0f, -1.0f, 0.0f,   0.0f, 0.0f, 0.0f,   1.0f, 0.0f,   // bottom right
    -1.0f, -1.0f, 0.0f,   0.0f, 0.0f, 0.0f,   0.0f, 0.0f,   // bottom left
    -1.0f,  1.0f, 0.0f,   0.0f, 0.0f, 0.0f,   0.0f, 1.0f    // top left
};


unsigned int indices[] =
{
    // note that we start from 0!
     0, 1, 3,  // first triangle
     1, 2, 3   // second triangle
};

///
/// \brief CHECK_ERROR_GL
/// \details Checking for GL-related errors
///
void CHECK_ERROR_GL()
{
    GLenum err = glGetError();
    if(err != GL_NO_ERROR)
    {
        std::cerr << "GL Error!"  << std::endl;
        exit(-1);
    }
}

void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    glViewport(0, 0, width, height);
}


///
/// \brief   set_tex_paras
/// \details configuring the parameters of an OpenGL texture
/// \note    data, the last entry in glTexImage2D, must be a nullpointer!
///          the type specified must match with Cuda Surface Object
///          float4 <-> rgba
/// \param   tex  the texture to be configured
/// \return
///
GLuint config_tex(GLuint tex, int width, int height)
{
    glGenTextures(1, &tex);

    glBindTexture(GL_TEXTURE_2D, tex);
    {
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST        );
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST        );
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S,     GL_CLAMP_TO_BORDER);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T,     GL_CLAMP_TO_BORDER);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T,     GL_CLAMP_TO_BORDER);

        glTexImage2D(GL_TEXTURE_2D, 0,GL_RGBA32F, width,height, 0,  GL_RGBA, GL_FLOAT, NULL);
    }
    glBindTexture(GL_TEXTURE_2D, 0);

    CHECK_ERROR_GL();
    return tex;
}

///
/// \brief   checkTex
/// \details used for checking if the kernel changes are saved in the texture
/// \param   texID
/// \param   width
/// \param   height
/// \return
///
int checkTex(GLuint *texID,int width, int height)
{

//    glActiveTexture(GL_TEXTURE0);
//    glEnable(GL_TEXTURE_2D);
    int numElements = width*height*4;

    float *data = new float[numElements];

    glBindTexture(GL_TEXTURE_2D, *texID);
    {
        glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_FLOAT, data);
    }
    glBindTexture(GL_TEXTURE_2D, 0);

    bool fail = false;
    for (int i = 0; i < width; i++)
    {
        for (int j = 0; j < height; j++)
        {
            printf("%f ", data[i * width + j]);
            if(data[i * width + j] != 255.0f || data[i] != 0.0f)
            {
                std::cerr << "texture doesnt have right values" << std::endl;
                fail = true;
            }
        }
        printf("\n");
    }
    delete [] data;
    return 0;
}


///
/// \brief   init_opengl
/// \details sets up GLFW, an OpenGL context, and GLAD
/// \return
///
int init_opengl()
{
    glfwInit();

    //configure GLFW
    // what option we want to configure
    //  second argument is an integer that sets the value of our option.
    // Version of opengl we wanna use: find current install on linux w $ glxinfo
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 4);
    // elling GLFW we want to use the core-profile means we'll get access to a smaller subset of OpenGL features without backwards-compatible features we no longer need.
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    //glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

    // create window obj
    // A window and its OpenGL or OpenGL ES context are created with
    // glfwCreateWindow, which returns a handle to the created window object. For example
    //window = glfwCreateWindow(800, 600, "Game Of Life", glfwGetPrimaryMonitor(), NULL);
    window = glfwCreateWindow(800, 600, "Game Of Life", NULL, NULL);


    if (window == NULL)
    {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    // tell GLFW to make the context of our window the main context on the current thread.
    glfwMakeContextCurrent(window);


    // glad ////////////////////////////////////////////

    //  initialize GLAD before we call any OpenGL function bc GLAD manages
    // function pointers for OpenGL
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

    //  tell OpenGL the size of the rendering window so OpenGL knows
    // how we want to display the data and coordinates with respect to the window.
    glViewport(0, 0, 800, 600);

    // register a callback function on the window that gets called each time the window is resized
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    std::cerr << "OPENGL SUCCESSFULLY INITIALIZED" << std::endl;

    return 0;
}

///
/// \brief setUp_rendering
/// \return
///
int setUp_rendering()
{

    init_opengl();

    shaderProgram = shader_init();
    // Specifies the program object to be queried.
    uniform_sampler_ourTexture = glGetUniformLocation(shaderProgram, "ourTexture");
    check_shaderProgram(shaderProgram);

    // vbo & VAO & ebo /////////////////////////////////////////
    glGenVertexArrays(1, &vaoID);
    glGenBuffers(1, &vboID);
    glGenBuffers(1, &eboID);

    // bind the Vertex Array Object first,
    // then bind and set vertex buffer(s),
    // and then configure vertex attributes(s).
    glBindVertexArray(vaoID);

    // bind to buffer type
    glBindBuffer(GL_ARRAY_BUFFER, vboID);
    // copies vertices data into buffer mem
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW); // GL_STATIC_DRAW: the data is set only once and used many times.

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, eboID);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

    // tell OpenGL how it should interpret the vertex data before rendering.
    //Each vertex attribute takes its data from memory managed by the VBO (cur bound)
    // position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    // color attribute
  // glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(3 * sizeof(float)));
   //glEnableVertexAttribArray(1);
    // texture coord attribute
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(6 * sizeof(float)));
    glEnableVertexAttribArray(2);


    // texture //////////////////////////////////////////////////////////////////////////////7

    // unbind
    // note that this is allowed, the call to glVertexAttribPointer registered VBO as the vertex attribute's bound vertex buffer object so afterwards we can safely unbind
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    return 0;
}


///
/// \brief rendering
/// \param texIDhere
///
void rendering(GLuint texIDhere)
{
    glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    glUseProgram(shaderProgram);

    glBindTexture(GL_TEXTURE_2D, texIDhere);

    glUniform1i(uniform_sampler_ourTexture, 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, eboID);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

    glBindTexture(GL_TEXTURE_2D, 0); //?

    glfwSwapBuffers(window);
    glfwPollEvents();
}

///
/// \brief   end_rendering
/// \details clean/delete all of GLFW's resources that were allocated
/// \return
///
void end_rendering()
{
    glDeleteVertexArrays(1, &vaoID);
    glDeleteBuffers(1, &vboID);
    glDeleteBuffers(1, &eboID);
    glDeleteProgram(shaderProgram);
    CHECK_ERROR_GL();
    glfwTerminate();
}


#endif // OPENGL_STUFF_H
