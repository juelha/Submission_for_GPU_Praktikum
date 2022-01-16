#ifndef OPENGL_STUFF_H
#define OPENGL_STUFF_H


// OpenGL Graphics includes
#include <glad/glad.h>
#include <GLFW/glfw3.h>
//#include "shader.h"

// includes, system
#include <iostream>

// settings
const unsigned int SCR_WIDTH = 800;
const unsigned int SCR_HEIGHT = 600;


char title[] = "Texture - Checkerboard Pattern";  // Title in windowed mode
int windowPosX   = 50;      // Windowed mode's top-left corner x
int windowPosY   = 50;      // Windowed mode's top-left corner y
bool fullScreenMode = true; // Full-screen or windowed mode?

GLFWwindow* window;

GLFWwindow* getWindow() {
  return  window;
}

unsigned int vboID;
unsigned int vaoID;
unsigned int eboID; // element.. used for indexed drawing

unsigned int vertexShaderID;
unsigned int fragmentShaderID;
unsigned int shaderProgram; //  final linked version of multiple shaders combined.

unsigned uniform_sampler_ourTexture;

void CHECK_ERROR_GL() {
    GLenum err = glGetError();
    if(err != GL_NO_ERROR) {
//        std::cerr << "GL Error: " << gluErrorString(err) << std::endl;
        std::cerr << "GL Error: "  << std::endl;
        exit(-1);
    }
}

// functions
void framebuffer_size_callback(GLFWwindow* window, int width, int height);

void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    glViewport(0, 0, width, height);
}


/////////////////
/// \brief set_tex_paras
/// \param tex
/// \return
///data may be a null pointer.
/// In this case, texture memory is allocated to accommodate a texture of width width and height
/// You can then download subtextures to initialize this texture memory.
/// The image is undefined if the user tries to apply an uninitialized portion of the
/// texture image to a primitive.
/// glTexImage2D specifies the two-dimensional texture for the texture object
///  bound to the current texture unit, specified with glActiveTexture.
///glTexImage2D specifies the two-dimensional texture for the texture object bound to the current texture unit, specified with glActiveTexture.
//Note that the type of this texture is an RGBA UNSIGNED_BYTE type. When CUDA surfaces
//are synchronized with OpenGL textures, the surfaces will be of the same type.
//They won't know or care about their data types though, for they are all just byte arrays
//at heart. So be careful to ensure that any CUDA kernel that handles a CUDA surface
//uses it as an appropriate type. You will see that the update_surface kernel (defined
//above) treats each pixel as four unsigned bytes along the X-axis: one for red, green, blue,
//and alpha respectively.
GLuint set_tex_paras(GLuint tex, int width, int height){
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


void checkTex(GLuint *texID,int width, int height)
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
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < height; j++) {
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
}
const char* vertex_shader_src =
        "#version 460 core\n"
        "layout (location = 0) in vec3 aPos;\n"
        "layout (location = 1) in vec3 aColor;\n"
        "layout (location = 2) in vec2 aTexCoord;\n"
        "\n"
        "out vec3 ourColor;\n"
        "out vec2 TexCoord;\n"
        "\n"
        "void main()\n"
        "{\n"
        "    gl_Position = vec4(aPos, 1.0);\n"
        "    ourColor = aColor;\n"
        "    TexCoord = aTexCoord;\n"
        "}";
const char* fragment_shader_src =
    "#version 460 core\n"
    "out vec4 FragColor;\n"
    "\n"
    "in vec3 ourColor;\n"
    "in vec2 TexCoord;\n"
    "\n"
    "uniform sampler2D ourTexture;\n"
    "\n"
    "void main()\n"
    "{\n"
    "    FragColor = vec4(vec3(texture(ourTexture, TexCoord).r), 1.);\n"
    "}";



float vertices[] = {
    // positions          // colors           // texture coords
     1.0f,  1.0f, 0.0f,   1.0f, 0.0f, 0.0f,   1.0f, 1.0f,   // top right
     1.0f, -1.0f, 0.0f,   0.0f, 1.0f, 0.0f,   1.0f, 0.0f,   // bottom right
    -1.0f, -1.0f, 0.0f,   0.0f, 0.0f, 1.0f,   0.0f, 0.0f,   // bottom left
    -1.0f,  1.0f, 0.0f,   1.0f, 1.0f, 0.0f,   0.0f, 1.0f    // top left
};


unsigned int indices[] = {  // note that we start from 0!
     0, 1, 3,  // first triangle
     1, 2, 3    // second triangle
};

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



int begin_rendering()
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
    window = glfwCreateWindow(800, 600, "My Title", NULL, NULL);

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
    //

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

    // texID[0] = get_checker_texture();


    std::cerr << "OPENGL SUCCESSFULLY INITIALIZED" << std::endl;

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


    // check for error
    int  success;
    char infoLog[512];
    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
    if(!success) {
        glGetProgramInfoLog(shaderProgram, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" << infoLog << std::endl;
      }
    glDeleteShader(vertexShaderID);
    glDeleteShader(fragmentShaderID);

    // Specifies the program object to be queried.
    uniform_sampler_ourTexture = glGetUniformLocation(shaderProgram, "ourTexture");
    int success1;
    char infoLog1[512];

    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success1);
    if(!success1) {
        glGetProgramInfoLog(shaderProgram, 512, NULL, infoLog1);
        printf("%s\n", infoLog1);
    }

    // vbo & VAO & ebo/////////////////////////////////////////7
    glGenVertexArrays(1, &vaoID);
    glGenBuffers(1, &vboID);
    glGenBuffers(1, &eboID);

    // bind the Vertex Array Object first, then bind and set vertex buffer(s), and then configure vertex attributes(s).
    glBindVertexArray(vaoID);

    // bind to buffer type
    glBindBuffer(GL_ARRAY_BUFFER, vboID);
    // copies vertices data into buffer mem
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW); // GL_STATIC_DRAW: the data is set only once and used many times.

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, eboID);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

    // tell  OpenGL how it should interpret the vertex data before rendering.
    //Each vertex attribute takes its data from memory managed by a VBO (cur bound)
    // position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    // color attribute
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    // texture coord attribute
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(6 * sizeof(float)));
    glEnableVertexAttribArray(2);


    // texture //////////////////////////////////////////////////////////////////////////////7

    // unbind
    // note that this is allowed, the call to glVertexAttribPointer registered VBO as the vertex attribute's bound vertex buffer object so afterwards we can safely unbind
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    // remember: do NOT unbind the EBO while a VAO is active as the bound element buffer object IS stored in the VAO; keep the EBO bound.
     //glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

    // You can unbind the VAO afterwards so other VAO calls won't accidentally modify this VAO, but this rarely happens. Modifying other
    // VAOs requires a call to glBindVertexArray anyways so we generally don't unbind VAOs (nor VBOs) when it's not directly necessary.
  //  glBindVertexArray(0);

    // uncomment this call to draw in wireframe polygons.
    //glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);





return 0;

}



void rendering_loop(
       // GLFWwindow* window,
        GLuint texIDhere)
{
    glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    glUseProgram(shaderProgram);

   // glBindVertexArray(vaoID); // seeing as we only have a single VAO there's no need to bind it every time, but we'll do so to keep things a bit more organized

    //glActiveTexture(GL_TEXTURE0);
   // glEnable(GL_TEXTURE_2D);
  // glActiveTexture(GL_TEXTURE0);
  // glActiveTexture(GL_TEXTURE1);

   glBindTexture(GL_TEXTURE_2D, texIDhere);

   glUniform1i(uniform_sampler_ourTexture, 0);
   glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, eboID);
   glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

   glBindTexture(GL_TEXTURE_2D, 0); //?

    //  glDrawArrays(GL_TRIANGLES, 0, 3);
    //            mode,        n_vertices,  offset
  //  glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
  //  glDrawElements(GL_PATCHES, 4, GL_UNSIGNED_INT, 0);

    // glBindVertexArray(0); // no need to unbind it every time

    glfwSwapBuffers(window);
    glfwPollEvents();


}

void end_rendering(){

    // clean/delete all of GLFW's resources that were allocated


    glDeleteVertexArrays(1, &vaoID);
    glDeleteBuffers(1, &vboID);
    glDeleteBuffers(1, &eboID);
    glDeleteProgram(shaderProgram);

    glfwTerminate();

}














/*
 *
 *
/////
/// \brief initGLTex
/// \param texID
///
void initGLTex(GLuint texID) {

    // bind a named texture to a texturing target
    glBindTexture(GL_TEXTURE_2D, texID);
    {

        //  set texture parameters

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST        );
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST        );
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S,     GL_CLAMP_TO_BORDER);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T,     GL_CLAMP_TO_BORDER);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T,     GL_CLAMP_TO_BORDER);


     //   glTextureStorage2D(texID,0,GL_LUMINANCE,IMAGE_COLS,IMAGE_ROWS);




        // Create texture from image data
        //  create a texture using glTexImage
        // update its contents with glTexSubImage
      glTexImage2D(
             //glTextureStorage2D(
                   GL_TEXTURE_2D, // type of texture
                    0,          // Level 0 is the base image levelPyramid level (for mip-mapping) - 0 is the top level
                    GL_LUMINANCE, // internal format, Specifies the number of color components in the texture.
                    IMAGE_COLS, //width
                    IMAGE_ROWS, // height
                    0,          // border (must be 0)/ Border width in pixels (can either be 1 or 0)
                    GL_LUMINANCE, //input image format Each element is a single luminance value. The GL converts it to floating point, then assembles it into an RGBA element by replicating the luminance value three times for red, green, and blue and attaching 1 for alpha. Each component is then clamped to the range [0,1].
                  //GL_INT,
                      GL_UNSIGNED_BYTE, // image datatype, Specifies the data type of the texel data.
                   &texID);
                    //&texID);  // Specifies a pointer to the image data in memory.





     //  glTexImage2D(GL_TEXTURE_2D, 0, 3, IMAGE_COLS, IMAGE_ROWS, 0, GL_RGB,
       //      GL_UNSIGNED_BYTE, imageData);


    }

    // unbind and free image
 //  glBindTexture(GL_TEXTURE_2D, 0);
    //stbi_image_free(data);
    //glBindTexture(GL_TEXTURE_2D, 0);

    //Enable the use of 2D textures
    glEnable(GL_TEXTURE_2D);

   // glEnable(GL_TEXTURE_2D);  // Enable 2D texture
  // glutMainLoop();    // Enter the infinitely event-handling loop
    //

}





////////////////////////////////////////////////////////////////////////////////
//! Check Texture
////////////////////////////////////////////////////////////////////////////////
void checkTex(GLuint texID)
{
    int numElements = IMAGE_ROWS*IMAGE_COLS;//*sizeof(GL_UNSIGNED_BYTE);
    int *data = new int[numElements];

    glBindTexture(GL_TEXTURE_2D, texID);
    {

        glGetTexImage(GL_TEXTURE_2D, 0,   GL_LUMINANCE,  GL_UNSIGNED_BYTE, data);

    }
    glBindTexture(GL_TEXTURE_2D, 0);


    bool fail = false;
    for(int i = 0; i < numElements ; i++)
    {            printf("%s","here ");

        printf("%d",data[i] );
       // cerr << data[i] << endl;
        if(data[i] != 1)//0 || data[i] != 255)
        {
          //  cerr << "Not 0 0r 255, failed writing to texture" << endl;
            fail = true;
           // printf("%s","failed");
        }
    }
    if(!fail)
    {
     //   cerr << "All Elements == 1.0f, texture write successful" << endl;
    }

    delete [] data;
}




void CHECK_ERROR_GL() {
    GLenum err = glGetError();
    if(err != GL_NO_ERROR) {
        std::cerr << "GL Error: " << gluErrorString(err) << std::endl;
        exit(-1);
    }
}
void renderGLTex(){

      glBindFramebuffer(GL_FRAMEBUFFER, 0);

      float vertices[] = {
      //   positions     texture coordinates
          -1.0f,  1.0f,  0.0f, 1.0f,
          -1.0f, -1.0f,  0.0f, 0.0f,
           1.0f, -1.0f,  1.0f, 0.0f,

          -1.0f,  1.0f,  0.0f, 1.0f,
           1.0f, -1.0f,  1.0f, 0.0f,
           1.0f,  1.0f,  1.0f, 1.0f
      };
      // Clear color buffer with black
      glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
      glClearDepth(1.0f);
      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

      // Face culling
      glEnable(GL_CULL_FACE);




}

void initFramebuffer(    GLuint fboID,GLuint texID[2])
{
  bool Validated(true);

  // s: https://stackoverflow.com/questions/10230976/unhandled-exception-using-glgenbuffer-on-release-mode-only-qt
  GLenum init = glewInit();

  glGenFramebuffers(1, &fboID);

  glBindFramebuffer(GL_FRAMEBUFFER, fboID);
 // glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, texID[1], 0); // error pos


  glFramebufferTexture2D(GL_FRAMEBUFFER,  // target: the framebuffer type we're targeting (draw, read or both).
                         GL_COLOR_ATTACHMENT0, // : the type of attachment we're going to attach
                         GL_TEXTURE_2D,  // the type of the texture you want to attach.
                         texID[0], //the actual texture to attach.
                            0);  // mipmap lvl




  //glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, texID[1], 0);

 // glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, m_frontfaceInfoDepthTextureName, 0);

  if(glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE){

    std::cout << "ERROR::FRAMEBUFFER:: Framebuffer is not complete!" << std::endl;
    //exit(-1);

  }
    // execute victory dance
// And finally don't forget to unbind the framebuffer so that you don't accidentally render to it:
  glBindFramebuffer(GL_FRAMEBUFFER, 0);
}



/////
/// HANDLER STUFF
///
void displayCallback()
{
  glClearColor(0, 0, 0, 0);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  // Calculate the size of each cell in each direction.
 // GLfloat xSize = zoomFactor * (ptr->right - ptr->left) / ptr->width_;
//  GLfloat ySize = zoomFactor * (ptr->top - ptr->bottom) / ptr->height_;

  GLint width = window_width;//ptr->width_;
  GLint height = window_width;//ptr->height_;


  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();


  glMatrixMode(GL_MODELVIEW);
  // Load the identity transformation matrix.
  glLoadIdentity();

  // Define the scale transformation so as to properly view the grid.
  //glScalef(xSize, ySize, 1.0f);

  // Apply a translation transformation so as to place the center of the grid
  // on the center of the window and move it when the user moves it using the
  // keyboard arrow keys.
  glTranslatef(-width / 2.0f, height / 2.0f, 0.0f);

  glMatrixMode(GL_TEXTURE);
  glLoadIdentity();
 // glTranslatef(deltaX, deltaY, 0.0f);


 // glBindTexture(GL_TEXTURE_2D, ptr->gl_texturePtr);

  //glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, ptr->colorBufferId_);
  glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, 0);

  glBegin(GL_QUADS);

      glTexCoord2f(0.0f, 0.0f);
      glVertex2f(0.0f, 0.0f);

      glTexCoord2f(1.0f, 0.0f);
      glVertex2f(width, 0.0f);

      glTexCoord2f(1.0f, 1.0f);
      glVertex2f(width, -height);

      glTexCoord2f(0.0f, 1.0f);
      glVertex2f( 0.0f, -height);

  glEnd();

  // if (ptr->gpuOn_)
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
  glBindTexture(GL_TEXTURE_2D, 0);

 // ptr->drawGameInfo();

  glFlush();
  glutSwapBuffers();

  return;
}

// Handler for window paint and re-paint event
void display(void) {
   glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // Clear screen and depth buffers

   // Draw cube
   glLoadIdentity();   // Reset the view
   glTranslatef(0.0f, 0.0f, -5.0f);

   glBegin(GL_QUADS);

      // 1 plane
      glTexCoord2f(0.0f, 0.0f); glVertex3f(-1.0f, -1.0f,  1.0f);
      glTexCoord2f(1.0f, 0.0f); glVertex3f( 1.0f, -1.0f,  1.0f);
      glTexCoord2f(1.0f, 1.0f); glVertex3f( 1.0f,  1.0f,  1.0f);
      glTexCoord2f(0.0f, 1.0f); glVertex3f(-1.0f,  1.0f,  1.0f);


   glEnd();
   glutSwapBuffers(); // Swap front and back buffers (double buffered mode)

   //glFlush();
}

// Handler for window's re-size event
void reshape(GLsizei width, GLsizei height) {  // GLsizei: non-negative integer
   if (height == 0) height = 1;  // prevent divide by 0

   // Set the viewport (display area) to cover entire application window
   glViewport(0, 0, width, height);

   // Select the aspect ratio of the clipping area to match the viewport
   glMatrixMode(GL_PROJECTION);  // Select the Projection matrix
   glLoadIdentity();             // Reset the Projection matrix
   gluPerspective(45.0, (float)width / (float)height, 0.1, 100.0);

   // Reset the Model-View matrix
   glMatrixMode(GL_MODELVIEW);  // Select the Model-View matrix
   glLoadIdentity();            // Reset the Model-View matrix
}

// Handler for key event
void keyboard(unsigned char key, int x, int y) {
   switch (key) {
      case 27:  // ESC key: exit the program
        // exit(0); break;
      default: break;
   }
}

void wait_for_data(void)
{
    //glBindTexture(GL_TEXTURE_2D, texID[1]);

   // glutPostRedisplay(); //everytime you are done
    // drawing you put it on the screen

}

// periodic func
void updating(int i){

    int time_intervall = 1000; // once a sec

    glutPostRedisplay(); // opengl will be calling dispplay every time intervall

    // calling itself every timer intervall
    glutTimerFunc(
                time_intervall,
                updating,
                0
                );

    glBindTexture(GL_TEXTURE_2D, texID[i%2]);

}


void initGlut(int argc, char** argv){
    glutInit(&argc, argv);      // Initialize GLUT
    glutInitDisplayMode(GLUT_LUMINANCE | GLUT_DOUBLE); // Set mode
                                          // Bit mask to select a window with a ``luminance'' color model.
    glutInitWindowSize(windowWidth, windowHeight);  // Initial window width and height
    glutInitWindowPosition(windowPosX, windowPosY); // Initial window top-left corner (x, y)
    glutCreateWindow(title);     // Create window with the given title

   glutDisplayFunc(display);    // Register handler for window re-paint

    // calls specified func after a specified time intervall has passed
  //  glutTimerFunc(1000, updating, 0 );


   // glutDisplayFunc(displayCallback);
  //  glutIdleFunc(gameLoopCallback);


    glutReshapeFunc(reshape);    // Register handler for window re-size
    glutKeyboardFunc(keyboard);  // Register handler for key event


    // default initialization
    glClearColor(0.0, 0.0, 0.0, 1.0);
    glDisable(GL_DEPTH_TEST);


    // viewport
    glViewport(0, 0, window_width, window_height);

    // projection
    glMatrixMode(GL_PROJECTION);     // Make a simple 2D projection on the entire window
    glLoadIdentity();

    glOrtho(0.0, window_width, window_height, 0.0, 0.0, 100.0);

    glMatrixMode(GL_MODELVIEW);    // Set the matrix mode to object modeling

    gluPerspective(60.0, (GLfloat)window_width / (GLfloat) window_height, 0.1, 10.0);

    CHECK_ERROR_GL();

}

*/


#endif // OPENGL_STUFF_H
