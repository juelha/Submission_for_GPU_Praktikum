# cuda stuff s: https://forum.qt.io/topic/114853/cuda-10-2-in-qt-5-14-ubuntu-18-04/12
# opengl stuff s:


TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt

# explain source/ header assignment
# https://forums.developer.nvidia.com/t/multiple-definition-error-on-device-function-in-header-file/21530/5

SOURCES += \
    glad.c
           # main.cpp \
     #checkerboard.cpp

HEADERS += \ shader.h \
                cuda_stuff.cuh \
    interop_stuff.cuh \
    opengl_stuff.h \



# Cuda sources
CUDA_SOURCES += \
                main.cu \



#############
# OpenGL    #
#############


QT       += core gui opengl # added opengl

LIBS += \
        #-lglfw3 \
        -lglfw3 -lGL -lX11 -lpthread -lXrandr -lXi -ldl \# https://learnopengl.com/Getting-started/Creating-a-window
        -lGLU -lGL #-lglut -lGLU -lGL \# added linux specific,
                            #if error:  sudo apt-get install libglu1-mesa-dev freeglut3-dev mesa-common-dev
                            # source: http://www.codebind.com/linux-tutorials/install-opengl-ubuntu-linux/
       # -lGLEW # http://www.cplusplus.com/forum/general/154392/

# -glew32

#############
# CUDA      #
#############

## SYSTEM_TYPE - compiling for 32 or 64 bit architecture
SYSTEM_TYPE = 64

# Path to cuda installation
CUDA_DIR = /usr/local/cuda

#CUDA_DEFINES += /usr/local/cuda-9.0

# Path to header and libs files
INCLUDEPATH += \
               $$CUDA_DIR/include \
               $$CUDA_DIR/samples/common/inc

QMAKE_LIBDIR += \
                $$CUDA_DIR/lib64 \
                $$CUDA_DIR/samples/common/inc

# GPU architecture

CUDA_DIR = /usr/local/cuda
INCLUDEPATH  += $$CUDA_DIR/include
QMAKE_LIBDIR += $$CUDA_DIR/lib64
LIBS += -lcudart -lcuda
CUDA_ARCH = sm_86                # CHANGE HERE TO ARCH
                                 # https://arnon.dk/matching-sm-architectures-arch-and-gencode-for-various-nvidia-cards/
# Here are some NVCC flags I've always used by default.
NVCCFLAGS     = --compiler-options -fno-strict-aliasing -use_fast_math --ptxas-options=-v #-g -G
CUDA_INC = $$join(INCLUDEPATH,' -I','-I',' ')

#-g # standard gnu switch for building a debug (host) code#
#-G # builds debug device code

cuda.commands = $$CUDA_DIR/bin/nvcc -m64 -O3 -arch=$$CUDA_ARCH -c $$NVCCFLAGS \
                $$CUDA_INC $$LIBS  ${QMAKE_FILE_NAME} -o ${QMAKE_FILE_OUT} \
                2>&1 | sed -r \"s/\\(([0-9]+)\\)/:\\1/g\" 1>&2
cuda.dependency_type = TYPE_C
cuda.depend_command = $$CUDA_DIR/bin/nvcc -O3 -M $$CUDA_INC $$NVCCFLAGS ${QMAKE_FILE_NAME}| sed \"s/^.*: //\"
cuda.input = CUDA_SOURCES

cuda.output = ${OBJECTS_DIR}${QMAKE_FILE_BASE}.o
#cuda.output = $${OBJECTS_DIR}/${QMAKE_FILE_BASE}$${QMAKE_EXT_OBJ}
QMAKE_EXTRA_COMPILERS += cuda ```

DISTFILES += \
    fragment_shader.fs \
    vertex_shader.vs
