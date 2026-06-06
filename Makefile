# Compiler and flags
CXX = g++ -g
CXXFLAGS = -std=c++17 -Wall -Wextra -O0 -g

IFLAGS += -I/usr/local/astro_boost/include
LFLAGS += -L/usr/local/astro_boost/lib
CXXFLAGS += -Wno-parentheses -Wno-deprecated-copy -Wno-implicit-fallthrough

IFLAGS += -I/usr/include/bullet/ -I/usr/local/include/bullet/ \
         -I/usr/include/eigen3 -I/usr/local/include/pcl-1.8 -I/usr/include/vtk-7.1 -I/usr/include/GLFW

LFLAGS += `pkg-config --libs opencv` -lpthread -lBulletCollision -lBulletDynamics \
         -lBulletSoftBody -lLinearMath -lboost_thread-mt -lrt -lboost_system \
         -lpcl_io -lpcl_common -lpcl_io_ply -lpcl_visualization -lpcl_filters -lpcl_segmentation \
         -lGL -lGLU -lglfw -lGLEW
LFLAGS += -lvtkRenderingCore-7.1 -lvtkCommonDataModel-7.1 -lvtkCommonMath-7.1 -lvtkCommonCore-7.1 -lvtkIOLegacy-7.1 -lvtkIOCore-7.1

# Source files and targets
SOURCES = voxel_representation.cpp show_point_cloud.cpp opengl_viewer.cpp utils.cpp
TARGETS = voxel_representation show_point_cloud

# Object files
OBJS = $(SOURCES:.cpp=.o)

.PHONY: all clean

all: $(TARGETS)

# Generic rule for building object files
%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(IFLAGS) -c $< -o $@

voxel_representation: voxel_representation.o opengl_viewer.o
	$(CXX) $(CXXFLAGS) $(IFLAGS) $^ -o $@ $(LFLAGS)

show_point_cloud: show_point_cloud.o opengl_viewer.o utils.o
	$(CXX) $(CXXFLAGS) $(IFLAGS) $^ -o $@ $(LFLAGS)

clean:
	rm -f $(TARGETS) *.o
