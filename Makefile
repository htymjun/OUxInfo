FC = gfortran
CC = g++
# Python and pybind11
PYTHON_VERSION = 3.10
PYTHON_INC     = $(shell python$(PYTHON_VERSION)-config --includes)
PYBIND11_INC   = $(shell python3 -m pybind11 --includes)
# FLAGS
FFLAGS  = -Ofast -g -Wall -march=native
CFLAGS  = --compile -fPIC -O3 -std=c++17 $(PYTHON_INC) $(PYBIND11_INC)
LDFLAGS = -shared -fPIC

.SUFFIXES : .f90

%.o: %.f90
		$(COMPILE.f) $<

%.mod: %.f90 %.o
		@:
vpath %f90 ./special:./KDTree

FSRC = special.o \
      kdtree2.o \
      shannon.o \
      shannon_cbind.o
CSRC = wrapper.cpp

FOBJ = $(FSRC:.f90=.o)
COBJ = $(CSRC:.cpp=.o)
OBJ  = $(FOBJ) $(COBJ)

TARGET = shannon.so
all: $(TARGET)

$(TARGET): $(OBJ)
	$(FC) $(LDFLAGS) -o $@ $(OBJ) -lstdc++ 

%.o: %.f90
	$(FC) $(FFLAGS) -c $<

%.o: %.cpp
	$(CC) $(CFLAGS) -c $< -o $@

# dependencies
shannon.o: kdtree2.mod special.mod
shannon_cbind.o: shannon.mod
wrapper.o: $(FSRC:.f90=.o)
cufd.so: wrapper.o

clean:
		rm -f *.mod *.o *.out

