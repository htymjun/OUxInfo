# pybind11 + nanoflann のコンパイル例

PYTHON_INCLUDE := $(shell python3 -m pybind11 --includes)
PYTHON_LIBS := $(shell python3-config --ldflags)

CXX = g++
PYTHON = python3.12
CXXFLAGS = -Ofast -std=c++14 -fPIC $(shell $(PYTHON) -m pybind11 --includes)
LDFLAGS = -shared

TARGET = nanoflann_knn$(shell $(PYTHON)-config --extension-suffix)
SRC = nanoflann_knn.cpp
OBJ = $(SRC:.cpp=.o)

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@
	
clean:
	rm -f $(TARGET)

