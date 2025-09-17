# pybind11 + nanoflann のコンパイル例

PYTHON_INCLUDE := $(shell python3 -m pybind11 --includes)
PYTHON_LIBS := $(shell python3-config --ldflags)

CXX = g++
CXXFLAGS = -O3 -std=c++11 -fPIC $(PYTHON_INCLUDE)
LDFLAGS = -shared $(PYTHON_LIBS)

TARGET = nanoflann_knn$(shell python3-config --extension-suffix)

all: $(TARGET)

$(TARGET): nanoflann_knn.cpp nanoflann.hpp
	$(CXX) $(CXXFLAGS) nanoflann_knn.cpp -o $@ $(LDFLAGS)

clean:
	rm -f $(TARGET)

