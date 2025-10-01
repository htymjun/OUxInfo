CXX = g++
PYTHON = python3.13
CXXFLAGS = -Ofast -std=c++14 -fPIC $(shell $(PYTHON) -m pybind11 --includes)
LDFLAGS = -shared

TARGET = shannon_entropy_cpp$(shell $(PYTHON)-config --extension-suffix)
SRC = shannon_entropy_cpp.cpp
OBJ = $(SRC:.cpp=.o)

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

shannon_entropy.hpp: point_cloud.hpp adaptor.hpp
kullback_leibler_divergence.hpp: point_cloud.hpp adaptor.hpp
mutual_information.hpp: point_cloud.hpp adaptor.hpp
shannon_entropy_cpp.o: shannon_entropy.hpp kullback_leibler_divergence.hpp mutual_information.hpp

clean:
	rm -f *.o *.so

