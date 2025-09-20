CXX = g++
PYTHON = python3.13
CXXFLAGS = -Ofast -std=c++14 -fPIC $(shell $(PYTHON) -m pybind11 --includes)
LDFLAGS = -shared

TARGET = shannon_entropy_cpp$(shell $(PYTHON)-config --extension-suffix)
SRC = k_nearest_neighbor.cpp \
      shannon_entropy.cpp \
      kullback_leibler_divergence.cpp \
      shannon_entropy_cpp.cpp
OBJ = $(SRC:.cpp=.o)

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

kullback_leibler_divergence.o: k_nearest_neighbor.cpp
shannon_entropy_cpp.o: shannon_entropy.cpp kullback_leibler_divergence.cpp

clean:
	rm -f $(TARGET)

