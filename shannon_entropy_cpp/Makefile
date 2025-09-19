CXX = g++
CXXFLAGS = -Ofast -std=c++14 -fPIC $(shell python3 -m pybind11 --includes)
LDFLAGS = -shared

TARGET = shannon_entropy_cpp$(shell python3-config --extension-suffix)
SRC = shannon_entropy_cpp.cpp

all:
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET) $(LDFLAGS)

clean:
	rm -f $(TARGET)

