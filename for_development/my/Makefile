CXX = g++
PYTHON = python3.12
CXXFLAGS = -Ofast -std=c++14 -fPIC $(shell $(PYTHON) -m pybind11 --includes)
LDFLAGS = -shared

TARGET = mutual_information_wrapper$(shell $(PYTHON)-config --extension-suffix)
SRC = mutual_information.cpp 
OBJ = $(SRC:.cpp=.o)

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f *.o *.so

