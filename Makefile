CC = g++
CFLAGS = -c -Wall -std=c++0x
SOURCES = src/Ann.cpp src/Layer.cpp src/Neurone.cpp src/main.cpp
OBJECTS = $(SOURCES:.cpp=.o)
EXECUTABLE = ann


all: $(SOURCES) $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS) 
	$(CC) $(OBJECTS) -o $@

.cpp.o:
	$(CC) -c $(CFLAGS) -o "$@" "$<"


