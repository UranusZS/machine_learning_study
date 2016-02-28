DIR_OBJ = ./obj
DIR_SRC = ./src
DIR_INC = ./include
TEM_INC = ./template

SRC = $(wildcard ${DIR_SRC}/*.cpp)
OBJ = $(patsubst %.cpp, ${DIR_OBJ}/%.o, $(notdir $(SRC))) 
INC = $(patsubst %, -I%, $(shell find $(DIR_INC) -name '[a-zA-Z0-9]*'.h))
DEPS = $(notdir $(SRC))

BIN_TARGET = knn

CC = g++
CFLAGS = -g -Wall -std=c++0x -I$(DIR_INC) -I$(TEM_INC)

${BIN_TARGET}: ${OBJ}
	$(CC) -o $@ $^ $(CFLAGS)

${DIR_OBJ}/%.o: ${DIR_SRC}/%.cpp
	@mkdir -p $(DIR_OBJ)
	$(CC) $(CFLAGS) -c $< -o $@

# 防止命名冲突例如有名为clean的文件
.PHONY:clean
clean:
	find ${DIR_OBJ} -name *.o -exec rm -rf {} \;
	rm $(BIN_TARGET)
