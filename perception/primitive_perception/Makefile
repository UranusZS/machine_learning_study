CC = g++
CFLAGS =  
OBJS = utils.o primitive_perception.o main.o
TARGET = primitive_perception
RM = rm -f

$(TARGET):$(OBJS)
	$(CC) -o $(TARGET) $(OBJS) $(CFLAGS)

$(OBJS):%o:%cpp
	$(CC) -c $(CFLAGS) $< -o $@

clean:
	$(RM) $(TARGET) $(OBJS)
