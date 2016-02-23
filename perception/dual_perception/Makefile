CC = g++
CFLAGS =  
OBJS = utils.o dual_perception.o main.o
TARGET = dual_perception
RM = rm -f

$(TARGET):$(OBJS)
	$(CC) -o $(TARGET) $(OBJS) $(CFLAGS)

$(OBJS):%o:%cpp
	$(CC) -c $(CFLAGS) $< -o $@

clean:
	$(RM) $(TARGET) $(OBJS)
