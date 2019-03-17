NAME = libnn.a

COMPILER = g++

FLAGS = 
SOURCES = Neuron.cpp Connection.cpp Layer.cpp Network.cpp Network_Fit.cpp activationfunctions.cpp costfunctions.cpp utils.cpp debug.cpp

INCLUDES = -I includes

SRCS = $(addprefix sources/, $(SOURCES))
OBJS = $(SRCS:sources/%.cpp=obj/%.o)

all: dir $(NAME)

$(NAME): $(OBJS)
	ar rc $(NAME) $(OBJS)
	ranlib $(NAME)

dir:
	if [ ! -d "obj" ]; then mkdir obj; fi

obj/%.o: sources/%.cpp includes/*
	$(COMPILER) $(FLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -f $(OBJS)

fclean: clean
	rm -f $(NAME)

re: fclean all
