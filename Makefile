LD_LIBRARY_PATH=$(LD_LIBRARY_PATH);/usr/local/lib

all:
	gcc scanner.c structures/Vector.c -l:libiw.so -lm -o out
