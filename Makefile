LIB        = -L. -lfftw3 
INCLUDE    = -I.
CFLAGS     = -O3
EXEC       = NanoRET.x
CXX        = g++

${EXEC}: NanoRET.c 
	${CXX} ${CFLAGS} ${INCLUDE} ${LIB} NanoRET.c -o ${EXEC}

clean:
	rm -f *.o

%.o: $.cpp
	${CXX} -c ${CFLAGS} ${INCL} -cpp -o $*.o $<

