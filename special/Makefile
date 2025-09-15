FC = gfortran
FFLAGS = -Ofast -g -Wall -fcheck=all -march=native

.SUFFIXES : .f90

%.o: %.f90
		$(COMPILE.f) $<

%.mod: %.f90 %.o
		@:

OBJ =	special.o \
      main.o

a.out: $(OBJ)
		$(FC) $(FFLAGS) -I./ $^

main.o: main.f90 special.mod

clean:
		rm -f *.mod *.o *.out

