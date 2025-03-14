#
# Beware: this example does not work on VSC-3
#
.SUFFIXES:	.oct

run:	wrapper.oct ot.m
	echo ot.m | octave -q 2>&1 | grep -v built-in-docstrings

.cc.oct:	setup.f90
	mkoctfile $< setup.f90

clean:
	rm *.oct *.o
