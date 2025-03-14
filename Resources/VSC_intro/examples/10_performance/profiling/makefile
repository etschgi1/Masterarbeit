short:	a.out
	time ./a.out 5000
	gprof -Q -b a.out gmon.out
	time ./a.out 20000
	gprof -Q -b a.out gmon.out

long:	gmon.out
	gprof a.out gmon.out | less

gmon.out:	a.out
	time ./a.out
	
a.out:	test_gprof.c
	gcc -std=c99 -Wall -pg test_gprof.c

clean:
	rm gmon.out a.out 2>/dev/null

