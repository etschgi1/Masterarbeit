SUBROUTINE SETUP(N, D, A)
        IMPLICIT NONE
        INTEGER N, D
        DOUBLE PRECISION  A(N,N)
        CALL RANDOM_NUMBER(A(1:N,1:N))
        A(1,1)=1.0D0
        A(2,2)=2.0D0
        A(3,3)=3.0D0
        A(4,4)=D
END SUBROUTINE
