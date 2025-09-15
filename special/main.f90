program main
  use special
  implicit none
  integer, parameter :: n = 100
  integer, parameter :: kind = 8
  real(kind) :: xmin = 0.01d0, xmax = 10.d0, dx
  real(kind) :: x(n), digamma(n)
  integer i
  real(kind) t1, t2
  
  call cpu_time(t1)
  dx = (xmax - xmin) / dble(n)
  x(1) = xmin
  digamma(1) = psi(x(1))
  do i = 1, n-1
    x(i+1)   = x(i) + dx
    digamma(i+1) = psi(x(i+1))
  enddo
  call cpu_time(t2)
  print *, "Elapsed time:", t2 - t1
  
  open(10, file = 'psi_fortran.d', action='write')
  write(10,*) "x     psi"
  do i = 1, n
    write(10,*) x(i), digamma(i)
  enddo
  close(10)
end program main

