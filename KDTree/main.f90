program main
  use kdtree2_module
  implicit none
  real(4), allocatable :: x(:,:)
  type(kdtree2), pointer :: tree
  integer, parameter :: dim = 2
  real(4) :: query(dim)
  integer :: n = 1000
  allocate(x(dim,n))

  call random_number(x)
  tree => kdtree2_create(x, dim)
  query = (/0.5e0, 0.5e0/)
  
  block
    integer, parameter :: k = 5
    integer i
    type(kdtree2_result), allocatable :: results(:)

    allocate(results(k))
    call kdtree2_n_nearest(tree, query, k, results)
    print *, "k nearest neighbors:"
    do i = 1, k
      print *, "idx=", results(i)%idx, " dist^2=", sqrt(results(i)%dis)
    enddo
    deallocate(results)
  end block

  block
    real(4) :: r
    integer count
    r = 0.1e0
    count = kdtree2_r_count(tree, query, r)
    print *, "piints within radius r =", r, ":", count
  end block

  deallocate(x)
end program main

