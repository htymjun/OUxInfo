module test_kdtree2
  use kdtree2_module
  implicit none
  private
  public :: kdtree_query
contains
  function kdtree_query(k, dx, N, x) result(d)
    integer, intent(in) :: k, dx, N
    real(4), intent(in) :: x(dx, N)
    type(kdtree2), pointer :: tree
    type(kdtree2_result), allocatable :: distance(:)
    real(4) :: d
    tree => kdtree2_create(x, dx)
    allocate(distance(k+1))
    call kdtree2_n_nearest(tree, x(:,1), k+1, distance)
    call kdtree2_sort_results(k+1, distance)
    d = distance(k+1)%dis
    deallocate(distance)
    call kdtree2_destroy(tree)
  end function kdtree_query
end module test_kdtree2

