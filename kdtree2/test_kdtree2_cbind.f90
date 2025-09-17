module test_kdtree2_cbind
  use iso_c_binding
  use test_kdtree2
  implicit none
contains
  subroutine kdtree_query_cbind(k, dx, N, x_ptr, d) bind(C, name='kdtree_query_cbind')
    integer(c_int), intent(in), value :: k, dx, N
    type(c_ptr),  intent(in)          :: x_ptr
    real(c_float), intent(out)        :: d
    real(c_float), pointer :: x(:,:)
    call c_f_pointer(x_ptr, x, [dx,N])
    d = kdtree_query(k, dx, N, x)
  end subroutine kdtree_query_cbind
end module test_kdtree2_cbind

