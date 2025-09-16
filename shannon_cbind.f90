module shannon_cbind
  use iso_c_binding
  use shannon
  implicit none
contains
  function shannon_entropy_cbind(k, dx, N, x_ptr) result(H) bind(C, name='shannon_entropy_cbind')
    integer(c_int), intent(in)         :: k, N, dx
    real(c_float),  intent(in), target :: x_ptr(:,:)
    real(c_float) H
    real(4), pointer :: x(:,:)
    call c_f_pointer(c_loc(x_ptr), x, [dx,N])
    H = shannon_entropy(k, dx, N, x)
  end function shannon_entropy_cbind
end module shannon_cbind

