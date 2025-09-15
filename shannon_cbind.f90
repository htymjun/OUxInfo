module shannon_cbind
  use iso_c_binding
  use shannon
  implicit none
contains
  function shannon_entropy_cbind(k, dx, N, x) result(H) bind(C, name='shannon_entropy_cbind')
    integer(c_int), intent(in) :: k, N, dx
    real(c_float),  intent(in) :: x(dx, N)
    real(c_float) H
    H = shannon_entropy(k, dx, N, x)
  end function shannon_entropy_cbind
end module shannon_cbind

