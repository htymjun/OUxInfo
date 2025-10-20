module shannon_cbind
  use iso_c_binding
  use shannon
  implicit none
contains
  subroutine shannon_entropy_cbind(k, dx, N, x_ptr, H) bind(C, name='shannon_entropy_cbind')
    integer(c_int), intent(in), value :: k, dx, N
    type(c_ptr),  intent(in)          :: x_ptr
    real(c_float), intent(out)        :: H
    real(c_float), pointer :: x(:,:)
    call c_f_pointer(x_ptr, x, [dx,N])
    H = shannon_entropy(k, dx, N, x)
  end subroutine shannon_entropy_cbind
end module shannon_cbind

