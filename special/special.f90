module special
  implicit none
  private
  public :: psi
  interface psi
    module procedure psi4, psi8
  end interface
contains
  !===================================================================
  ! Digamma function based on SLATEC
  ! x must be positive
  !===================================================================
  function psi4(x) result(psi)
    real(4), intent(in) :: x
    real(4) psi, s, w, y
    integer i, n
    if (x <= 0.e0) then
      psi = 0.e0
      return
    end if
    y = x
    ! use recursion to reduce argument to >= 10
    psi = 0.e0
    if (y < 10.e0) then
      n = 10 - int(y)
      do i = 1, n
        psi = psi - 1.e0 / (y + real(i-1))
      enddo
      y = y + real(n)
    endif
    ! asymptotic expansion
    w = 1.e0 / y
    s = ((-0.08333333e0 * w * w + 0.00833333e0) * w * w - &
         0.003968253968e0) * w * w
    psi = psi + log(y) - 0.5e0 * w + s
  end function psi4
  
  function psi8(x) result(psi)
    real(8), intent(in) :: x
    real(8) psi, s, w, y
    integer i, n
    if (x <= 0.d0) then
      psi = 0.d0
      return
    end if
    y = x
    ! use recursion to reduce argument to >= 10
    psi = 0.d0
    if (y < 10.d0) then
      n = 10 - int(y)
      do i = 1, n
        psi = psi - 1.d0 / (y + dble(i-1))
      enddo
      y = y + dble(n)
    endif
    ! asymptotic expansion
    w = 1.d0 / y
    s = ((-0.08333333333333333d0 * w * w + 0.008333333333333333d0) * w * w - &
         0.003968253968253968d0) * w * w
    psi = psi + log(y) - 0.5d0 * w + s
  end function psi8
end module special

