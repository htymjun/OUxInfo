module shannon
  use special
  use kdtree2_module
  implicit none
  private
  public :: shannon_entropy
contains
  function shannon_entropy(k, dx, N, x) result(H)
    integer, intent(in) :: k, dx, N
    real(4), intent(in) :: x(dx, N)
    real(4), parameter  :: pi = acos(-1.e0)
    type(kdtree2), pointer :: tree
    type(kdtree2_result), allocatable :: distance(:)
    real(4) :: H, Cdx, Elog_eps = 0.e0, eps(N)
    integer i
    tree => kdtree2_create(x, dx)
    allocate(distance(k))
    do i = 1, N
      call kdtree2_n_nearest(tree, x(:,i), k, distance)
      call kdtree2_sort_results(k, distance)
      eps(i) = 2.e0 * distance(k)%dis
      Elog_eps = Elog_eps + log(eps(i))
    enddo
    deallocate(distance)
    call kdtree2_destroy(tree)
    Elog_eps = Elog_eps / real(N)
    Cdx = pi**(0.5e0 * real(dx)) / gamma(1.e0 + 0.5e0 * real(dx)) / 2.e0**real(dx)
    H = - psi(real(k)) + psi(real(N)) + log(Cdx) + real(dx) * Elog_eps
  end function shannon_entropy


  !function mutual_info(k, dx, dy, N, x, y) result(I)
  !  integer, intent(in) :: k, dx, dy, N
  !  real(4), intent(in) :: x(dx,N), y(dy,N)
  !  real(4) I
  !end function mutual_info

  
  !function KLdiv(k, dx, dy, N, x, y) result(Dkl)
  !  integer, intent(in) :: k, dx, dy, N
  !  real(4), intent(in) :: x(dx,N), y(dy,N)
  !  real(4) Dkl
  !end function KLdiv
end module shannon

