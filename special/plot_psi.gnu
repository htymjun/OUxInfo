#! /usr/bin/gnuplot/
set tics font "Times-New-Roman, 16"
set key off
set xtics nomirror
set ytics nomirror
set mxtics 5

scipy   = "psi_scipy.d"
fortran = "psi_fortran.d"

set output "psi.png"
set term pngcairo size 480, 480

set size square
set ylabel "{/=18{/Times-New-Roman:Italic psi}}"
set format y "%.0f"
set ytics 1
set mytics 2
set format x "%.0f"
set xtics 1
set mxtics 2
set yrange [-10:3]

set xlabel "{/=18 {/Times-New-Roman:Italic x}}"
set xrange [0:10]
plot scipy   using 1:2 with lines dt 1 lw 2 lc rgb "black", \
     fortran using 1:2 with lines dt 2 lw 2 lc rgb "blue"

