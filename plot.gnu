#!/usr/bin/gnuplot

set terminal postscript enhanced color 'Helvetica' 25
set output 'ET_vs_R.eps'
a = 0.21051991721124
f(x) = a/x**6


set xlabel 'Separation (nm)'
set ylabel 'Energy Transferred (eV)'
plot 'TEST.txt' u 1:3 w l lw 4 title 'Ag - Malachite-Green', \
f(x) w l lw 4 dt 2 title 'a/x^6'
