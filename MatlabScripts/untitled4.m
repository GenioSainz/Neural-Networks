clear all;clc;close all


txt = '$\left(\begin{array}{cc} 1+s & 2+s\\{\textbf{1+s}} & {\textbf{1+s}}\\ \end{array}\right)$';
title(txt,'interpreter','latex',FontSize=20)

syms s

m= [s+1 s+2;s+3 s+4;s+3 s+4];

latex(m)