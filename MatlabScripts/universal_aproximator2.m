clear all;clc;close all

t1 = [2 4 3 5]'/10;
t2 = [6 8 7 9]'/10;

w1 = 1000;
nt = 2;
k1 = [w1 0;
      w1 0;
      0  w1
      0  w1];

b1 = [t1;t2];

W1 = [];
for i=1:nt
    W1 = [W1;k1];
end

h  = 10;
W2 = h*[ones(1,4*nt);-ones(1,4*nt)]