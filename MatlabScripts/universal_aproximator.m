clear all;clc;

n     = 100;
x     = linspace(0,1,n);
y     = linspace(0,1,n);
[X,Y] = meshgrid(x,x);

aL = zeros(n,n);
zL = zeros(n,n);
for i=1:length(y)
    for j=1:length(x)

        xy = [x(j),y(i)]';
        [zL(i,j),aL(i,j)] = net(xy) ;

    end
end

[a,a,a0]=net([0.2,0.1]');
[a,a,a1]=net([0.5,0.1]');
[a,a,a2]=net([0.7,0.1]');
[a,a,a3]=net([0.5,0.5]');

a0,a1,a2,a3


clf
figure(1)

subplot(2,1,1)
hold on;grid on;box on;
title('Weight Output')
mesh(X,Y,zL)
view(-40,30)
xlabel('X')
ylabel('Y')
xticks(0:0.1:1)
yticks(0:0.1:1)


subplot(2,1,2)
hold on;grid on;box on;
title('Neural Net Output')
mesh(X,Y,aL)
view(-40,30)
xlabel('X')
ylabel('Y')
xticks(0:0.1:1)
yticks(0:0.1:1)

function [z2,a2,a1] = net(xy)

    si = @(z) 1./(1+exp(-z));
    
    w1 = 1000;
    W1 =[w1 0;
         w1 0;
         0 w1;
         0 w1];
    
    b1 = -100*[4 6 3 7]';
    
    h  = 100;
    W2 = h*[1 -1 1 -1];
    b2 = -3*h/2;
    a1 = si(W1*xy+b1);
    z2 = W2*a1+b2;
    a2 = si(z2);

end
















