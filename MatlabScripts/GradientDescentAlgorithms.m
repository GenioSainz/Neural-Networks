clear all;clc;close all;format longE


set(gcf,'position',[0 0 1600 1000]);set(gcf,'color','w');
ti = tiledlayout(3,2,TileSpacing = 'compact',Padding = 'compact');

n     = 5;
size  = 0.8;
delta = 1-size;
txtFS = 22;
GD_COLOR = [0 1 0];

nexttile(1)
    hold on;box on;grid on;daspect([1 1 1]);axis([1 n 1 n]+[-1 1 -1 1]*size);
    annotation('rectangle',[0 0 1 1 ],'Color','k',LineWidth=4)
    title('$Batch-GradientDescent \rightarrow B_{SIZE}=X_{TRAIN}$','interpreter','latex','fontSize',txtFS)
    plotArray(n,size,'b');
    xy0 = 1-size/2-delta;
    wh  = n+delta;
    rectangle('Position',[xy0 xy0 wh wh],'EdgeColor',GD_COLOR,LineWidth = 1.5)
    text(6,3,'$Updates = Epoch$','interpreter','latex','fontSize',txtFS)

nexttile(3)
    hold on;box on;grid on;daspect([1 1 1]);axis([1 n 1 n]+[-1 1 -1 1]*size)
    title('$Stochastic-GradientDescent \rightarrow B_{SIZE}=1$','interpreter','latex','fontSize',txtFS)
    plotArray(n,0.5,'b');
    plotArray(n,0.8,GD_COLOR);
    text(6,3,'$Updates = Epoch*X_{train}$','interpreter','latex','fontSize',txtFS)

nexttile(5)
    hold on;box on;grid on;daspect([1 1 1]);axis([1 n 1 n]+[-1 1 -1 1]*size)
    title('$MiniBatch-GradientDescent \rightarrow B_{SIZE}>=1$','interpreter','latex','fontSize',txtFS)
    plotArray(n,0.5,'b');
    plotBach(n,0.8,GD_COLOR);
    text(6,3,'$Updates = Epoch*\frac{X_{TRAIN}}{B_{SIZE}}$','interpreter','latex','fontSize',txtFS)
    
nexttile(4)
    hold on;box on;daspect([1 1 1]);axis([-2 4 1 n]);
    title('$Update(W_i,b_i) = GradientDescent Steps$','interpreter','latex','fontSize',txtFS)

    t0 = text(1,2,'$X_{Train}$','interpreter','latex','fontSize',txtFS,'color','r');
    t0.HorizontalAlignment = 'center';

    t1=text(1,4,'$Feed_{Forward}-Back_{Propagation}$','interpreter','latex','fontSize',txtFS,'color','b');
    t1.HorizontalAlignment = 'center';

    t2=text(1,3,'$GradientDescentStep$','interpreter','latex','fontSize',txtFS,'color','g');
    t2.HorizontalAlignment = 'center';

    set(gca,'XTick',[])
    set(gca,'YTick',[])

exportgraphics(gcf,'imgs/GradientDescentAlgorithms.png','Resolution',300)
    
function plotArray(n,size,color)

    count = 0;
    for y=n:-1:1
        for x=1:n
             rectangle('Position',[x-size/2 y-size/2 size size],'EdgeColor',color,LineWidth = 1.5)
             txt = text(x,y,['X_{',num2str(count),'}'],'Color','r');
             txt.HorizontalAlignment = 'center';
             count = count +1;
        end
    end
end

function plotBach(n,size,color)

    delta = 1-size;
    x0    = 1-size/2;

    for y=n:-1:1
        y0 = y-size/2;
        rectangle('Position',[x0 y0 n-delta size],'EdgeColor',color,LineWidth = 1.5);
    end

end






