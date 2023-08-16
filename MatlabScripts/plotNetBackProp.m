clc;close all;clear all
 
x = [0 2 6 8];
y = [0 2 4];

set(gcf,'position',[0 0 1600 800]);set(gcf,'color','w');
hold on;box on;axis off
rectangle('Position',[-1 -5 11.5 13],LineWidth=0.1);

%%% plot nodes
%%%%%%%%%%%%%%%%
SizeData = 200;
for j = 1:length(x)
    for i = 1:length(y)
        s = scatter(x(j),y(i));
        s.MarkerFaceColor = [1 1 0];
        s.MarkerEdgeColor = [0 0 0];
        s.SizeData        = SizeData ;
    end
end

s = scatter(10,2);
s.MarkerFaceColor = [1 1 0];
s.MarkerEdgeColor = [0 0 0];
s.SizeData        = SizeData ;

fontS = 34;
%%% ROW 1
%%%%%%%%%%%%
ytext = 7;
txt   = '$z^{L-1}$';
text(x(1),ytext,txt,'interpreter','latex',FontSize=fontS, ...
         HorizontalAlignment = 'center',FontWeight='bold');

txt   = '$a^{L-1}$';
text(x(2),ytext,txt,'interpreter','latex',FontSize=fontS, ...
         HorizontalAlignment = 'center',FontWeight='bold');

txt   = '$z^{L}$';
text(x(3),ytext,txt,'interpreter','latex',FontSize=fontS, ...
         HorizontalAlignment = 'center',FontWeight='bold')

txt   = '$a^{L}$';
text(x(4),ytext,txt,'interpreter','latex',FontSize=fontS, ...
         HorizontalAlignment = 'center',FontWeight='bold')

txt   = '$C $';
text(10,ytext,txt,'interpreter','latex',FontSize=fontS, ...
         HorizontalAlignment = 'center',FontWeight='bold');

txt   = '$W^{L}$';
text(4,ytext,txt,'interpreter','latex',FontSize=fontS, ...
         HorizontalAlignment = 'center');

%%% ROW 2
%%%%%%%%%%%%
ytext = 5;
txt   = '$\delta^{L}$';
text(x(3),ytext,txt,'interpreter','latex',FontSize=fontS, ...
         HorizontalAlignment = 'center',Color='k');

txt   = '$\delta^{L-1}$';
text(x(1),ytext,txt,'interpreter','latex',FontSize=fontS, ...
         HorizontalAlignment = 'center',Color='k');

%%% ROW 4
%%%%%%%%%%%%
ytext = -2;
txt = '$\frac{\partial a^{L-1}}{\partial z^{L-1}}$';
text(1,ytext,txt,'interpreter','latex',FontSize=fontS, ...
         HorizontalAlignment = 'center',FontWeight='bold');

txt = '$\frac{\partial z^{L}}{\partial a^{L-1}}$';
text(4,ytext,txt,'interpreter','latex',FontSize=fontS, ...
         HorizontalAlignment = 'center',FontWeight='bold');

txt = '$\frac{\partial a^{L}}{\partial z^{L}}$';
text(7,ytext,txt,'interpreter','latex',FontSize=fontS, ...
         HorizontalAlignment = 'center',FontWeight='bold');

txt = '$\frac{\partial C}{\partial a^{L}}$';
text(9,ytext,txt,'interpreter','latex',FontSize=fontS, ...
         HorizontalAlignment = 'center',FontWeight='bold');

%%% ROW 5
%%%%%%%%%%%%
ytext = -4;
txt = '$\sigma''(z^{L-1})$';
text(1,ytext,txt,'interpreter','latex',FontSize=fontS, ...
         HorizontalAlignment = 'center',FontWeight='bold');

txt = '$(W^L)^T$';
text(4,ytext,txt,'interpreter','latex',FontSize=fontS, ...
         HorizontalAlignment = 'center',FontWeight='bold');

txt = '$\sigma''(z^{L})$';
text(7,ytext,txt,'interpreter','latex',FontSize=fontS, ...
         HorizontalAlignment = 'center',FontWeight='bold');

txt = '$\nabla_{a^L}C$';
text(9,ytext,txt,'interpreter','latex',FontSize=fontS, ...
         HorizontalAlignment = 'center',FontWeight='bold');

%%% Edges
%%%%%%%%%%%%
scale = 0.9;
LW    = 2;
color = [1 0 0];
arrow = 0.75;
quiver(0,0,2,0,scale,LineWidth=LW,Color=color,MaxHeadSize=arrow)
quiver(0,2,2,0,scale,LineWidth=LW,Color=color,MaxHeadSize=arrow)
quiver(0,4,2,0,scale,LineWidth=LW,Color=color,MaxHeadSize=arrow)

quiver(6,0,2,0,scale,LineWidth=LW,Color=color,MaxHeadSize=arrow)
quiver(6,2,2,0,scale,LineWidth=LW,Color=color,MaxHeadSize=arrow)
quiver(6,4,2,0,scale,LineWidth=LW,Color=color,MaxHeadSize=arrow)


quiver(8,0,2,2,scale,LineWidth=LW,Color=color,MaxHeadSize=arrow)
quiver(8,2,2,0,scale,LineWidth=LW,Color=color,MaxHeadSize=arrow)
quiver(8,4,2,-2,scale,LineWidth=LW,Color=color,MaxHeadSize=arrow)


scale = 0.95;
LW    = 2;
arrow = 0.2;

color = [1 0 1];
quiver(2,4,4,0,scale,LineWidth=LW,Color=color,MaxHeadSize=arrow)
quiver(2,4,4,-2,scale,LineWidth=LW,Color=color,MaxHeadSize=arrow)
quiver(2,4,4,-4,scale,LineWidth=LW,Color=color,MaxHeadSize=arrow)

color = [0 0 1];
quiver(2,2,4,2,scale,LineWidth=LW,Color=color,MaxHeadSize=arrow)
quiver(2,2,4,0,scale,LineWidth=LW,Color=color,MaxHeadSize=arrow)
quiver(2,2,4,-2,scale,LineWidth=LW,Color=color,MaxHeadSize=arrow)

color = [0 1 0];
quiver(2,0,4,4,scale,LineWidth=LW,Color=color,MaxHeadSize=arrow)
quiver(2,0,4,2,scale,LineWidth=LW,Color=color,MaxHeadSize=arrow)
quiver(2,0,4,0,scale,LineWidth=LW,Color=color,MaxHeadSize=arrow)

exportgraphics(gcf,'imgs/netBackProp.png','Resolution',300)



