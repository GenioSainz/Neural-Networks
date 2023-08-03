clc;close all


nodes.names = {'t','u','v','x','y','z'};
nodes.n     = [1  ,2  ,3  ,4  ,5  ,6  ];
nodes.x     = [-1 ,0  ,0  ,1  ,1  ,2  ];
nodes.y     = [1  ,2  ,0  ,2  ,0  ,1  ];

edges.name    = {'du/dt','dv/dt','dx/du','dy/du','dx/dv','dy/dv','dz/dx','dz/dy'};
edges.n1      = [1      ,1      ,2      ,2      ,3      ,3      , 4     ,5];
edges.n2      = [2      ,3      ,4      ,5      ,4      ,5      , 6     ,6];
edges.weights = 1:8;

G = digraph(edges.n1, ...
            edges.n2, ...
            edges.weights, ...
            nodes.names);

set(gcf,'position',[0 0 1400 1000]);set(gcf,'color','w');
hold on;box on;xticklabels({});set(gca,'XTick',[]);set(gca,'YTick',[]);
annotation('rectangle',[0 0 1 1 ],'Color','k',LineWidth=4);
t1 = '$z=f(x,y)$';
t2 = '$x,y=f(u,v)$';
t3 = '$u,v=f(t)$';
t4 = '$\frac{\partial{z}}{\partial{u}}= \frac{\partial{z}}{\partial{x}}\frac{\partial{x}}{\partial{u}}+\frac{\partial{z}}{\partial{y}}\frac{\partial{y}}{\partial{u}}$';
t5 = '$\frac{\partial{z}}{\partial{v}}= \frac{\partial{z}}{\partial{x}}\frac{\partial{x}}{\partial{v}}+\frac{\partial{z}}{\partial{y}}\frac{\partial{y}}{\partial{v}}$';
t6 = '$\frac{\partial{z}}{\partial{t}}= \frac{\partial{z}}{\partial{u}}\frac{\partial{u}}{\partial{t}}+\frac{\partial{z}}{\partial{v}}\frac{\partial{v}}{\partial{t}}$';
title({t1,t2,t3,t4,t5,t6},'interpreter','latex','fontSize',30)

h = plot(G);
h.MarkerSize=12;
h.NodeFontSize=40;
h.EdgeColor = 'r';
h.XData = nodes.x;
h.YData = nodes.y;
h.LineWidth = 3;
h.ArrowSize = 30;
h.ArrowPosition=0.75;
h.Interpreter='latex';

ChainRulePlotEdges(nodes,edges,0.35)

exportgraphics(gcf,'imgs/chainRule1.png','Resolution',300)
%exportgraphics(gcf,'imgs/chainRulee.pdf','Resolution',300,'BackgroundColor','none','ContentType','vector')


