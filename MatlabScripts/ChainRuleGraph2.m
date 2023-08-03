clc;close all


nodes.names = {'$x_1$','$x_2$','$y_1$','$y_2$','$y_3$','$z$'};
nodes.x     = [1      ,1      ,2      ,2      ,2      ,3  ];
nodes.y     = [2.5    ,1.5    ,3      ,2      ,1      ,2  ];
nodes.n     = [1      ,2      ,3      ,4      ,5      ,6  ];

edges.name    = {'$dy_1/dx_1$','$dy_2/dx_1$','$dy_3/dx_1$','$dy_1/dx_2$','$dy_2/dx_2$','$dy_3/dx_2$','$dz/y_1$','$dz/y_2$','$dz/y_3$' };
edges.n1      = [1            ,1            ,1            ,2            ,2            ,2            ,3         ,4          ,5];
edges.n2      = [3            ,4            ,5            ,3            ,4            ,5            ,6         ,6          ,6];
edges.weights = 1:9;

G = digraph(edges.n1, ...
            edges.n2, ...
            edges.weights, ...
            nodes.names);

set(gcf,'position',[0 0 1400 1000]);set(gcf,'color','w');
hold on;box on;xticklabels({});set(gca,'XTick',[]);set(gca,'YTick',[]);
annotation('rectangle',[0 0 1 1 ],'Color','k',LineWidth=4);
t1 = '$z=f(y_1,y_2,y_3)$';
t2 = '$y_1,y_2,y_3=f(x_1,x_2)$';
t3 = '$\frac{\partial{z}}{\partial{x_1}}= \frac{\partial{z}}{\partial{y_1}}\frac{\partial{y_1}}{\partial{x_1}}+ \frac{\partial{z}}{\partial{y_2}}\frac{\partial{y_2}}{\partial{x_1}}+ \frac{\partial{z}}{\partial{y_3}}\frac{\partial{y_3}}{\partial{x_1}}$';

title({t1,t2,t3},'interpreter','latex','fontSize',30)
h = plot(G);
h.MarkerSize=12;
h.NodeFontSize=40;
h.EdgeColor = 'r';
h.XData = nodes.x;
h.YData = nodes.y;
h.LineWidth = 3;
h.ArrowSize = 30;
h.ArrowPosition=0.25;
h.Interpreter='latex';

ChainRulePlotEdges(nodes,edges,0.7)
exportgraphics(gcf,'imgs/chainRule2.png','Resolution',300)

