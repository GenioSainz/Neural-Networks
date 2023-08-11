clc;close all;clear all
 

layers    = [5 3 3];
layersTxt = {'l-1','l','l+1'};
dx        = 2;
dy        = 2;
node      = 1;
jNeuron1  = 1;
jNeuron2  = 1;
dw        = 2;

%%% NODES
%%%%%%%%%%%%%%%%%%%%%%
for i=1:length(layers)
    
    for l=1:layers(i)

           nodes.names{node} = ['$a^{',layersTxt{i},'}_',num2str(l-1),'$'];
           nodes.n(node)     = node;
           nodes.x(node)     = dx*i;
           nodes.y(node)     = (layers(i)-1)-(l-1)*dy;
           node              = node +1;
    end

end

%%% EDGES
%%%%%%%%%%%%%%%%%%%%%%
edgeCount = 1;
for i=1:layers(1)

        edges.n1(edgeCount)      = i;
        edges.n2(edgeCount)      = layers(1)+jNeuron1;
        edges.weights(edgeCount) = 1;
        edges.names{edgeCount}   = ['$w_{',num2str(jNeuron1-1),num2str(i-1),'}$'];
        edgeCount = edgeCount +1;
end
for i=1:layers(2)

        edges.n1(edgeCount)      = edgeCount;
        edges.n2(edgeCount)      = layers(1)+layers(2)+jNeuron2;
        edges.weights(edgeCount) = 1;
        edges.names{edgeCount}   = ['$w_{',num2str(jNeuron2-1),num2str(i-1),'}$'];
        edgeCount = edgeCount +1;
end

%%% PLOT ACTIVATIONS
%%%%%%%%%%%%%%%%%%%%%
t1 = '$ a^l = \sigma ( W^la^{l-1}+b^l) $';
text(4,4,t1,'interpreter','latex','fontSize',30,HorizontalAlignment = 'center')

%%% PLOT MATRIX l
%%%%%%%%%%%%%%%%%%%%%%
str = '';
for j=1:layers(2)

    for k=1:layers(1)
         
        jj  = num2str(j-1);
        kk  = num2str(k-1);
        w   = [num2str(jj),num2str(kk)];

        if k<layers(1) && j ~= jNeuron1
            newStr=['w_{',w,'} &'];

        elseif k==layers(1) && j~= jNeuron1
            newStr=['w_{',w,'}'];

        elseif k<layers(1) && j == jNeuron1
             w=['\textbf{',w,'}'];
            newStr=['\textbf{w}_{',w,'} &'];

        elseif k==layers(1) && j == jNeuron1
             w=['\textbf{',w,'}'];
            newStr=['\textbf{w}_{',w,'}'];
        end

        str    = strcat(str,newStr);
    end  
        
        str = strcat(str,' \\ ');    
end

cols = '';
for i=1:layers(1)
     cols = strcat(cols ,'c');
end
ec1 = ['$W^l=\left(\begin{array}{',cols,'}'];
ec2 = ' \end{array}\right)$';

str =strcat(ec1,str);
str =strcat(str,ec2);
text(3,min(nodes.y)-dw,str,'interpreter','latex','fontSize',18,HorizontalAlignment = 'center')

%%% PLOT MATRIX l+1
%%%%%%%%%%%%%%%%%%%%%%
str = '';
for j=1:layers(3)

    for k=1:layers(2)
         
        jj  = num2str(j-1);
        kk  = num2str(k-1);
        w   = [num2str(jj),num2str(kk)];

        if k<layers(2) && j ~= jNeuron2
            newStr=['w_{',w,'} &'];

        elseif k==layers(2) && j~= jNeuron2
            newStr=['w_{',w,'}'];

        elseif k<layers(2) && j == jNeuron2
             w=['\textbf{',w,'}'];
            newStr=['\textbf{w}_{',w,'} &'];

        elseif k==layers(2) && j == jNeuron2
             w=['\textbf{',w,'}'];
            newStr=['\textbf{w}_{',w,'}'];
        end

        str    = strcat(str,newStr);
    end  
        
        str = strcat(str,' \\ ');    
end

cols = '';
for i=1:layers(2)
     cols = strcat(cols ,'c');
end
ec1 = ['$W^l=\left(\begin{array}{',cols,'}'];
ec2 = ' \end{array}\right)$';

str =strcat(ec1,str);
str =strcat(str,ec2);
text(5,min(nodes.y)-dw,str,'interpreter','latex','fontSize',18,HorizontalAlignment = 'center')


G = digraph(edges.n1, ...
            edges.n2, ...
            edges.weights,...
            nodes.names);

set(gcf,'position',[0 0 1400 1000]);
set(gcf,'color','w');
hold on;axis off;
set(gca,'XTick',[]);set(gca,'YTick',[]);


h = plot(G);
h.MarkerSize=65;
h.NodeFontSize=1;
h.EdgeColor = 'r';
h.XData = nodes.x;
h.YData = nodes.y;
h.LineWidth = 3;
h.ArrowSize = 30;
h.ArrowPosition=0.25;
h.Interpreter='latex';

highlight(h,5,6,'EdgeColor','g')

PlotNet(nodes,edges,0.5);

rectangle('Position',[1.5 -8.5 5 14],LineWidth=0.1);

%%exportgraphics(gcf,'imgs/net535_0.png','Resolution',300)

function PlotNet(nodes,edges,t)
     
    % Edges labels
    for i=1:length(edges.n1)
         
        x1 = nodes.x(edges.n1(i));
        y1 = nodes.y(edges.n1(i));
    
        x2 = nodes.x(edges.n2(i));
        y2 = nodes.y(edges.n2(i));
        
        x =(x2-x1)*t + x1;
        y =(y2-y1)*t + y1;
    
        text(x,y,edges.names(i),'interpreter','latex',FontSize=24, ...
            HorizontalAlignment = 'center',...
            BackgroundColor=[1 1 1], ...
            Color=[0 0 0]);
    end

    % nodes labels
    for i=1:length(nodes.names)
         
        x = nodes.x(i);
        y = nodes.y(i);
    
        text(x,y,nodes.names(i),'interpreter','latex',FontSize=28, ...
          HorizontalAlignment = 'center',...
          Color=[1 1 1],FontWeight='bold');
    end

end
