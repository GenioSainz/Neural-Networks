function plotEdges(nodes,edges,t)

    for i=1:length(edges.n1)
         
        x1 = nodes.x(edges.n1(i));
        y1 = nodes.y(edges.n1(i));
    
        x2 = nodes.x(edges.n2(i));
        y2 = nodes.y(edges.n2(i));
        
        x =(x2-x1)*t + x1;
        y =(y2-y1)*t + y1;
    
        text(x,y,edges.name(i),'interpreter','latex',FontSize=28, ...
          HorizontalAlignment = 'center',...
            BackgroundColor=[1 1 1]);
    end

end