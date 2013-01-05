function [ Histogram ] = slbp_binning( Features )

[H, W] = size(Features);

Histogram = zeros(W, W);

for i=1:H
    Feature = Features(i, :);
    
    InsideArc = 0;
    ArcStart = 0;
    Arcs = [];
    for j=1:W
        if Feature(j) == 1 && InsideArc == 0
            ArcStart = j;
            InsideArc = 1;
        elseif Feature(j) == 0 && InsideArc == 1
            InsideArc = 0;
            Arcs = [Arcs; [ArcStart, j-1]];
        end
    end   
    
    [ArcH, ArcW] = size(Arcs);
    if ArcH == 2 && Arcs(1,1) == 1 && Arcs(2,2) == W
        Angle = Arcs(2,1);
        Length = Arcs(2,2)-Angle+1+Arcs(1,2);
    elseif ArcH == 1
        Angle = Arcs(1,1);
        Length = Arcs(1,2)-Angle+1;
    else
        Angle = 0;
        Length = 0;
    end
    
    if Angle ~= 0 && Length ~=0
        Histogram(Angle, Length) = Histogram(Angle, Length) + 1;
    end
end


end

