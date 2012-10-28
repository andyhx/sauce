function [ Features ] = slbp_threshold( Image, Threshold, Mask )

[H, W] = size(Image);
Features = [];
Nah = [];
for i=1:Mask:H-Mask
    for j=1:Mask:W-Mask
        Kernel = Image(i:i+Mask-1, j:j+Mask-1);
        Center = round(Mask/2);
        CenterVal = Kernel(Center, Center);
        Output = zeros(4*(Mask-1), 1);
        
        Edge = Kernel(1, 1:Mask-1)';
        Edge = [Edge; Kernel(1:Mask-1, Mask)];
        Edge = [Edge; Kernel(Mask, Mask:-1:2)'];
        Edge = [Edge; Kernel(Mask:-1:2, 1)];
        
        for ii=1:length(Edge)
            if abs(Edge(ii) - CenterVal) > Threshold
                Output(ii) = 1;
            end
        end
        
        Features = [Features; Output'];
    end
end


end

