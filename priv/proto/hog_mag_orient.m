function [ Magnitude, Orientation ] = hog_mag_orient( Gx, Gy )

Magnitude = sqrt(Gx.^2 + Gy.^2);

[H, W] = size(Magnitude);
for i=1:H
    for j=1:W
        Or = atan2(Gy(i,j), Gx(i,j));
        if(Or < 0)
            Or = -Or + pi;
        end
        Orientation(i,j) = Or;
    end
end

end

