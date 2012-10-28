function [ Covariances ] = cov_features( Image, BlockSize )

SobelX = [-1 0 1; -2 0 2; -1 0 1];
SobelY = SobelX';
[H, W] = size(Image);

Covariances = [];
Orientations = [];
for i=1:BlockSize:H-BlockSize
    for j=1:BlockSize:W-BlockSize
        Block = Image(i:i+BlockSize-1, j:j+BlockSize-1);
        Ix = conv2(double(Block), SobelX, 'same');
        Iy = conv2(double(Block), SobelY, 'same');
        Ixx = conv2(Ix, SobelX, 'same');
        Iyy = conv2(Iy, SobelY, 'same');
        Magnitude = sqrt(Ix.^2 + Iy.^2);
        Orientation = atan2(Ix, Iy);
        
        AIx = abs(Ix);
        AIy = abs(Iy);
        AIxx = abs(Ixx);
        AIyy = abs(Iyy);
        
        Features = [AIx(:), AIy(:), AIxx(:), AIyy(:), Magnitude(:), Orientation(:)];
        Covariance = cov(Features);
        Covariances = [Covariances; Covariance(:)'];
    end
end

end

