clear all;
close all;

PositiveSetTrain = 'F:/inz/priv/sanity_test/train/pos';
NegativeSetTrain = 'F:/inz/priv/sanity_test/train/neg';
PositiveSetTest = 'F:/inz/priv/sanity_test/test/pos';
NegativeSetTest = 'F:/inz/priv/sanity_test/test/neg';

PositivesT = dir(PositiveSetTrain);
for i=3:length(PositivesT)
    ImageFile = [PositiveSetTrain, '/', PositivesT(i, 1).name];
    Image = imread(ImageFile);
    ImageGrayscale = rgb2gray(Image);
    
    [Gx, Gy] = hog_gradient(ImageGrayscale);
    [Magnitude, Orientation] = hog_mag_orient(Gx, Gy);
    
    Bins = hog_binning(9, Magnitude, Orientation);
    Bins = Bins / (norm(Bins, 1) + 0.001);
    
    Food(i-2, :) = Bins(:);
    fprintf('%s\n', ImageFile);
end


